#!/usr/bin/env python
import caffe
import numpy as np
import os
import cv2
import rospy
from std_msgs.msg import String
import pyrealsense2 as rs
from beginner_tutorials.msg import float_arr

# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
depth_to_color_extrincs = depth_profile.get_extrinsics_to(color_profile)
print(depth_to_color_extrincs)
depth_intrinsics = depth_profile.get_intrinsics()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

data_path = '/media/wooseok/로컬 디스크/SynthHands_Release/female_noobject/seq01/cam01/01/'
depth_suffix = '_depth.png'
colored_depth_suffix = '_color_on_depth.png'
caffe.set_device(0)
caffe.set_mode_gpu()
net1 = caffe.Net('/home/wooseok/ICCV2017_Model/HALNet/HALNet_deploy.prototxt', '/home/wooseok/ICCV2017_Model/HALNet/HALNet_weights.caffemodel', caffe.TEST)
# net2 = caffe.Net('/home/wooseok/ICCV2017_Model/JORNet/JORNet_deploy.prototxt', '/home/wooseok/ICCV2017_Model/JORNet/JORNet_weights.caffemodel', caffe.TEST)
net3 = caffe.Net('/home/wooseok/GanHandsAPI/data/CNNClassifier/rgb-crop_232_FINAL_synth+GAN_ProjLayer/merged_net.prototxt',
                 '/home/wooseok/GanHandsAPI/data/CNNClassifier/rgb-crop_232_FINAL_synth+GAN_ProjLayer/merged_snapshot_iter_300000.caffemodel', caffe.TEST)
#net4 = caffe.Net('/home/wooseok/openpose/models/hand/pose_deploy.prototxt', '/home/wooseok/openpose/models/hand/pose_iter_102000.caffemodel', caffe.TEST)
invlaid_depth = 32001
maximum_depth = 1000
# depth_intrinsics = np.array([[383.076 / 2,       0, 319.648 / 2],
#                              [0,  383.076 / 2, 234.626 / 2],
#                              [0,       0,       1]])
depth_intrinsics = np.array([[475.62/2,       0, 311.125/2],
                             [0,  475.62/2, 245.965/2],
                             [0,       0,       1]])

depth_intrinsics_inv = np.linalg.inv(depth_intrinsics)

width = 320
height = 240
crop_size = 128
num_joints = 21
o1_parent = [0, 0, 1, 2, 3, 0, 5, 6, 7,  0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
root_id = 9
crop_size_factor = 1.15 * 23500

image_list = sorted(os.listdir(data_path))
image_list_depth = [img for img in image_list if img.endswith(depth_suffix) and not img.endswith(colored_depth_suffix)]
image_list_color = [img for img in image_list if img.endswith(colored_depth_suffix)]
num_images = len(image_list_depth)
if num_images != len(image_list_color):
    raise ValueError('Unequal amount of depth and color depth images')
i=0
uv_root_ = 0
alpha = 0.01
try:
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(100)  # 10hz
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image_full = np.int16(np.asanyarray(aligned_depth_frame.get_data()))
        color_image_full = np.asanyarray(color_frame.get_data())
        depth_image = np.transpose(cv2.resize(depth_image_full, (width, height), interpolation=cv2.INTER_NEAREST))
        depth_image = np.where(depth_image > maximum_depth, maximum_depth, depth_image)
        depth_image = depth_image / maximum_depth
        depth_image = depth_image[:, :, np.newaxis]
        grey_color = 0
        depth_image_3d = np.dstack((depth_image_full, depth_image_full, depth_image_full))  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > maximum_depth) | (depth_image_3d <= 0), grey_color, color_image_full)
        color_image = np.transpose(cv2.resize(bg_removed, (width, height), interpolation=cv2.INTER_LINEAR) / 255, [1, 0, 2])
        color_image_ = np.transpose(cv2.resize(color_image_full, (width, height), interpolation=cv2.INTER_LINEAR) / 255, [1, 0, 2])
        data = np.concatenate((depth_image, color_image), axis=2)
        data = np.transpose(data, [2, 1, 0])
        data = data[np.newaxis, :, :, :]
        net1.blobs['input'].data[...] = data
        p = net1.forward()
        p_heatmap_2D = p['heatmap_final']

        heatmap_root = p_heatmap_2D[0, root_id, :, :]
        heatmap_sized = cv2.resize(heatmap_root, (width, height), interpolation=cv2.INTER_CUBIC)
        maxLoc = np.unravel_index(np.argmax(heatmap_sized), heatmap_sized.shape)

        uv_root = np.resize(maxLoc, [2, 1])

        if uv_root_ is 0:
            uv_root_ = uv_root
        else:
            uv_root = alpha * uv_root + (1 - alpha) * uv_root_
            uv_root = uv_root.astype(np.int16)

        start_uv = np.maximum(uv_root - 2, np.zeros([2, 1], dtype=np.int8))
        end_uv = np.maximum(uv_root + 2, np.array([[height], [width]], dtype=np.int8))
        if end_uv[0, 0] >= 240:
            end_uv[0, 0] = 239
        elif end_uv[1, 0] >= 320:
            end_uv[1, 0] = 319

        mean_depth = 0
        num_valid = 0
        for h in range(start_uv[0, 0], end_uv[0, 0] + 1):
            for w in range(start_uv[1, 0], end_uv[1, 0] + 1):
                if depth_image[w, h] != 1.0:
                    mean_depth = mean_depth + maximum_depth * depth_image[w, h]
                    num_valid = num_valid + 1
        if num_valid > 0 and mean_depth > 0:
            mean_depth = mean_depth / num_valid
            radCrop = int(np.round(crop_size_factor  * (1 / mean_depth)))
        else:
            chk = np.transpose(color_image_, [1, 0, 2])
            cv2.imshow('test', chk)
            cv2.waitKey(1)
            continue

        normPoint = np.matmul(depth_intrinsics_inv, np.array([[maxLoc[1]], [maxLoc[0]], [1]]))
        normPoint = (normPoint / normPoint[2]) * mean_depth
        radcrop_size = int(2 * radCrop + 1)
        crop_depth = np.ones([radcrop_size, radcrop_size])
        crop_color = np.zeros([radcrop_size, radcrop_size, 3])

        norm_z = mean_depth / maximum_depth
        uv_bb_start = uv_root - np.array([[radCrop], [radCrop]])
        uv_bb_end = uv_root + np.array([[radCrop], [radCrop]]) + 1

        target_uv_start = np.maximum(np.zeros([2, 1], dtype=np.int8), -uv_bb_start)
        target_uv_end = np.minimum(np.array([[radcrop_size], [radcrop_size]]), np.array([[radcrop_size], [radcrop_size]]) - uv_bb_end + np.array([[height], [width]]))

        crop_depth[target_uv_start[1, 0]: target_uv_end[1, 0], target_uv_start[0, 0]: target_uv_end[0, 0]] = \
            depth_image[np.maximum(0, uv_bb_start[1, 0]): np.minimum(width, uv_bb_end[1, 0]),
            np.maximum(0, uv_bb_start[0, 0]): np.minimum(height, uv_bb_end[0, 0]), 0]
        """for ICCV2017"""
        # crop_color[target_uv_start[1, 0]: target_uv_end[1, 0], target_uv_start[0, 0]: target_uv_end[0, 0]] = \
        #     color_image[np.maximum(0, uv_bb_start[1, 0]): np.minimum(width, uv_bb_end[1, 0]),
        #     np.maximum(0, uv_bb_start[0, 0]): np.minimum(height, uv_bb_end[0, 0]), :]
        """for CVPR2018"""
        crop_color[target_uv_start[1, 0]: target_uv_end[1, 0], target_uv_start[0, 0]: target_uv_end[0, 0]] = \
            color_image_[np.maximum(0, uv_bb_start[1, 0]): np.minimum(width, uv_bb_end[1, 0]),
            np.maximum(0, uv_bb_start[0, 0]): np.minimum(height, uv_bb_end[0, 0]), :]

        crop_image_sized = cv2.resize(crop_depth, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
        crop_image_sized = np.where(crop_image_sized != 1.0, crop_image_sized - norm_z, crop_image_sized)
        crop_image_sized = crop_image_sized[:, :, np.newaxis]
        crop_color_sized = cv2.resize(crop_color, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        """for CVPR2018"""
        crop_data = np.transpose(crop_color_sized, [2, 1, 0])
        crop_data = crop_data[np.newaxis, :, :, :]
        cv2.imshow('check', crop_color)
        net3.blobs['color_crop'].data[...] = crop_data
        p = net3.forward()
        """for ICCV2017"""
        # crop_data = np.concatenate((crop_image_sized, crop_color_sized), axis=2)
        # crop_data = np.transpose(crop_data, [2, 1, 0])
        # crop_data = crop_data[np.newaxis, :, :, :]
        # net2.blobs['input'].data[...] = crop_data
        # p = net2.forward()
        p_2D = p['heatmap_final']
        p_rel3D = p['joints3D_final_vec']
        pred_3D = np.zeros([3, num_joints])
        proj_pred3D = np.zeros([3, num_joints])
        orig_uv = np.zeros([2, num_joints])
        tmp = np.transpose(color_image_, [1, 0, 2])
        cv2.rectangle(tmp, (uv_bb_start[1], uv_bb_start[0]), (uv_bb_end[1], uv_bb_end[0]), (0, 255, 0))
        conf = np.max(p_2D[0, 9, :, :])
        if conf < 0.1:
            chk = np.transpose(color_image_, [1, 0, 2])
            cv2.imshow('test', chk)
            cv2.waitKey(1)
            continue
        print('conf:')
        print(conf)
        for j in range(num_joints):
            p_j_3D = 100 * np.reshape(p_rel3D[0, j, 0, :], [3, 1]) + normPoint
            if j == 0:
                root_3d = p_j_3D
            p_j_3D_norm = p_j_3D / p_j_3D[2]
            proj_3D = np.matmul(depth_intrinsics, p_j_3D_norm)
            pred_3D[:, j] = p_j_3D[:, 0]
            proj_pred3D[:, j] = proj_3D[:, 0]
            heatmap_j = cv2.resize(p_2D[0, j, :, :], (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
            max_heat_pos = np.unravel_index(np.argmax(heatmap_j), heatmap_j.shape)
            orig_uv[:, j] = np.squeeze(uv_bb_start) + np.array(max_heat_pos) * ((2 * radCrop + 1) / crop_size)

            if j == 0:
                continue
            elif j % 4 == 1:
                tmp_draw = cv2.line(tmp, (int(orig_uv[1, j]), int(orig_uv[0, j])), (int(orig_uv[1, 0]), int(orig_uv[0, 0])), (0, 0, 255))
                tmp_draw = cv2.line(tmp, (int(proj_pred3D[0, j]), int(proj_pred3D[1, j])), (int(proj_pred3D[0, 0]), int(proj_pred3D[1, 0])), (0, 255, 0))
            else:
                tmp_draw = cv2.line(tmp, (int(orig_uv[1, j]), int(orig_uv[0, j])), (int(orig_uv[1, j-1]), int(orig_uv[0, j-1])), (0, 0, 255))
                tmp_draw = cv2.line(tmp, (int(proj_pred3D[0, j]), int(proj_pred3D[1, j])), (int(proj_pred3D[0, j-1]), int(proj_pred3D[1, j-1])), (0, 255, 0))
            if j == 9:
                tmp_draw = cv2.circle(tmp, (int(orig_uv[1, 9]), int(orig_uv[0, 9])), 2, (0, 0, 255))
                tmp_draw = cv2.circle(tmp, (int(proj_3D[0, 0]), int(proj_3D[1, 0])), 2, (0, 255, 0))

        prt_output = str(root_3d[0, 0]) + " " + str(root_3d[1, 0]) + " " + str(root_3d[2, 0])
        rospy.loginfo(prt_output)
        pub.publish(prt_output)
        rate.sleep()
        cv2.imshow('test', tmp_draw)
        cv2.waitKey(1)
except rospy.ROSInterruptException:
    pass

finally:
    pipeline.stop()

