#!/usr/bin/env python
import mujoco_py
import rospy
from std_msgs.msg import Float64
from rospy.numpy_msg import numpy_msg
import numpy as np
import cv2
import gym
from beginner_tutorials.msg import float_arr
import pickle
import time

env = gym.make('ManiThrow-v0')
# sim = env.get_sim()
#viewer = mujoco_py.MjViewer(sim)


class teleop:
    def __init__(self):
        self.episode = {}
        self.episode['ag'] = np.empty((1, 1, 3), np.float)
        self.episode['g'] = np.empty((1, 1, 3), np.float)
        self.episode['o'] = np.empty((1, 1, 25), np.float)
        self.episode['info_is_success'] = np.empty((1, 1, 1), np.float)
        self.episode['u'] = np.empty((1, 1, 4), np.float)
        self.count = 0


    def callback(self, data):
        #rospy.loginfo(rospy.get_caller_id() + "I heard {}".format(data.data))
        pos = np.array(data.data[:3])
        rot_pos = np.array([pos[1], pos[2], -pos[0]])
        thumb_grip = np.linalg.norm(data.data[21:23])
        middle_grip = np.linalg.norm(data.data[13:15])
        # print('thumb:{}'.format(thumb_grip))
        # print('middle:{}'.format(middle_grip))
        env.render()
        obs, reward, done, info, action = env.control(rot_pos, thumb_grip, middle_grip)
        if self.count == 0:
            self.episode['g'][0, 0, :] = obs['desired_goal']
            self.episode['o'][0, 0, :] = obs['observation']
            self.episode['ag'][0, 0, :] = obs['achieved_goal']
            self.episode['info_is_success'][0, 0, :] = info['is_success']
            self.episode['u'][0, 0, :] = action
        elif self.count < 100:
            self.episode['g'] = np.append(self.episode['g'], obs['desired_goal'][np.newaxis, np.newaxis, :], axis=1)
            self.episode['o'] = np.append(self.episode['o'], obs['observation'][np.newaxis, np.newaxis, :], axis=1)
            self.episode['ag'] = np.append(self.episode['ag'], obs['achieved_goal'][np.newaxis, np.newaxis, :], axis=1)
            self.episode['info_is_success'] = np.append(self.episode['info_is_success'],
                                                        np.array([[[info['is_success']]]]), axis=1)
            self.episode['u'] = np.append(self.episode['u'], action[np.newaxis, np.newaxis, :], axis=1)
        elif self.count == 100:
            self.episode['o'] = np.append(self.episode['o'], obs['observation'][np.newaxis, np.newaxis, :], axis=1)
            self.episode['ag'] = np.append(self.episode['ag'], obs['achieved_goal'][np.newaxis, np.newaxis, :], axis=1)

        self.count += 1

    def listener(self):
        rospy.init_node('listner', anonymous=True)

        rospy.Subscriber("chatter", float_arr, self.callback)
        rospy.spin()


if __name__ == '__main__':
    env.reset()
    op = teleop()
    start_time = time.time()
    op.listener()
    end_time = time.time()
    duration = end_time - start_time

    a_file = open("data.pkl", "wb")
    pickle.dump(op.episode, a_file)
    a_file.close()

    a_file = open("data.pkl", "rb")
    output = pickle.load(a_file)
    print(duration, output)