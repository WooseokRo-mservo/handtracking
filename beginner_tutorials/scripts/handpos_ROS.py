#!/usr/bin/env python
import gym
import rospy
import numpy as np
from beginner_tutorials.msg import float_arr

class handpos_ROS():
    def __init__(self):
        self.env = gym.make('Handpos-v1')
        self.state = self.env.reset()
        self.trg_state = self.state[:-7]
        self.old_state = np.zeros((27,))
        self.origin_before = np.zeros(7)
        self.alpha = 0.8

    def callback(self, data):
        state = np.array(data.data)
        state = 0.5*self.old_state + 0.5*state
        origin_state = state[:7]
        pinky_state = state[7:11] * 1.2
        middle_state = state[11:15]* 1.2
        ring_state = state[15:19]* 1.2
        thumb_state = -state[19:23]* 1.2
        index_state = state[23:27]* 1.2
        pinky_state[1] *= -1
        middle_state[1] *= -1
        ring_state[1] *= -1
        thumb_state[0] *= -1
        index_state[1] *= -1

        origin = self.origin_before * self.alpha + origin_state * (1 - self.alpha)

        self.trg_state = np.concatenate((origin, thumb_state, index_state, middle_state, ring_state, pinky_state))
        self.env.set_wrist(self.trg_state[:7])
        self.origin_before = origin

    def listner(self):
        rospy.init_node('handpos', anonymous=True)
        rospy.Subscriber("chatter", float_arr, self.callback)
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            self.state, _, _, _ = self.env.step(self.trg_state[7:])
            print(self.state[:7])
            self.env.render()
            rate.sleep()

if __name__ == '__main__':
    hand = handpos_ROS()
    hand.listner()
    hand.env.close()