#!/usr/bin/env python
# encoding: utf-8
"""
Please open the scenes/ur5.ttt scene before running this script.

@Authors: Arturo Gil
@Time: April 2021
"""
from robots.simulation import Simulation
from robots.ur5 import RobotUR5
from robots.camera import Camera
import numpy as np


def capture_image():
    simulation = Simulation()
    clientID = simulation.start()
    robot = RobotUR5(clientID=clientID)
    robot.start()
    camera = Camera(clientID=clientID)
    camera.start()

    q0 = np.array([-np.pi/4, -np.pi/4, np.pi/4, np.pi/4, -np.pi/2, np.pi/2])
    robot.set_joint_target_positions(q0, precision=True)
    image = camera.get_image()
    camera.save_image('image.png')

    simulation.stop()


if __name__ == "__main__":
    capture_image()
