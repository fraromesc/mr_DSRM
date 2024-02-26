#!/usr/bin/env python
# encoding: utf-8
"""
Please open the scenes/more/irb140_paint_application.ttt scene before running this script.

@Authors: Arturo Gil
@Time: November 2022
"""
import numpy as np

from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.rotationmatrix import RotationMatrix
from artelib.vector import Vector
from robots.abbirb140 import RobotABBIRB140
from robots.simulation import Simulation


def paint():
    simulation = Simulation()
    simulation.start()
    robot = RobotABBIRB140(simulation=simulation)
    robot.start()
    robot.set_TCP(HomogeneousMatrix(Vector([0, 0, 0.2]), RotationMatrix(np.eye(3))))

    q0 = np.array([0, 0, 0, 0, 0, 0])
    robot.moveAbsJ(q_target=q0, precision=True)


    target_positions = [[0.4, 0.5, 0.1],
                        [0.4, 0.5, 0.6],
                        [0.4, 0.5, 1.2]]

    target_orientations = [[0, 0, 0],
                           [0, np.pi/3, 0],
                           [0, 2*np.pi/3, 0]]
    for i in range(10):
        for i in range(len(target_positions)):
            robot.moveJ(target_position=target_positions[i],
                    target_orientation=target_orientations[i])
            robot.wait(10)

    simulation.stop()


if __name__ == "__main__":
    paint()

