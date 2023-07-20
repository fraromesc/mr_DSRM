#!/usr/bin/env python
# encoding: utf-8
"""
Please open the scenes/irb140.ttt scene before running this script.

@Authors: Arturo Gil
@Time: April 2022
"""
import numpy as np
from robots.abbirb140 import RobotABBIRB140
from robots.simulation import Simulation


if __name__ == "__main__":
    # Start simulation
    simulation = Simulation()
    simulation.start()
    # Connect to the robot
    robot = RobotABBIRB140(simulation=simulation)
    robot.start()

    q1 = np.array([-np.pi/4, np.pi/8, np.pi/8, np.pi/4, -np.pi/4, np.pi/4])
    q2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q3 = np.array([np.pi/8, 0, -np.pi/4, 0, -np.pi/4, 0])

    robot.moveAbsJ(q1, endpoint=True)
    robot.moveAbsJ(q2, endpoint=True)
    robot.moveAbsJ(q3, endpoint=True)
    # Stop arm and simulation
    simulation.stop()

