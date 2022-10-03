#!/usr/bin/env python
# encoding: utf-8
"""
Please open the scenes/irb140.ttt scene before running this script.

@Authors: Arturo Gil
@Time: April 2022
"""
import numpy as np
from artelib.euler import Euler
from sceneconfig.scene_configs import init_simulation_ABBIRB140


def inverse_kinematics(robot, target_position, target_orientation):
    """
    Find q that allows the robot to achieve the specified target position and orientaiton
    CAUTION: target_orientation must be specified as a quaternion.

    caution: closest tooooo
    """
    q = robot.inversekinematics(target_position, target_orientation)
    return q


def get_closest_to(qa, qb):
    """
    From a column wise list of solutions in qa, find the closest to qb.
    """
    distances = []
    for i in range(8):
        d = np.linalg.norm(qa[:, i]-qb)
        distances.append(d)
    distances = np.array(distances)
    distances = np.nan_to_num(distances, nan=np.inf)
    idx = np.argmin(distances)
    return qa[:, idx]


def pick(robot):
    target_positions = [[0.6, 0.25, 0.4],  # approximation
                        [0.6, 0.25, 0.37]]
    target_orientations = [[0, np.pi, 0],
                           [0, np.pi, 0]]

    q0 = np.array([0, 0, 0, 0, np.pi / 2, 0])
    q1 = inverse_kinematics(robot=robot, target_position=target_positions[0],
                            target_orientation=Euler(target_orientations[0]))
    q2 = inverse_kinematics(robot=robot, target_position=target_positions[1],
                            target_orientation=Euler(target_orientations[1]))

    q1 = get_closest_to(q1, q0)
    q2 = get_closest_to(q2, q0)
    robot.open_gripper(precision=True)
    robot.set_joint_target_positions(q1, precision=True)
    robot.set_joint_target_positions(q2, precision=True)
    robot.close_gripper(precision=True)
    robot.set_joint_target_positions(q1, precision=True)


def place(robot, i):
    base_target_position = [0.2, -0.65, 0.38] # pallet base position
    base_target_orientation = [0, np.pi, 0]
    q0 = np.array([0, 0, 0, 0, np.pi / 2, 0])
    n = 3 # n rows n columns
    k1 = i % n
    k2 = np.floor((i / n) % n)
    k3 = np.floor(i / (n * n))
    target_position = base_target_position + k1*np.array([-0.052, 0, 0]) + k2*np.array([0, 0.052, 0]) + k3*np.array([0, 0, 0.05])
    q3 = inverse_kinematics(robot=robot, target_position=target_position,
                            target_orientation=Euler(base_target_orientation))
    target_position = target_position + np.array([0, 0, -0.03])
    q4 = inverse_kinematics(robot=robot, target_position=target_position,
                            target_orientation=Euler(base_target_orientation))
    q3 = get_closest_to(q3, q0)
    q4 = get_closest_to(q4, q0)

    robot.set_joint_target_positions(q3, precision=True)
    robot.set_joint_target_positions(q4, precision=True)
    robot.open_gripper(precision=True)
    robot.set_joint_target_positions(q3, precision=True)


def pick_and_place():
    robot = init_simulation_ABBIRB140()
    q0 = np.array([0, 0, 0, 0, np.pi / 2, 0])

    robot.wait(60)

    for i in range(64):
        robot.wait(25)
        robot.set_joint_target_positions(q0, precision=True)
        pick(robot)
        robot.set_joint_target_positions(q0, precision=True)
        place(robot, i)

    # Stop arm and simulation
    robot.stop_arm()
    robot.stop_simulation()
    robot.plot_trajectories()


if __name__ == "__main__":
    pick_and_place()

