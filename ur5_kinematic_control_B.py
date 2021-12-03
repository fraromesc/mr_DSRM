#!/usr/bin/env python
# encoding: utf-8
"""
Please open the scenes/ur5.ttt scene before running this script.

@Authors: Arturo Gil
@Time: April 2021

"""
import time
import sim
import sys
import numpy as np

from artelib.tools import R2quaternion, compute_w_between_R
from artelib.ur5 import RobotUR5
from artelib.scene import Scene
import matplotlib.pyplot as plt

# standard delta time for Coppelia, please modify if necessary
DELTA_TIME = 50.0/1000.0


def init_simulation():
    # global pathpositions
    # Python connect to the V-REP client
    sim.simxFinish(-1)
    clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

    if clientID != -1:
        print("Connected to remote API server")
        # stop previous simiulation
        sim.simxStopSimulation(clientID=clientID, operationMode=sim.simx_opmode_blocking)
        time.sleep(3)
        sim.simxStartSimulation(clientID=clientID, operationMode=sim.simx_opmode_blocking)
        # enable the synchronous mode
        sim.simxSynchronous(clientID=clientID, enable=True)
        # time.sleep(3)
    else:
        print("Connection not successful")
        sys.exit("Connection failed,program ended!")

    armjoints = []
    gripper = []
    objects = []
    # Get the handles of the relevant objects
    errorCode, robotbase = sim.simxGetObjectHandle(clientID, 'UR5', sim.simx_opmode_oneshot_wait)
    errorCode, end_effector = sim.simxGetObjectHandle(clientID, 'end_effector', sim.simx_opmode_oneshot_wait)

    errorCode, q1 = sim.simxGetObjectHandle(clientID, 'UR5_joint1', sim.simx_opmode_oneshot_wait)
    errorCode, q2 = sim.simxGetObjectHandle(clientID, 'UR5_joint2', sim.simx_opmode_oneshot_wait)
    errorCode, q3 = sim.simxGetObjectHandle(clientID, 'UR5_joint3', sim.simx_opmode_oneshot_wait)
    errorCode, q4 = sim.simxGetObjectHandle(clientID, 'UR5_joint4', sim.simx_opmode_oneshot_wait)
    errorCode, q5 = sim.simxGetObjectHandle(clientID, 'UR5_joint5', sim.simx_opmode_oneshot_wait)
    errorCode, q6 = sim.simxGetObjectHandle(clientID, 'UR5_joint6', sim.simx_opmode_oneshot_wait)

    errorCode, gripper1 = sim.simxGetObjectHandle(clientID, 'RG2_openCloseJoint', sim.simx_opmode_oneshot_wait)

    # errorCode, gripper1 = sim.simxGetObjectHandle(clientID, 'MicoHand_fingers12_motor1', sim.simx_opmode_oneshot_wait)
    # errorCode, gripper2 = sim.simxGetObjectHandle(clientID, 'MicoHand_fingers12_motor2', sim.simx_opmode_oneshot_wait)

    errorCode, target = sim.simxGetObjectHandle(clientID, 'target', sim.simx_opmode_oneshot_wait)

    # errorCode, sphere = sim.simxGetObjectHandle(clientID, 'Sphere', sim.simx_opmode_oneshot_wait)

    armjoints.append(q1)
    armjoints.append(q2)
    armjoints.append(q3)
    armjoints.append(q4)
    armjoints.append(q5)
    armjoints.append(q6)

    gripper.append(gripper1)
    # gripper.append(gripper2)

    # objects.append(sphere)
    robot = RobotUR5(clientID=clientID, wheeljoints=[],
                    armjoints=armjoints, base=robotbase,
                    end_effector=end_effector, gripper=gripper, target=target)
    scene = Scene(clientID=clientID, objects=objects)
    return robot, scene


def plot_trajectories(q_rs):
    q_rs = np.array(q_rs)
    plt.figure()

    for i in range(0, 6):
        plt.plot(q_rs[:, i], label='q' + str(i + 1))
    plt.legend()
    plt.show(block=True)









def main_loop():
    robot, scene = init_simulation()

    # TRY TO REACH DIFFERENT TARGETS
    target_positions = [[0.5, 0.27, 0.25],
                        [0.61, 0.27, 0.25],
                        [0.3, -0.45, 0.5],
                        [0.3, -0.45, 0.25]]
    target_orientations = [[0, np.pi / 2, 0],
                           [0, np.pi / 2, 0],
                           [0, np.pi / 2, 0],
                           [0, np.pi, 0]]

    q0 = np.array([-np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4, -np.pi/4, -np.pi/4])
    q0 = np.zeros(6)

    # plan trajectories
    [q1_path, _] = robot.inversekinematics(target_position=target_positions[0],
                                           target_orientation=target_orientations[0], q0=q0)
    [q2_path, _] = robot.inversekinematics(target_position=target_positions[1],
                                           target_orientation=target_orientations[1], q0=q1_path[-1])
    [q3_path, _] = robot.inversekinematics(target_position=target_positions[2],
                                           target_orientation=target_orientations[2], q0=q2_path[-1])
    [q4_path, _] = robot.inversekinematics(target_position=target_positions[3],
                                           target_orientation=target_orientations[3], q0=q3_path[-1])

    robot.set_target_position_orientation(target_positions[3], target_orientations[3])

    # set initial position of robot
    robot.set_arm_joint_target_positions(q0)
    # execute trajectories
    robot.open_gripper()
    robot.follow_q_trajectory(q1_path, 3)
    robot.follow_q_trajectory(q2_path, 3)
    robot.close_gripper(wait=True)
    robot.follow_q_trajectory(q3_path, 3)
    robot.follow_q_trajectory(q4_path, 3)
    #
    plot_trajectories(q1_path)
    robot.stop_arm()
    scene.stop_simulation()


if __name__ == "__main__":
    main_loop()
