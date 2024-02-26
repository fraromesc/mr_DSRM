#!/usr/bin/env python
# encoding: utf-8
"""
Please open the scenes/irb140.ttt scene before running this script.

The script shows how to use the camera on the robot arm to detect a color using a simple image processing.
The mean color of the image is returned and compared to pure colors.
A simple classification in RGB is then used to place the pieces in three main destinations (without any order)

Please: beware that increasing the image resolution may lead to a slow execution.

@Authors: Arturo Gil
@Time: April 2022
"""
import numpy as np
from artelib.euler import Euler
from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.rotationmatrix import RotationMatrix
from artelib.vector import Vector
from artelib.path_planning import compute_3D_coordinates
from robots.abbirb140 import RobotABBIRB140
from robots.grippers import GripperRG2, SuctionPad
from robots.proxsensor import ProxSensor
from robots.simulation import Simulation
from robots.camera import Camera


def find_color(robot, camera):
    """
    Places the camera on top of the piece.
    Saves the image for inspection.
    Computes the image to get the mean RGB value and the color name (red, green, blue).
    """
    # position and orientation so that the camera sees the piece
    tp1 = Vector([0.6, 0.1, 0.4])
    tp2 = Vector([0.6, 0.1, 0.3])
    to = Euler([0 + np.pi/2, np.pi, 0])
    q0 = np.array([0, 0, 0, 0, np.pi / 2, 0])
    robot.moveAbsJ(q0)
    robot.moveJ(target_position=tp1, target_orientation=to)
    robot.moveL(target_position=tp2, target_orientation=to, endpoint=True)
    # capture an image and returns the closest color
    print('get_image')
    color = camera.get_color_name()
    print('Piece is: ', color)
    robot.moveL(target_position=tp1, target_orientation=to, endpoint=True)
    return color


def pick(robot, gripper):
    """
    Picks the piece from the conveyor belt.
    tp1 = Vector([0.6, 0.04, 0.23])  # approximation
    tp2 = Vector([0.6, 0.04, 0.19])  # pick
    to = Euler([0, np.pi, 0])
    gripper.open(precision=True)
    robot.moveJ(target_position=tp1, target_orientation=to, precision=True, endpoint=True)
    robot.moveL(target_position=tp2, target_orientation=to, precision=True, endpoint=True)
    gripper.close(precision=True)

    """
    q0 = np.array([0, 20, 0, 0, np.pi/2, 0])
    tp1 = Vector([0.6, 0.04, 0.26])  # approximation
    tp2 = Vector([0.6, 0.04, 0.21])  # pick
    to2 = Euler([0, np.pi, 0])
    to1 = Euler([0, np.pi, 0])
    robot.moveAbsJ(q0, endpoint=True)

    gripper.open(precision=True)
    robot.moveJ(target_position=tp1, target_orientation=to1, endpoint=True, precision=False)
    robot.moveL(target_position=tp2, target_orientation=to2, endpoint=True, precision=True, vmax=0.1)
    gripper.close(precision=True)
    robot.moveL(target_position=tp1, target_orientation=to1, endpoint=False, precision=False)

def place(robot, gripper, color, i_colors):
    """
    Places, at three different heaps the pieces
    """
    if color == 'R':
        i = i_colors[0]
        i_colors[0] = i_colors[0] + 1
        pallet_position = Vector([-0.6, -0.6, 0.15])
        pallet_orientation = Euler([0, 0, 0])
    elif color == 'G':
        i = i_colors[1]
        i_colors[1] = i_colors[1] + 1
        pallet_position = Vector([-0.15, -0.65, 0.15])
        pallet_orientation = Euler([0, 0, 0])
    else:
        i = i_colors[2]
        i_colors[2] = i_colors[2] + 1
        pallet_position = Vector([0.35, -0.65, 0.15])
        pallet_orientation = Euler([0, 0, 0])


    piece_length = 0.08
    piece_gap = 0.01
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Pose pallet left
    T0m = HomogeneousMatrix(pallet_position, pallet_orientation)
    # POSICION DE LA PIEZA i EN EL SISTEMA MÓVIL m (RELATIVA)
    pi = compute_3D_coordinates(index=i, n_x=3, n_y=3, n_z=2, piece_length=piece_length, piece_gap=piece_gap)
    # POSICION p0 INICIAL SOBRE EL PALLET
    p0 = pi + np.array([0, 0, 2.5 * piece_length])
    Tmp0 = HomogeneousMatrix(p0, Euler([0, np.pi, 0]))
    # POSICIÓN p1 EXACTA DE LA PIEZA (considerando la mitad de su lado)
    p1 = pi + np.array([0, 0, 0.5 * piece_length])
    Tmp1 = HomogeneousMatrix(p1, Euler([0, np.pi, 0]))

    # TARGET POINT 0 y 1
    T0 = T0m * Tmp0
    T1 = T0m * Tmp1

    robot.moveAbsJ(q0, precision=True)
    robot.moveJ(target_position=T0.pos(), target_orientation=T0.R(), endpoint=True)
    robot.moveL(target_position=T1.pos(), target_orientation=T1.R(), vmax=0.1, endpoint=True)
    gripper.open(precision=True)
    robot.moveL(target_position=T0.pos(), target_orientation=T0.R(), endpoint=True)


def pick_and_place():
    # Start simulation
    simulation = Simulation()
    simulation.start()
    # Connect to the robot
    robot = RobotABBIRB140(simulation=simulation)
    robot.start(base_name='/IRB140')
    # Connect to the proximity sensor
    conveyor_sensor = ProxSensor(simulation=simulation)
    conveyor_sensor.start(name='/conveyor/prox_sensor')
    # Connect to the gripper
    # gripper = GripperRG2(simulation=simulation)
    # gripper.start(name='/IRB140/RG2/RG2_openCloseJoint')
    # set the TCP of the RG2 gripper
    # robot.set_TCP(HomogeneousMatrix(Vector([0, 0, 0.19]), RotationMatrix(np.eye(3))))
    # Connect a camera to obtain images
    camera = Camera(simulation=simulation)
    camera.start(name='/IRB140/RG2/camera')

    # para usar la ventosa
    gripper = SuctionPad(simulation=simulation)
    gripper.start()
    # set the TCP of the suction pad
    robot.set_TCP(HomogeneousMatrix(Vector([0, 0.065, 0.11]), Euler([-np.pi/2, 0, 0])))


    q0 = np.array([0, 0, 0, 0, np.pi / 2, 0])
    robot.moveAbsJ(q0, endpoint=True)
    n_pieces = 48
    i_colors = np.array([0, 0, 0])
    for i in range(n_pieces):
        while True:
            if conveyor_sensor.is_activated():
                break
            robot.wait()
        color = find_color(robot, camera)
        pick(robot, gripper)
        place(robot, gripper, color, i_colors)
    simulation.stop()


if __name__ == "__main__":
    pick_and_place()

