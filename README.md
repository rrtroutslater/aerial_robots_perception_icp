# About
This repository contains the implementation of the ICP algorithm, which is part of the final project for CS691 (Introduction to Aerial Robotics - Lecturer: Kostas Alexis) of the University of Nevada, Reno.

Authors: Russell Reinhart and Carlos Braile

# Modules

The implemented modules for this project are contained on `src/modules`. Each module is described individually below. 

## util.py

Core transformation functions.

## features.py

Implementation of features used for matching. In this project, we used the features proposed on [1].

[1] - Zhang, J. Singh, S. **LOAM: Lidar Odometry and Mapping in Real- time**. 2015.

## icp.py

Implementation of the ICP method based on [2].

[2] - Besl, P. McKay, N. **A Method for Registration of 3-D Shapes**. 1992.

# To Do

* [X] - Implement ICP
* [X] - Add point matching using features
* [X] - Integrate with ROS
* [X] - Test with real sensors.

# ROS Node

The ICP algorithm is implemented in a ROS node.  The node subscribes to pointclouds (default is for velodyne sensor) and performs the following:

* Perform ICP on the most recent cloud.
* Apply the transform recovered from ICP on the most recent cloud, to rotate it back to align with the previos cloud.
* Publish this transformed cloud.
* Publish the previous cloud for comparison.
