#!/usr/bin/env python
import numpy as np
import rospy
import ros_numpy
import pcl_ros
from sensor_msgs.msg import PointCloud2, PointField
from icp import *
from pointclouds import *

icp_publisher = rospy.Publisher("transformed_cloud", PointCloud2)
source_publisher = rospy.Publisher("source_cloud", PointCloud2)
CLOUDS = []
TRANSFORMS = []
POSE = []
POSE.append(np.zeros(shape=(4,)))
POSE[0][3] = 1

def icp_callback(msg):
    """ """

    # pointcloud message -> numpy array of points
    cloud_rec_arr = pointcloud2_to_array(msg)
    points = get_xyz_points(cloud_rec_arr)

    # CLOUDS.insert(0, points)
    CLOUDS.insert(0, cloud_rec_arr)
    if len(CLOUDS) > 2:
        CLOUDS.pop()

        # get transform, apply it to the robot pose estimate (from ICP)
        transform = icp(
            get_xyz_points(CLOUDS[1])[:13000], 
            get_xyz_points(CLOUDS[0])[:13000])

        # apply all transform, create a message, publish the message
        if transform is not None:
            print 'transform found!'
            prev_pose = np.array( [POSE[0][0], POSE[0][1], POSE[0][2], 1] )
            POSE.insert( 0, np.dot(transform, prev_pose) )
            print 'POSE:', POSE[0]

            points_transformed = np.dot(
                transform, 
                to_homogeneous(get_xyz_points(CLOUDS[1])).T).T
            transformed_rec_arr = CLOUDS[1]
            transformed_rec_arr['x'][0] = points_transformed[:,0].T
            transformed_rec_arr['y'][0] = points_transformed[:,1].T
            transformed_rec_arr['z'][0] = points_transformed[:,2].T
            cloud_msg = array_to_pointcloud2(
                transformed_rec_arr,
                stamp = msg.header.stamp,
                frame_id = msg.header.frame_id)
            
            # publish transformed previous cloud, and current cloud
            icp_publisher.publish(cloud_msg)
            source_publisher.publish(msg)
        else:
            # assume no movement maybe...
            POSE.insert( 0, POSE[0])
            source_publisher.publish(msg)
            print '***NO TRANFORM FOUND***'

    return


def main():
    rospy.init_node('icp_publisher')
    
    pcl_subscriber = rospy.Subscriber(
        '/velodyne_points', PointCloud2, icp_callback )

    rospy.spin()


if __name__ == "__main__":
    main()











