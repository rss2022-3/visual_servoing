#!/usr/bin/env python
import rospy
import numpy as np

from visual_servoing.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped
from utilities.controllers import PurePursuit
from utilities.Trajectory import LinearTrajectory

class ParkingController():
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        rospy.Subscriber("/relative_cone", ConeLocation,
            self.relative_cone_callback)

        DRIVE_TOPIC = rospy.get_param("~drive_topic") # set in launch file; different for simulator vs racecar
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC,
            AckermannDriveStamped, queue_size=10)
        self.error_pub = rospy.Publisher("/parking_error",
            ParkingError, queue_size=10)

        self.parking_distance = .75 # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0
    
        self.pursuit = PurePursuit(0.325)
    
    def v_function(self, v_desired, traj):
        #adaptive velocity function
        Lfw, lfw = 0, 0
        v_desired = abs(v_desired)
        if v_desired < 2:
            Lfw = v_desired
        elif v_desired >=2 and v_desired < 6:
            Lfw = 1.5*v_desired
        else:
            Lfw = 12
        return Lfw, lfw
    
    def drive(self, theta, speed, theta_dot = None,  acceleration = None):
        """
            Takes in steering and speed commands for the car.
            :param theta: steering angle [rad]. right sided
            :type theta: float
            :param theta_dot: steering angular velocity [rad/s]
            :type theta_dot: float
            :param speed: speed in [m/s]
            :type speed: float

            :param speed: speed in [m/s]
            :type speed: float
        """
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = rospy.get_rostime()
        ack_msg.header.frame_id = 'base_link'
        ack_msg.drive.steering_angle = np.clip(theta, -0.34, 0.34)
        ack_msg.drive.speed = speed
        if not theta_dot is None:
            ack_msg.drive.steering_angle_velocity = theta_dot
        if not acceleration is None:
            ack_msg.drive.acceleration = acceleration
        return ack_msg
    

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos

        dist = np.sqrt(self.relative_x**2 + self.relative_y**2) - self.parking_distance
        heading = np.arctan2(self.relative_y, self.relative_x)

        x_d = np.cos(heading)*dist
        y_d = np.sin(heading)*dist
        #################################

        #generate trajectory 
        traj_knots = np.array([[0,0],
                              [x_d, y_d],
                              [x_d, y_d]])
        t_breaks = np.array([0,2,2.1])

        trj_d = LinearTrajectory(t_breaks, traj_knots)

        steer, speed = self.pursuit.adaptiveControl(trj_d, self.v_function)
        drive_cmd = self.drive(steer, speed)

        #################################

        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        #################################
        dist = np.sqrt(self.relative_x**2 + self.relative_y**2) - self.parking_distance
        heading = np.arctan2(self.relative_y, self.relative_x)

        x_d = np.cos(heading)*dist
        y_d = np.sin(heading)*dist

        error_msg.x_error = x_d
        error_msg.y_error = y_d
        error_msg.distance_error = np.linalg.norm([x_d, y_d])
        #################################
        
        self.error_pub.publish(error_msg)

if __name__ == '__main__':
    try:
        rospy.init_node('ParkingController', anonymous=True)
        ParkingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
