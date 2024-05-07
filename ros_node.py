import os

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
import nvdiffrast.torch as dr
from std_msgs.msg import String
import json
from image_geometry import PinholeCameraModel
import cv2
import numpy as np
import trimesh
import sys
from scipy.spatial.transform import Rotation
import yaml

def imgmsg_to_cv2_rgb(img_msg):
    '''
    https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
    '''
    dtype = np.dtype('uint8')  # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
                              # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                              dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def imgmsg_to_cv2_depth(img_msg):
    '''
    https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
    '''
    dtype = np.dtype('uint16')  # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width),
                              dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv


class DishPoseEstimator:

    def __init__(self, img_topic, aligned_depth_topic, cam_info_topic, model_directory,
                 model_up_axis=1, world_plane=(0,0,1)):

        # Params regarding semantic description of models/outputpose
        self.model_up_axis = model_up_axis      # Which axis in the model corresponds to the up direction?
        self.world_plane = world_plane          # Normal vector defining side-to-side movement

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        # Load in models to be analyzed
        self.models = {}
        with open(os.path.join(model_directory, 'semantics.yaml'), 'r') as fh:
            self.model_semantics = yaml.safe_load(fh)

        for file in os.listdir(model_directory):
            if 'IGNORE' in file or 'semantics' in file:
                continue

            if file not in self.model_semantics:
                print("{} isn't in the semantics file, will not attempt to load".format(file))

            model_path = os.path.join(model_directory, file)
            try:
                self.models[model_path] = self.process_model_file(model_path)
            except ValueError:
                continue
        if not self.models:
            raise ValueError("The specified model directory did not contain any valid models!")

        # State variables
        self.last_rgb = None
        self.last_rgb_ts = None
        self.last_depth = None
        self.camera = PinholeCameraModel()
        self.camera.fromCameraInfo(rospy.wait_for_message(cam_info_topic, CameraInfo, rospy.Duration(5.0)))

        self.estimator = None
        self.last_pose = None
        self.active_model = None

        # ROS utils

        self.image_sub = rospy.Subscriber(img_topic, Image, queue_size=1, callback=self.handle_image_message)
        self.depth_sub = rospy.Subscriber(aligned_depth_topic, Image, queue_size=1, callback=self.handle_depth_message)
        self.request_sub = rospy.Subscriber('/pose_tracking_request', String, queue_size=1, callback=self.handle_request)
        self.pose_pub = rospy.Publisher('/tracked_pose', PoseStamped, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.2), self.update_tracking)

    def reorient_pose(self, tf):

        # Reorients the computed pose based on the semantics
        up_axis = tf[:3,self.model_up_axis]
        side_axis_raw = self.world_plane
        fwd_axis = np.cross(side_axis_raw, up_axis)
        fwd_axis = fwd_axis / np.linalg.norm(fwd_axis)
        side_axis = np.cross(up_axis, fwd_axis)

        new_tf = tf.copy()
        new_tf[:3,0] = fwd_axis
        new_tf[:3,1] = side_axis
        new_tf[:3,2] = up_axis

        # Use the model semantics to transform the frame to the edge of the cup
        active_model = os.path.split(self.active_model)[-1]
        info = self.model_semantics[active_model]
        height_offset = info.get('height_offset', 0.0)
        if info['shape'] == 'circular':
            offset_homog = np.array([info['radius'], 0, height_offset, 1.0])
            new_tf[:3,3] = (new_tf @ offset_homog)[:3]
        else:
            raise ValueError("Unknown model semantic value {}".format(info['shape']))

        return new_tf

    def publish_tf_pose(self, tf, stamp=None, reorient=True):

        if reorient:
            tf = self.reorient_pose(tf)

        self.pose_pub.publish(self.convert_tf_mat_to_pose(tf, stamp=stamp))

    def convert_tf_mat_to_pose(self, tf, stamp=None):

        pose = PoseStamped()
        pose.header.frame_id = self.camera.tf_frame
        if stamp is not None:
            pose.header.stamp = stamp

        pose.pose.position = Point(*tf[:3,3])
        pose.pose.orientation = Quaternion(*Rotation.from_matrix(tf[:3,:3]).as_quat())
        return pose


    @staticmethod
    def process_model_file(file):
        mesh = trimesh.load(file)
        if isinstance(mesh, trimesh.Scene):

            geometries = list(mesh.geometry.values())
            mesh = max(geometries, key=lambda g: len(g.vertices))

        return mesh

    def handle_image_message(self, msg):
        self.last_rgb = imgmsg_to_cv2_rgb(msg)
        self.last_rgb_ts = msg.header.stamp

    def handle_depth_message(self, msg):
        depth = imgmsg_to_cv2_depth(msg) / 1000.0
        depth = cv2.resize(depth, (self.camera.width, self.camera.height), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.1) | (depth >= 5.0)] = 0
        self.last_depth = depth

    def reset(self):
        self.estimator = None
        self.last_pose = None
        self.active_model = None

    def handle_request(self, msg):

        data = json.loads(msg.data)

        if not data:
            self.reset()
            return


        # Process the mask for initializing the pose
        method = data.pop('mask_format')
        image_data = data.pop('mask_data')
        if method == 'polygon':
            mask = np.zeros((self.camera.height, self.camera.width), dtype=np.uint8)
            polygon_pts = np.array(image_data).astype(int).reshape((-1, 1, 2))
            mask = cv2.polylines(mask, [polygon_pts], True, (255, 255, 255), 1) > 128

        elif method == 'file':
            mask = cv2.imread(image_data)
            if mask is None:
                raise FileNotFoundError(f"Could not read the file from {image_data}!")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 128

        else:
            raise ValueError(f"Unknown mask method {method}")

        model_prior = data.pop('model_prior', None)

        self.start_tracking(mask, model_prior=model_prior)

    def start_tracking(self, mask, model_prior=None):

        rgb = self.last_rgb
        depth = self.last_depth
        stamp = self.last_rgb_ts

        if rgb is None or depth is None:
            raise ValueError("Cannot start tracking, no images have been received!")

        if model_prior is not None:
            if model_prior not in self.models:
                self.models[model_prior] = self.process_model_file(model_prior)
            models = {model_prior: self.models[model_prior]}

        else:
            models = self.models

        best_score = None
        best_estimator = None
        best_pose = None
        best_model_match = None

        print("Evaluating models...")
        for model_file, mesh in models.items():
            estimator = FoundationPose(mesh.vertices, mesh.vertex_normals, mesh=mesh, scorer=self.scorer, refiner=self.refiner, glctx=self.glctx)
            pose = estimator.register(self.camera.K, rgb, depth, mask, glctx=self.glctx, iteration=2)
            scores = estimator.scores
            score = scores[0]

            print("\t{}: {:.5f}".format(os.path.split(model_file)[-1], score))

            if best_score is None or score > best_score:
                best_score = score
                best_estimator = estimator
                best_pose = pose
                best_model_match = model_file

        print("Best model match was for {}!".format(os.path.split(best_model_match)[-1]))
        self.estimator = best_estimator
        self.last_pose = best_pose
        self.active_model = best_model_match

        self.publish_tf_pose(best_pose, stamp=stamp)


    def update_tracking(self, *_):

        if self.estimator is None:
            return

        estimator = self.estimator
        rgb = self.last_rgb
        depth = self.last_depth
        stamp = self.last_rgb_ts

        new_pose = estimator.track_one(rgb=rgb, depth=depth, K=self.camera.K, iteration=2)
        self.last_pose = new_pose
        self.publish_tf_pose(new_pose, stamp=stamp)


if __name__ == '__main__':

    # See here for existing models and semantics
    # https://drive.google.com/drive/folders/17rlwbM6ekuNZfsmVwbC0m-qQxVsRfj3b?usp=sharing

    rospy.init_node("dish_pose_estimator")

    img_topic = '/camera/color/image_raw'
    depth_topic = '/camera/aligned_depth_to_color/image_raw'
    cam_topic = '/camera/color/camera_info'
    model_dir = '/home/alexyoufv/models_3d/for_tracker'

    estimator = DishPoseEstimator(img_topic, depth_topic, cam_topic, model_dir)
    rospy.spin()