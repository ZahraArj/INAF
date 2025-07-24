"""
Copyright (c) 2019 Andres Milioto, Jens Behley, Ignacio Vizzo, and Cyrill Stachniss
https://github.com/PRBonn/lidar-bonnetal
"""

import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt
import yaml
import os
import math
import tensorflow as tf
import time


class LaserScan:

    def __init__(self, H=64, W=720, fov_up=3.0, fov_down=-25.0,
                 min_depth=1, max_depth=80, mother_folder='./', idx='0'):
        self.mother_folder = mother_folder
        self.idx = idx                                                                              #Use for file saving
        self.proj_range = None
        self.proj_normal = None
        self.proj_mask = None
        self.proj_x = None
        self.proj_y = None
        self.proj_remission = None
        self.proj_idx = None
        self.proj_xyz = None
        self.unproj_range = None
        self.remissions = None
        self.points = None

        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.reset()

        with open("Mytools/config.yaml", 'r') as stream:
            cfg = yaml.safe_load(stream)

        Net_config = cfg['Visualization']
        self.pixelvalue = Net_config.get('pixelvalue')
        self.do_vis = Net_config.get('do_vis', 'False')

    def __len__(self):
        return self.size()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y
        self.remissions = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        self.proj_range = np.full((self.proj_H, self.proj_W), 0., dtype=np.float32)  # [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), 0, dtype=np.float32)
        self.proj_normal = np.full((self.proj_H, self.proj_W, 3), 0, dtype=np.float32)
        self.proj_remission = np.full((self.proj_H, self.proj_W), 0., dtype=np.float32)
        self.proj_idx = np.full((self.proj_H, self.proj_W), 0, dtype=np.int32)
        self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)  # if it contains a point or not

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def open_scan(self, pointsr):

        scan = pointsr

        # depth = np.linalg.norm(scan[:, 0:3], 2, axis=1)
        depth = tf.norm(scan[:, 0:3], ord=2,  axis=1)

        # max_idx = np.argwhere(depth > self.max_depth)
        max_idx = tf.where(depth > self.max_depth)

        # min_idx = np.argwhere(depth < self.min_depth)
        min_idx = tf.where(depth < self.min_depth)
        
        # idx = np.vstack((min_idx, max_idx))
        idx = tf.concat([min_idx, max_idx], 0)
        
        scan = np.delete(scan, idx, axis=0)
     
        self.points = scan[:, 0:3]  # get xyz
        self.remissions = scan[:, 3]  # get remission
        
        self.do_range_projection()
        # self.do_normal_projection1()
        self.do_normal_projection()
        # self.visualize_()

    def do_range_projection(self):

        # _________________________________________________________________________________________________Field of view
        fov_up = self.proj_fov_up / 180.0 * math.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * math.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # _____________________________________________________________________________________________xyz d unprojected
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]
        # depth = np.linalg.norm(self.points, 2, axis=1)
        depth = tf.norm(self.points, ord=2,  axis=1)
        # self.unproj_range = np.copy(depth)

        # _____________________________________________________________________________________________________Pitch/Yaw
        # yaw = -np.arctan2(scan_y, scan_x)
        yaw = -tf.math.atan2(scan_y, scan_x)
        # pitch = np.arcsin(scan_z / depth)
        pitch = tf.math.asin(scan_z / depth)


        # _________________________________________________get projections in image coords in [0.0, 1.0] > [W,H] > round
        # 1____________________________________________________[0.0, 1.0]
        proj_x = 0.5 * (yaw / math.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov
        # 2_________________________________________________________[W,H]
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]
        # 3_1______________________________________________________round_x
        # proj_x = np.round(proj_x)
        proj_x = tf.math.round(proj_x)
        # proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = tf.math.minimum(self.proj_W - 1, proj_x)
        # proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        proj_x = tf.math.maximum(0, proj_x)                # in [0,W-1]
        proj_x = tf.cast(proj_x, tf.int32)
        
        # self.proj_x = np.copy(proj_x)  # store a copy in orig order
        self.proj_x = tf.identity(proj_x)
        # 3_2______________________________________________________round_y
        # proj_y = np.round(proj_y)
        proj_y = tf.math.round(proj_y)
        # proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = tf.math.minimum(self.proj_H - 1, proj_y)
        # proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        proj_y = tf.math.maximum(0, proj_y)
        proj_y = tf.cast(proj_y, tf.int32)  # in [0,H-1]
        
        # self.proj_y = np.copy(proj_y)  # store a copy in original order
        self.proj_y = tf.identity(proj_y)

        # __________________________________________________________________________________________________________fill
        self.proj_range[self.proj_y, self.proj_x] = depth
        self.proj_xyz[self.proj_y, self.proj_x] = self.points
        self.proj_remission[self.proj_y, self.proj_x] = self.remissions
        # ______________________________________________________________________________________________________________

#     def do_normal_projection1(self):
#         points = self.proj_xyz.reshape(-1, 3)

#         # indices = np.where(np.all(points == [0., 0., 0.], axis=1))[0]
#         # points = np.delete(points, indices, axis=0)

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
#         # o3d.visualization.draw_geometries([pcd])

#         normals = np.asarray(pcd.normals)
#         # normals = tf.experimental.numpy.asarray(pcd.normals)

#         # normals3 = normals.reshape(self.proj_H, self.proj_W, 3)
#         # normals3 = np.reshape(normals, (self.proj_H, self.proj_W, 3))
#         normals3 = tf.reshape(normals, [self.proj_H, self.proj_W, 3])

#         # __________________________________________________________________________________________________________fill
#         self.proj_normal = normals3
#         # ______________________________________________________________________________________________________________

    def do_normal_projection(self):
        img = np.dstack((self.proj_xyz, self.proj_range))

        def calc_weights(x, alpha=-0.8):
            # return np.exp(alpha * np.abs(x))
            return tf.math.exp(alpha * tf.math.abs(x))
        

        diff_vertical = img[:-1, :, :] - img[1:, :, :]
        diff_horizontal = img[:, :-1, :] - img[:, 1:, :]
        # print(diff_vertical)

        x_diff_top = diff_vertical[:-1, 1:-1, :]
        x_diff_bottom = -diff_vertical[1:, 1:-1, :]

        x_diff_left = diff_horizontal[1:-1, :-1, :]
        x_diff_right = -diff_horizontal[1:-1, 1:, :]

        # x_range_diffs = np.stack(
        #     (x_diff_top[:, :, -1], x_diff_left[:, :, -1], x_diff_bottom[:, :, -1], x_diff_right[:, :, -1]), axis=2)

        x_range_diffs = tf.stack(
            (x_diff_top[:, :, -1], x_diff_left[:, :, -1], x_diff_bottom[:, :, -1], x_diff_right[:, :, -1]), axis=2)
        
        weights = calc_weights(x_range_diffs)

        # x_norm_tl = np.cross(weights[..., 0:1] * x_diff_top[..., :3], weights[..., 1:2] * x_diff_left[..., :3])
        # x_norm_lb = np.cross(weights[..., 1, None] * x_diff_left[..., :3],
        #                      weights[..., 2, None] * x_diff_bottom[..., :3])
        # x_norm_br = np.cross(weights[..., 2, None] * x_diff_bottom[..., :3],
        #                      weights[..., 3, None] * x_diff_right[..., :3])
        # x_norm_rt = np.cross(weights[..., 3, None] * x_diff_right[..., :3], weights[..., 0, None] * x_diff_top[..., :3])
                
        x_norm_tl = tf.linalg.cross(weights[..., 0:1] * x_diff_top[..., :3], weights[..., 1:2] * x_diff_left[..., :3])
        x_norm_lb = tf.linalg.cross(weights[..., 1, None] * x_diff_left[..., :3],
                             weights[..., 2, None] * x_diff_bottom[..., :3])
        x_norm_br = tf.linalg.cross(weights[..., 2, None] * x_diff_bottom[..., :3],
                             weights[..., 3, None] * x_diff_right[..., :3])
        x_norm_rt = tf.linalg.cross(weights[..., 3, None] * x_diff_right[..., :3], weights[..., 0, None] * x_diff_top[..., :3])
        

        # self.proj_normal = np.stack((x_norm_tl, x_norm_lb, x_norm_br, x_norm_rt))
        self.proj_normal = tf.stack((x_norm_tl, x_norm_lb, x_norm_br, x_norm_rt))
        # self.proj_normal = np.sum(self.proj_normal, axis=0)
        self.proj_normal = tf.reduce_sum(self.proj_normal, axis=0)

        # normalizing normals
        # print('a', np.shape(self.proj_normal))
        # self.proj_normal /= (np.linalg.norm(self.proj_normal, axis=2, keepdims=True) + 1e-8)
        # print('a', np.shape(self.proj_normal))
        # self.proj_normal = np.pad(self.proj_normal, ((1, 1), (1, 1), (0, 0)))
        # print('a', np.shape(self.proj_normal))
        
        self.proj_normal /= (tf.norm(self.proj_normal, axis=2, keepdims=True) + 1e-8)
        self.proj_normal = tf.pad(self.proj_normal, tf.constant([[1, 1,], [1, 1], [0, 0]]), mode='CONSTANT')


#     def visualize_(self):
#         if self.do_vis:
#             plt.close('all')

#             # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
#             if self.pixelvalue == 'reflectance':
#                 pixel_values = self.proj_remission
#             elif self.pixelvalue == 'range':
#                 pixel_values = self.proj_range
#             elif self.pixelvalue == 'xyz':
#                 pixel_values = self.proj_xyz[:, :, 1]
#             elif self.pixelvalue == 'normals':
#                 # print(self.proj_normal[30])
#                 pixel_values = self.proj_normal[:, :, 2]

#             dpi = 500  # Image resolution
#             fig, ax = plt.subplots(dpi=dpi)
#             fig.set_size_inches(16, 2)
#             indices = np.nonzero(pixel_values)
#             u = indices[0]
#             v = indices[1]

#             ax.scatter(v, u, s=1, c=tf.gather_nd(pixel_values,np.transpose(indices)), linewidths=0, alpha=1, cmap='gist_ncar')
#             ax.set_facecolor((0, 0, 0))  # Set regions with no points to black
#             ax.axis('scaled')
#             plt.xlim([0, self.proj_W])  # prevent drawing empty space outside of horizontal FOV
#             plt.ylim([0, self.proj_H])
#             if True:
#                 traj_path = os.path.join(self.mother_folder, 'scan2d')
#                 dir_i = os.path.join(traj_path, str(self.idx)) + '.png'
#                 plt.savefig(dir_i)
                
                

