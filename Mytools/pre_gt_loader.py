import numpy as np
from Mytools.kitti_open_items import Loader_KITTI
from Mytools.Myinv import rot_inv


class gt_loader:
    def __init__(self, pc_path, gt_path, seq):
        self.loader = Loader_KITTI(pc_path, gt_path, seq)
        self.initial_calib = self.loader.load_calib()

    def read_gt(self, i):
        # print('geo_gt', i - 1, i)
        if i == 0:
            gt_1 = np.array([1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0])
            gt_2 = self.loader.get_item_gt(i)
        else:
            gt_1 = self.loader.get_item_gt(i - 1)
            gt_2 = self.loader.get_item_gt(i)

        gt_T1 = np.array([[gt_1[0], gt_1[1], gt_1[2], gt_1[3]],
                          [gt_1[4], gt_1[5], gt_1[6], gt_1[7]],
                          [gt_1[8], gt_1[9], gt_1[10], gt_1[11]],
                          [0, 0, 0, 1]])
        gt_T2 = np.array([[gt_2[0], gt_2[1], gt_2[2], gt_2[3]],
                          [gt_2[4], gt_2[5], gt_2[6], gt_2[7]],
                          [gt_2[8], gt_2[9], gt_2[10], gt_2[11]],
                          [0, 0, 0, 1]])

        global_transform1 = gt_T1
        global_transform2 = gt_T2

        pose = [gt_2[3], gt_2[7], gt_2[11]]

        # print (global_transform1)
        # print (global_transform2)

        # global_transform1_inv = np.linalg.inv(global_transform1)
        # T = np.dot(global_transform1_inv, global_transform2)

        global_transform1_inv = rot_inv(global_transform1)
        T = np.dot(global_transform1_inv, global_transform2)

        # T = np.linalg.solve(global_transform1, global_transform2)

        # T = global_transform1

        return T, global_transform2, pose

    def read_gt_i(self, i):

        gt_1 = self.loader.get_item_gt(i)
        gt_2 = self.loader.get_item_gt(i+1)

        gt_T1 = np.array([[gt_1[0], gt_1[1], gt_1[2], gt_1[3]],
                          [gt_1[4], gt_1[5], gt_1[6], gt_1[7]],
                          [gt_1[8], gt_1[9], gt_1[10], gt_1[11]],
                          [0, 0, 0, 1]])
        gt_T2 = np.array([[gt_2[0], gt_2[1], gt_2[2], gt_2[3]],
                          [gt_2[4], gt_2[5], gt_2[6], gt_2[7]],
                          [gt_2[8], gt_2[9], gt_2[10], gt_2[11]],
                          [0, 0, 0, 1]])

        global_transform1 = gt_T1
        global_transform2 = gt_T2

        pose = [gt_2[3], gt_2[7], gt_2[11]]

        global_transform1_inv = rot_inv(global_transform1)
        T = np.dot(global_transform1_inv, global_transform2)

        return T, global_transform2, pose
