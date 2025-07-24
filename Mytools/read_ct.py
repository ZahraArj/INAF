import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import gc
import time
from datetime import datetime
from pytransform3d import transformations as pt



class ct_loader:

    def __init__(self):
        # Read YAML file
        self.icp_initial = None
        self.prev_scan_pts = None
        with open("Mytools/config.yaml", 'r') as stream:
            cfg = yaml.safe_load(stream)

        ds_config = cfg['datasets']['kitti']
        self.sequences = ds_config.get('sequences')
        self.scans = ds_config.get('scans')
        self.seq = ds_config.get('seq')

        self.process_number = ds_config.get('process_number')


        Net_config = cfg['Networks']
        self.time_size = Net_config.get('time_size')
        self.batch_gen = Net_config.get('batch_gen', False)
        self.batch_size = Net_config.get('Batch_size', 2)
        self.rot_rep = Net_config.get('rot_rep', 'expn')

        seq_int = int(self.seq)
        self.s_idx = self.time_size
        self.e_idx = int(self.scans[seq_int])
        self.internal_size = self.e_idx - self.s_idx


        self.path_ct = "/nas2/zahra/Kitti/ct-icp-outputs/cttxt"
        
    def rot_inv(self, T):
        invT = np.zeros((4, 4))
        inv_Rot = np.array([[T[0, 0], T[1, 0], T[2, 0]],
                            [T[0, 1], T[1, 1], T[2, 1]],
                            [T[0, 2], T[1, 2], T[2, 2]],
                            [0, 0, 0]])
        invT[0:4, 0:3] = inv_Rot
        transl = - np.dot(inv_Rot[0:3, :], T[0:3, 3])
        invT[0:3, 3] = transl
        invT[3, 3] = 1

        return invT

    def read_gt(self, gt_1, gt_2):
        # print('geo_gt', i - 1, i)


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

        global_transform1_inv = self.rot_inv(global_transform1)
        T = np.dot(global_transform1_inv, global_transform2)


        return T, global_transform2, pose
    
    
    def load_saved_data_all(self, seq):
        
        filename = os.path.join(self.path_ct, seq +".txt")
        data = np.loadtxt(filename)
        G_data = np.empty((len(data) - self.time_size +1, self.time_size, 8))
        # print("size", np.shape(data), np.shape(G_data))
        Ts_ct_dq = np.empty((len(data), 8))
        
        for i in range(len(data)):
            # print(i)
            if i == 0:
                gt_1 =np.array([1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0])
                gt_2 = data[i, 1:]
            else:
                gt_1 = data[i - 1, 1:]
                gt_2 = data[i, 1:]
            
            Trel_ct, _, _ =self.read_gt(gt_1, gt_2)
            Ts_ct_dq[i] = pt.dual_quaternion_from_transform(Trel_ct)
        
        for i in range(len(G_data)):
            G_data[i] = Ts_ct_dq[i:i+self.time_size]
                

        return G_data