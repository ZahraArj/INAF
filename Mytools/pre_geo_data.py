import numpy as np
import copy
from scipy.spatial.transform import Rotation as Rot_tr
import yaml
import pickle
import os
from pytransform3d import transformations as pt
from pytransform3d import rotations as rt
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import normalize

from Mytools.kitti_open_items import Loader_KITTI
from Mytools.pre_gt_loader import gt_loader
from Mytools.output_save import save2txt

import Mytools.UtilsPointcloud as Ptutils
import Mytools.ICP as ICP
from Mytools.Myinv import rot_inv


# scipy:            (x, y, z, w)
# pytransformed3d:  (w, x, y, z)


class Geometry_data_prepare:

    def __init__(self, mother_folder):
        # Read YAML file
        self.icp_initial = None
        self.prev_scan_pts = None
        with open("Mytools/config.yaml", 'r') as stream:
            cfg = yaml.safe_load(stream)

        ds_config = cfg['datasets']['kitti']
        self.sequences = ds_config.get('sequences')
        self.scans = ds_config.get('scans')
        # self.s_idx = ds_config.get('s_idx')
        # self.e_idx = ds_config.get('e_idx')
        self.pc_path = ds_config.get('pc_path')
        self.gt_path = ds_config.get('gt_path')
        self.seq = ds_config.get('seq')

        self.process_number = ds_config.get('process_number')
        # self.loader_KT = Loader_KITTI(self.pc_path, self.gt_path, self.seq)

        self.loader_gt = gt_loader(self.pc_path, self.gt_path, self.seq)

        self.mother_folder = mother_folder

        Net_config = cfg['Networks']
        self.time_size = Net_config.get('time_size')
        self.batch_gen = Net_config.get('batch_gen', False)
        self.batch_size = Net_config.get('Batch_size', 2)
        self.rot_rep = Net_config.get('rot_rep', 'expn')

        seq_int = int(self.seq)
        # self.s_idx = 5
        self.s_idx = self.time_size
        self.e_idx = int(self.scans[seq_int])
        self.internal_size = self.e_idx - self.s_idx
        # print(self.s_idx, self.e_idx)

        self.data_folder_path = os.path.join(self.pc_path, self.seq, 'velodyne')
        self.pcds_list = os.listdir(self.data_folder_path)
        self.pcds_list.sort()


    def load_saved_data(self):
        filename = os.path.join(self.mother_folder, 'network_input_files', 'geometry.pkl')
        geo_file = open(filename, "rb")
        data = pickle.load(geo_file)
        geo_file.close()
        G_data = data['Geo_noise']
        G_gt = data['gt_rel']
        # print('G_data.shape:', G_data.shape)
        # print('G_gt.shape:', G_gt.shape)

        return G_data, G_gt

    def load_saved_data_all(self, seq):
        if self.rot_rep == 'expn':
            filename = os.path.join(self.mother_folder, 'network_input_files', 'geometry_exp_' + seq + '.pkl')
        elif self.rot_rep == 'dquart':
            filename = os.path.join(self.mother_folder, 'network_input_files', 'geometry_dq_' + seq + '.pkl')
        else:
            filename = os.path.join(self.mother_folder, 'network_input_files', 'geometry_' + seq + '.pkl')
        geo_file = open(filename, "rb")
        data = pickle.load(geo_file)
        geo_file.close()
        G_data = data['Geo_noise']
        G_gt = data['gt_rel']
        return G_data, G_gt

    def create_geo_icp(self, seq=''):


        transform_icp_rel_rot = np.zeros([self.e_idx+1, 4, 4])
        transform_icp_gl = np.zeros([self.e_idx+1, 4, 4])
        Ts_dq = np.zeros([self.e_idx+1, 8])

        Ts_gt_dq = np.zeros([self.e_idx+1, 8])
        Trel_gt_all = np.zeros([self.e_idx+1, 4, 4])
        transform_gt_gl = np.zeros([self.e_idx+1, 4, 4])
        transform_icp_rel = np.zeros([self.e_idx+1, 4, 4])

        for idx in range(self.e_idx+1):
            # print("*******")
            print("idx", idx)
            # __________________________________________________________________________________________________________
            # Ground Truth
            # __________________________________________________________________________________________________________
            Trel_gt, global_transform2, pose2 = self.loader_gt.read_gt(idx)
            # _____________GT_local 8*1
            Ts_gt_dq[idx] = pt.dual_quaternion_from_transform(Trel_gt)
            # _____________GT_local 4*4 (directly save for output)
            Trel_gt_all[idx] = Trel_gt
            # _____________GT global 4*4
            transform_gt_gl[idx] = global_transform2

            # __________________________________________________________________________________________________________
            # ICP
            # __________________________________________________________________________________________________________
            scan_path = os.path.join(self.data_folder_path, self.pcds_list[idx])
            curr_scan_pts = Ptutils.readScan(scan_path)
            curr_scan_down_pts = Ptutils.random_sampling(curr_scan_pts, num_points=7000)

            if idx == 0:
                # self.prev_node_idx = curr_node_idx
                self.prev_scan_pts = copy.deepcopy(curr_scan_pts)
                self.icp_initial = np.eye(4)
                # transform_icp_rel[0] = np.eye(4)
                transform_icp_rel[idx] = np.array([[0.0, -1.0, 0.0, 0.0],
                                                   [0.0, 0.0, -1.0, 0.0],
                                                   [1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0]])
                continue

            prev_scan_down_pts = Ptutils.random_sampling(self.prev_scan_pts, num_points=7000)
            odom_transform, _, _ = ICP.icp(curr_scan_down_pts, prev_scan_down_pts, init_pose=self.icp_initial,
                                           max_iterations=50)

            self.prev_scan_pts = copy.deepcopy(curr_scan_pts)
            transform_icp_rel[idx] = odom_transform
            self.icp_initial = copy.deepcopy(odom_transform)

        # ________________________________________________________icp to global
        T_icp_gl = np.eye(4)

        for i in range(self.e_idx+1):
            T_last = copy.deepcopy(T_icp_gl)
            T_i = transform_icp_rel[i]
            T_icp_gl = np.dot(T_last, T_i)
            transform_icp_gl[i] = copy.deepcopy(T_icp_gl)

        # _____________________________________Pose alignment to first frame
        # idx_0 = sorted(list(poses_result.keys()))[0]
        pred_0 = transform_icp_gl[0]
        gt_0 = transform_gt_gl[0]

        # for cnt in range(len(transform_icp_gl)):
        #     transform_icp_gl[cnt] = np.linalg.inv(pred_0) @ transform_icp_gl[cnt]
        #     transform_gt_gl[cnt] = np.linalg.inv(gt_0) @ transform_gt_gl[cnt]

        # ________________________________________
        xyz_result = transform_icp_gl[:, 0:3, 3]
        xyz_result = xyz_result.T
        xyz_gt = transform_gt_gl[:, 0:3, 3]
        xyz_gt = xyz_gt.T

        # ________________________________________
        '''
        ax = plt.axes(projection='3d')
        ax.plot3D(xyz_gt[0, :], xyz_gt[1, :], xyz_gt[2, :], color='green')
        ax.plot3D(xyz_result[0, :], xyz_result[1, :], xyz_result[2, :], color='red')
        # ax.scatter(xyz_gt[0, 80:110], xyz_gt[1, 80:110], xyz_gt[2, 80:110], color='green', linewidths=0.25)
        # ax.scatter(xyz_result[0, 80:110], xyz_result[1, 80:110], xyz_result[2, 80:110], color='red',  linewidths=1)
        # for j in range (150):
        #     if np.remainder(j, 20) == 0:
        #         ax.text(xyz_result[0, j], xyz_result[1, j], xyz_result[2, j], int(j), fontsize=8)
        #         ax.text(xyz_gt[0, j], xyz_gt[1, j], xyz_gt[2, j], int(j), fontsize=8)

        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        plt.title("Before Optimization")
        ax.set_xlim3d([-300, 400])
        ax.set_ylim3d([-200, 400])
        ax.set_zlim3d([-100, 400])
        # ax.set_xlim3d([-10, 2])
        # ax.set_ylim3d([-20, 10])
        # ax.set_zlim3d([60, 200])
        plt.show()
        '''
        # ____________________________________________________________ optimize
        r, t, scale = umeyama_alignment(xyz_result, xyz_gt, with_scale=True)
        # print('r', r)
        # print('t', t)
        align_transformation = np.eye(4)
        # align_transformation[:3, :3] = r
        # align_transformation[:3, 3] = t

        for cnt in range(len(transform_icp_gl)):
            # print(cnt)
            # transform_icp_gl[cnt][:3, 3] *= scale
            transform_icp_gl[cnt] = align_transformation @ transform_icp_gl[cnt]
        #     __________________________________________
        xyz_result = transform_icp_gl[:, 0:3, 3]
        xyz_result = xyz_result.T
        '''
        ax = plt.axes(projection='3d')
        ax.plot3D(xyz_gt[0, :], xyz_gt[1, :], xyz_gt[2, :], color='green')
        ax.plot3D(xyz_result[0, :], xyz_result[1, :], xyz_result[2, :], color='red')
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        plt.title("After Optimization")
        ax.set_xlim3d([-300, 400])
        ax.set_ylim3d([-200, 400])
        ax.set_zlim3d([-100, 400])
        plt.show()
        '''
        # ______________________________________________________________________________________________________________
        # rows = np.zeros((self.e_idx - 1, 13))
        # for i in range(self.e_idx - 1):
        #     rows[i, 0] = int(i + self.time_size)
        #     T_gl = transform_icp_gl[i]
        #     rows[i, 1:13] = np.array([T_gl[0, 0], T_gl[0, 1], T_gl[0, 2], T_gl[0, 3],
        #                               T_gl[1, 0], T_gl[1, 1], T_gl[1, 2], T_gl[1, 3],
        #                               T_gl[2, 0], T_gl[2, 1], T_gl[2, 2], T_gl[2, 3]])
        #
        # filepath = os.path.join('results/supervised_all', 'network_input_files', 'Geo_for_vis', 'Ge_data_seq_' + seq + 'g.txt')
        # with open(filepath, "w+") as f:
        #     np.savetxt(f, rows, delimiter=' ', newline='\n', fmt="%.8g")
        # f.close()
        # ________________________________________to local
        # print(len(transform_icp_gl))
        for count, rot in enumerate(transform_icp_gl):
            # print(count)
            if count == 0:
                global_transform1_inv = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            else:
                global_transform1_inv = rot_inv(transform_icp_gl[count - 1])

            T = np.dot(global_transform1_inv, transform_icp_gl[count])
            transform_icp_rel_rot[count] = T
            RT8 = pt.dual_quaternion_from_transform(T)
            Ts_dq[count] = RT8

        s1_dq, s2_dq = Ts_dq.shape
        print(s1_dq, s2_dq)

        Ts_td_dq = np.empty([s1_dq - self.time_size + 1, self.time_size, s2_dq])
        
        # _________________________________________make window
        for i in range(s1_dq - self.time_size + 1):
            Ts_td_dq[i] = Ts_dq[i:i + self.time_size]


        data_td = {'Geo_noise': Ts_td_dq, 'gt_rel': Ts_gt_dq[self.time_size-1:]}
        # data_td = {'Geo_noise': Ts_td_dq, 'gt_rel': Ts_gt_dq[self.time_size:]}
        filename = os.path.join(self.mother_folder, 'network_input_files', 'geometry_dq' + seq + '.pkl')
        geo_file = open(filename, "wb")
        pickle.dump(data_td, geo_file, protocol=4)
        geo_file.close()

        save2txt(transform_icp_rel_rot, 'Ge_data_rel', self.mother_folder, file_id=seq, start_i=0)
        save2txt(Trel_gt_all, 'Gt_data_rel_raw', self.mother_folder, file_id=seq)


        print("_______________________________________________________________________________________________________")
        print('Geometry input saved. seq:', self.seq)
        print("_______________________________________________________________________________________________________")

        



def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def quaternion_multiply(Q0, Q1):
    # Extract the values from Q0 input: x y z w
    w0 = Q0[3]
    x0 = Q0[0]
    y0 = Q0[1]
    z0 = Q0[2]

    # Extract the values from Q1
    w1 = Q1[3]
    x1 = Q1[0]
    y1 = Q1[1]
    z1 = Q1[2]

    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion output: x y z w
    final_quaternion = np.array([Q0Q1_x, Q0Q1_y, Q0Q1_z, Q0Q1_w])

    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
    return final_quaternion


'''

    def create_geo_data(self):
        Ts = np.zeros([self.internal_size, 7])
        Ts_gt = np.zeros([self.internal_size, 7])
        raw_T_rel = np.zeros([self.internal_size, 4, 4])
        id0 = 0
        for idx in range(self.s_idx + 1, self.e_idx + 1):  # scipy: X, Y, Z, i, j, k, w
            # Ground Truth: R T
            # __________________________________________________________________________________________________________
            # Load gt + Noise
            # __________________________________________________________________________________________________________
            Trel_gt, global_transform1, pose = self.loader_gt.read_gt(idx)

            # _________________________________________________________Noise generation
            # noise to quaternion axis
            sig = abs(np.random.normal(0, 0.1))  # 0.1
            noise_r = np.random.normal(0, sig, (1, 3))
            noise_r_rot = Rot_tr.from_rotvec(noise_r * np.pi / 180)
            noise_r_q = noise_r_rot.as_quat()

            # noise to translation
            sig = abs(np.random.normal(0, 0.20))  # 0.2
            noise_t = np.random.normal(0, sig, (1, 3))

            # _____________________________________________________________Ground Truth
            R_mat_gt = Rot_tr.from_matrix(Trel_gt[0:3, 0:3])
            R_quat_gt = R_mat_gt.as_quat()
            RT6_gt = np.append(Trel_gt[0:3, 3], R_quat_gt)
            Ts_gt[id0] = RT6_gt

            raw_T_rel[id0] = Trel_gt

            # _________________________________________________________Convert to vector
            RT6 = np.append(RT6_gt[0:3] + noise_t, quaternion_multiply(R_quat_gt, noise_r_q[0]))
            Ts[id0] = RT6
            id0 += 1

        data = {'Geo_noise': Ts, 'gt_rel': Ts_gt, 'gt_rel44': raw_T_rel}
        filename = os.path.join(self.mother_folder, 'network_input_files', 'geometry.pkl')
        geo_file = open(filename, "wb")
        pickle.dump(data, geo_file, protocol=4)
        geo_file.close()

        save2txt(Ts, 'Ge_data', self.mother_folder)

        print("_______________________________________________________________________________________________________")
        print('Geometry input saved')
        print("_______________________________________________________________________________________________________")
        # 1: noise_rel (X, Y, Z, i, j, k, w)
        # 2: gt_rel (X, Y, Z, i, j, k, w)
        # 3: gt_rel 4X4
        # return self.Ts, self.Ts_gt, self.raw_T_rel

    def create_geo_timedist(self, seq=''):
        Ts = np.zeros([self.internal_size + self.time_size - 1, 7])
        Ts_gt = np.zeros([self.internal_size + self.time_size - 1, 7])
        raw_T_rel = np.zeros([self.internal_size + self.time_size - 1, 4, 4])
        id0 = 0
        for idx in range(self.s_idx + 1 - self.time_size + 1,
                         self.e_idx + 1):  # scipy: X, Y, Z, i, j, k, w
            # Ground Truth: R T
            # __________________________________________________________________________________________________________
            # Load gt + Noise
            # __________________________________________________________________________________________________________
            Trel_gt, global_transform2, pose2 = self.loader_gt.read_gt(idx)

            # _________________________________________________________Noise generation
            # noise to quaternion axis
            sig = abs(np.random.normal(0, 0.1))  # 0.1
            noise_r = np.random.normal(0, sig, (1, 3))
            noise_r_rot = Rot_tr.from_rotvec(noise_r * np.pi / 180)
            noise_r_q = noise_r_rot.as_quat()

            # noise to translation
            sig = abs(np.random.normal(0, 0.20))  # 0.2
            noise_t = np.random.normal(0, sig, (1, 3))

            # _____________________________________________________________Ground Truth
            R_mat_gt = Rot_tr.from_matrix(Trel_gt[0:3, 0:3])
            R_quat_gt = R_mat_gt.as_quat()
            RT6_gt = np.append(Trel_gt[0:3, 3], R_quat_gt)  # X Y Z q
            Ts_gt[id0] = RT6_gt

            # raw_T_rel[id0] = Trel_gt

            # _________________________________________________________Convert to vector
            RT6 = np.append(RT6_gt[0:3] + noise_t, quaternion_multiply(R_quat_gt, noise_r_q[0]))  # X Y Z q
            Ts[id0] = RT6
            id0 += 1

        s1, s2 = Ts.shape

        Ts_td = np.empty([self.e_idx - self.s_idx, self.time_size, s2])
        # Ts_gt_td = np.empty([self.e_idx - self.s_idx, self.time_size, s2])

        for i in range(self.e_idx - self.s_idx):
            Ts_td[i] = Ts[i:i + self.time_size]
            # Ts_gt_td[i] = Ts_gt[i:i + self.time_size]
        # data_td = {'Geo_noise': Ts_td, 'gt_rel': Ts_gt_td}

        data_td = {'Geo_noise': Ts_td, 'gt_rel': Ts_gt[self.time_size - 1:]}
        filename = os.path.join(self.mother_folder, 'network_input_files', 'geometry' + seq + '.pkl')
        geo_file = open(filename, "wb")

        pickle.dump(data_td, geo_file, protocol=4)
        geo_file.close()

        save2txt(Ts, 'Ge_data', self.mother_folder)

        # print(Ts_td.shape)
        print("_______________________________________________________________________________________________________")
        print('Geometry input saved')
        print("_______________________________________________________________________________________________________")
        # 1: noise_rel (X, Y, Z, i, j, k, w)
        # 2: gt_rel (X, Y, Z, i, j, k, w)
        # 3: gt_rel 4X4
        # return self.Ts, self.Ts_gt, self.raw_T_rel

    def create_geo_timedist_exp(self, seq=''):
        Ts = np.zeros([self.internal_size + self.time_size - 1, 7])
        Ts_gt = np.zeros([self.internal_size + self.time_size - 1, 7])
        transform_all = np.zeros([self.internal_size + self.time_size - 1, 4, 4])
        raw_T_rel = np.zeros([self.internal_size + self.time_size - 1, 4, 4])

        Ts_exp = np.zeros([self.internal_size + self.time_size - 1, 6])
        Ts_gt_exp = np.zeros([self.internal_size + self.time_size - 1, 6])

        id0 = 0
        for idx in range(self.s_idx + 1 - self.time_size + 1,
                         self.e_idx + 1):  # scipy: X, Y, Z, i, j, k, w
            # Ground Truth: R T
            # __________________________________________________________________________________________________________
            # Load gt + Noise
            # __________________________________________________________________________________________________________
            # print(idx)
            Trel_gt, global_transform2, pose2 = self.loader_gt.read_gt(idx)
            # _________________________________________________________Noise generation
            # noise to quaternion axis
            sig = abs(np.random.normal(0, 0.05))  # 0.1
            noise_r = np.random.normal(0, sig, (1, 3))
            noise_r_rot = Rot_tr.from_rotvec(noise_r * np.pi / 180)
            noise_r_q = noise_r_rot.as_quat()
            # print(noise_r * np.pi / 180)
            # noise_r_q = rt.quaternion_from_extrinsic_euler_xyz(noise_r * np.pi / 180)

            # noise to translation
            sig = abs(np.random.normal(0, 0.05))  # 0.2
            noise_t = np.random.normal(0, sig, (1, 3))

            # _____________________________________________________________Ground Truth
            R_quat_gt = rt.quaternion_from_matrix(Trel_gt[0:3, 0:3])  # w, x, y, z
            R_quat_gt = R_quat_gt[[1, 2, 3, 0]]
            # R_mat_gt = Rot_tr.from_matrix(Trel_gt[0:3, 0:3])
            # R_quat_gt = R_mat_gt.as_quat()
            RT6_gt = np.append(Trel_gt[0:3, 3], R_quat_gt)  # X Y Z q
            Ts_gt[id0] = RT6_gt

            Ts_gt_exp[id0] = pt.exponential_coordinates_from_transform(Trel_gt)
            raw_T_rel[id0] = Trel_gt

            # _________________________________________________________Convert to vector
            # sipy: X Y Z q   pt: x, y, z, qw, qx, qy, qz
            # print(quaternion_multiply(R_quat_gt, noise_r_q[0]))
            # print("noise:", noise_r_q)
            # print("quat:", R_quat_gt)
            RT6 = np.append(RT6_gt[0:3] + noise_t, quaternion_multiply(R_quat_gt, noise_r_q[0]))
            # Ts[id0] = RT6

            RT6_wX = RT6[[0, 1, 2, 6, 3, 4, 5]]
            transform_ = pt.transform_from_pq(RT6_wX)
            transform_all[id0] = transform_
            Ts_exp[id0] = pt.exponential_coordinates_from_transform(transform_)

            id0 += 1

        s1_exp, s2_exp = Ts_exp.shape

        Ts_td_exp = np.empty([self.e_idx - self.s_idx, self.time_size, s2_exp])

        for i in range(self.e_idx - self.s_idx):
            Ts_td_exp[i] = Ts_exp[i:i + self.time_size]

        data_td = {'Geo_noise': Ts_td_exp, 'gt_rel': Ts_gt_exp[self.time_size - 1:]}
        filename = os.path.join(self.mother_folder, 'network_input_files', 'geometry_exp' + seq + '.pkl')
        geo_file = open(filename, "wb")
        pickle.dump(data_td, geo_file, protocol=4)
        geo_file.close()

        save2txt(transform_all, 'Ge_data_screw', self.mother_folder, file_id=seq)
        save2txt(raw_T_rel, 'Ge_data_screw_raw', self.mother_folder, file_id=seq)

        print("_______________________________________________________________________________________________________")
        print('Geometry input saved')
        print("_______________________________________________________________________________________________________")

    def create_geo_timedist_dquart(self, seq=''):

        transform_all = np.zeros([self.internal_size + self.time_size - 1, 4, 4])
        raw_T_rel = np.zeros([self.internal_size + self.time_size - 1, 4, 4])

        Ts_dq = np.zeros([self.internal_size + self.time_size - 1, 8])
        Ts_gt_dq = np.zeros([self.internal_size + self.time_size - 1, 8])

        id0 = 0
        for idx in range(self.s_idx + 1 - self.time_size + 1,
                         self.e_idx + 1):  # scipy: X, Y, Z, i, j, k, w
            # __________________________________________________________________________________________________________
            # ICP
            # __________________________________________________________________________________________________________
            scan_path = os.path.join(self.data_folder_path, self.pcds_list[idx])
            curr_scan_pts = Ptutils.readScan(scan_path)
            curr_scan_down_pts = Ptutils.random_sampling(curr_scan_pts, num_points=7000)

            # save current node
            # curr_node_idx = idx
            if idx == self.s_idx + 1 - self.time_size + 1:
                # self.prev_node_idx = curr_node_idx
                self.prev_scan_pts = copy.deepcopy(curr_scan_pts)
                self.icp_initial = np.eye(4)
                continue

            # calc odometry
            prev_scan_down_pts = Ptutils.random_sampling(self.prev_scan_pts, num_points=7000)

            odom_transform, _, _ = ICP.icp(curr_scan_down_pts, prev_scan_down_pts, init_pose=self.icp_initial,
                                           max_iterations=20)

            self.prev_scan_pts = copy.deepcopy(curr_scan_pts)

            # Ground Truth: R T
            # __________________________________________________________________________________________________________
            # Load gt + Noise
            # p+eq: pw, px, py, pz, qw, qx, qy, qz
            # sipy: X Y Z q
            # quat: pq: (x, y, z, qw, qx, qy, qz)
            # __________________________________________________________________________________________________________
            # print(idx)
            Trel_gt, global_transform2, pose2 = self.loader_gt.read_gt(idx)
            # _________________________________________________________Noise generation
            # noise to quaternion axis
            np.random.seed(0)
            sig = np.random.normal(0, 0.005)  # 0.1
            noise_r = np.random.normal(0, sig, (1, 3))
            noise_r_rot = Rot_tr.from_rotvec(noise_r * np.pi / 180)
            noise_r_q = noise_r_rot.as_quat()

            # noise to translation
            sig = np.random.normal(0, 0.05)  # 0.2
            noise_t = np.random.normal(0, sig, (1, 3))

            # dq_noise = pt.concatenate_dual_quaternions(noise_r_dq, noise_t_dq)
            dq_noise = pt.dual_quaternion_from_pq([noise_t[0, 0], noise_t[0, 1], noise_t[0, 2],
                                                   noise_r_q[0, 3], noise_r_q[0, 0], noise_r_q[0, 1], noise_r_q[0, 2]])
            # _____________________________________________________________Ground Truth
            # _____________GT 8*1
            Ts_gt_dq[id0] = pt.dual_quaternion_from_transform(Trel_gt)
            # _____________GT 4*4
            raw_T_rel[id0] = Trel_gt
            # _________________________________________________________Convert to vector
            # RT8 = pt.concatenate_dual_quaternions(Ts_gt_dq[id0], dq_noise)
            # transform_ = pt.transform_from_dual_quaternion(RT8)
            transform_ = odom_transform
            transform_all[id0] = transform_

            RT8 = pt.dual_quaternion_from_transform(transform_)
            Ts_dq[id0] = RT8

            id0 += 1

        s1_exp, s2_exp = Ts_dq.shape

        Ts_td_dq = np.empty([self.e_idx - self.s_idx, self.time_size, s2_exp])

        for i in range(self.e_idx - self.s_idx):
            Ts_td_dq[i] = Ts_dq[i:i + self.time_size]

        data_td = {'Geo_noise': Ts_td_dq, 'gt_rel': Ts_gt_dq[self.time_size - 1:]}
        filename = os.path.join(self.mother_folder, 'network_input_files', 'geometry_dq' + seq + '.pkl')
        geo_file = open(filename, "wb")
        pickle.dump(data_td, geo_file, protocol=4)
        geo_file.close()

        save2txt(transform_all, 'Ge_data_screw', self.mother_folder, file_id=seq)
        save2txt(raw_T_rel, 'Ge_data_screw_raw', self.mother_folder, file_id=seq)

        print("_______________________________________________________________________________________________________")
        print('Geometry input saved. seq:', self.seq)
        print("_______________________________________________________________________________________________________")

    def create_geo_timedist_tfrecord(self):
        Ts = np.zeros([self.internal_size + self.time_size - 1, 7])
        Ts_gt = np.zeros([self.internal_size + self.time_size - 1, 7])
        raw_T_rel = np.zeros([self.internal_size + self.time_size - 1, 4, 4])
        id0 = 0
        for idx in range(self.s_idx + 1 - self.time_size + 1,
                         self.e_idx + 1):  # scipy: X, Y, Z, i, j, k, w
            # Ground Truth: R T
            # __________________________________________________________________________________________________________
            # Load gt + Noise
            # __________________________________________________________________________________________________________
            Trel_gt, global_transform2, pose2 = self.loader_gt.read_gt(idx)

            # _________________________________________________________Noise generation
            # noise to quaternion axis
            sig = abs(np.random.normal(0, 0.1))  # 0.1
            noise_r = np.random.normal(0, sig, (1, 3))
            noise_r_rot = Rot_tr.from_rotvec(noise_r * np.pi / 180)
            noise_r_q = noise_r_rot.as_quat()

            # noise to translation
            sig = abs(np.random.normal(0, 0.20))  # 0.2
            noise_t = np.random.normal(0, sig, (1, 3))

            # _____________________________________________________________Ground Truth
            R_mat_gt = Rot_tr.from_matrix(Trel_gt[0:3, 0:3])
            R_quat_gt = R_mat_gt.as_quat()
            RT6_gt = np.append(Trel_gt[0:3, 3], R_quat_gt)  # X Y Z q
            Ts_gt[id0] = RT6_gt

            raw_T_rel[id0] = Trel_gt

            # _________________________________________________________Convert to vector
            RT6 = np.append(RT6_gt[0:3] + noise_t, quaternion_multiply(R_quat_gt, noise_r_q[0]))  # X Y Z q
            Ts[id0] = RT6
            id0 += 1

        s1, s2 = Ts.shape

        Ts_td = np.empty([self.e_idx - self.s_idx, self.time_size, s2])
        Ts_gt_td = np.empty([self.e_idx - self.s_idx, self.time_size, s2])

        for i in range(self.e_idx - self.s_idx):
            Ts_td[i] = Ts[i:i + self.time_size]
            Ts_gt_td[i] = Ts_gt[i:i + self.time_size]

        # data_td = {'Geo_noise': Ts_td, 'gt_rel': Ts_gt[self.time_size - 1:]}
        return Ts_td, Ts_gt[self.time_size - 1:]

    def create_geo_timedist_bgen(self, batch_start):
        Ts = np.zeros([self.batch_size + self.time_size - 1, 7])
        Ts_gt = np.zeros([self.batch_size + self.time_size - 1, 7])
        raw_T_rel = np.zeros([self.batch_size + self.time_size - 1, 4, 4])
        id0 = 0
        for idx in range(batch_start + 1 - self.time_size + 1,
                         batch_start + 1 + self.batch_size):  # scipy: X, Y, Z, i, j, k, w
            # Ground Truth: R T
            # __________________________________________________________________________________________________________
            # Load gt + Noise
            # __________________________________________________________________________________________________________
            Trel_gt, global_transform2, pose2 = self.loader_gt.read_gt(idx)

            # _________________________________________________________Noise generation
            # noise to quaternion axis
            sig = abs(np.random.normal(0, 0.1))  # 0.1
            noise_r = np.random.normal(0, sig, (1, 3))
            noise_r_rot = Rot_tr.from_rotvec(noise_r * np.pi / 180)
            noise_r_q = noise_r_rot.as_quat()

            # noise to translation
            sig = abs(np.random.normal(0, 0.20))  # 0.2
            noise_t = np.random.normal(0, sig, (1, 3))

            # _____________________________________________________________Ground Truth
            R_mat_gt = Rot_tr.from_matrix(Trel_gt[0:3, 0:3])
            R_quat_gt = R_mat_gt.as_quat()
            RT6_gt = np.append(Trel_gt[0:3, 3], R_quat_gt)  # X Y Z q
            Ts_gt[id0] = RT6_gt

            raw_T_rel[id0] = Trel_gt

            # _________________________________________________________Convert to vector
            RT6 = np.append(RT6_gt[0:3] + noise_t, quaternion_multiply(R_quat_gt, noise_r_q[0]))  # X Y Z q
            Ts[id0] = RT6
            id0 += 1

        s1, s2 = Ts.shape

        Ts_td = np.empty([self.batch_size, self.time_size, s2])
        Ts_gt_td = np.empty([self.batch_size, self.time_size, s2])

        for i in range(self.batch_size):
            Ts_td[i] = Ts[i:i + self.time_size]
            Ts_gt_td[i] = Ts_gt[i:i + self.time_size]
        # data_td = {'Geo_noise': Ts_td, 'gt_rel': Ts_gt_td}

        data_td = {'Geo_noise': Ts_td, 'gt_rel': Ts_gt[self.time_size - 1:]}
        # filename = os.path.join(self.mother_folder, 'network_input_files', 'geometry.pkl')
        # geo_file = open(filename, "wb")
        # pickle.dump(data_td, geo_file, protocol=4)
        # geo_file.close()

        # save2txt(Ts, 'Ge_data', self.mother_folder)

        return data_td['Geo_noise'], data_td['gt_rel']

    def create_geo_timedist_tfdata(self, batch_start):
        Ts = np.zeros([self.time_size, 7])
        Ts_gt = np.zeros([self.time_size, 7])
        raw_T_rel = np.zeros([self.time_size, 4, 4])
        id0 = 0
        for idx in range(batch_start + 1 - self.time_size + 1,
                         batch_start + 2):  # scipy: X, Y, Z, i, j, k, w
            # print(idx)
            # Ground Truth: R T
            # __________________________________________________________________________________________________________
            # Load gt + Noise
            # __________________________________________________________________________________________________________
            Trel_gt, global_transform2, pose2 = self.loader_gt.read_gt(idx)

            # _________________________________________________________Noise generation
            # noise to quaternion axis
            sig = abs(np.random.normal(0, 0.1))  # 0.1
            noise_r = np.random.normal(0, sig, (1, 3))
            noise_r_rot = Rot_tr.from_rotvec(noise_r * np.pi / 180)
            noise_r_q = noise_r_rot.as_quat()

            # noise to translation
            sig = abs(np.random.normal(0, 0.20))  # 0.2
            noise_t = np.random.normal(0, sig, (1, 3))

            # _____________________________________________________________Ground Truth
            R_mat_gt = Rot_tr.from_matrix(Trel_gt[0:3, 0:3])
            R_quat_gt = R_mat_gt.as_quat()
            RT6_gt = np.append(Trel_gt[0:3, 3], R_quat_gt)  # X Y Z q
            Ts_gt[id0] = RT6_gt

            raw_T_rel[id0] = Trel_gt

            # _________________________________________________________Convert to vector
            RT6 = np.append(RT6_gt[0:3] + noise_t, quaternion_multiply(R_quat_gt, noise_r_q[0]))  # X Y Z q
            Ts[id0] = RT6
            id0 += 1

        Ts_td = Ts
        data_td = {'Geo_noise': Ts_td, 'gt_rel': Ts_gt[-1]}

        return data_td['Geo_noise'], data_td['gt_rel']

    def create_all_geo_timedist(self, file_id):
        loader_gt_all = gt_loader(self.pc_path, self.gt_path, str(file_id))
        data_gt_path = os.path.join(self.gt_path, str(file_id) + '.txt')

        file_gt = open(data_gt_path, 'r')
        Lines = file_gt.readlines()
        file_gt.close()
        num_data = len(Lines)

        Ts = np.zeros([num_data - 1, 7])
        Ts_gt = np.zeros([num_data - 1, 7])
        raw_T_rel = np.zeros([num_data - 1, 4, 4])
        id0 = 0

        for idx in range(1, num_data):  # scipy: X, Y, Z, i, j, k, w
            # Ground Truth: R T
            # __________________________________________________________________________________________________________
            # Load gt + Noise
            # __________________________________________________________________________________________________________
            Trel_gt, global_transform2, pose2 = loader_gt_all.read_gt(idx)

            # _________________________________________________________Noise generation
            # noise to quaternion axis
            sig = abs(np.random.normal(0, 0.1))  # 0.1
            noise_r = np.random.normal(0, sig, (1, 3))
            noise_r_rot = Rot_tr.from_rotvec(noise_r * np.pi / 180)
            noise_r_q = noise_r_rot.as_quat()

            # noise to translation
            sig = abs(np.random.normal(0, 0.20))  # 0.2
            noise_t = np.random.normal(0, sig, (1, 3))

            # _____________________________________________________________Ground Truth
            R_mat_gt = Rot_tr.from_matrix(Trel_gt[0:3, 0:3])
            R_quat_gt = R_mat_gt.as_quat()
            RT6_gt = np.append(Trel_gt[0:3, 3], R_quat_gt)  # X Y Z q
            Ts_gt[id0] = RT6_gt

            raw_T_rel[id0] = Trel_gt

            # _______________________________________________________________GT + Noise
            RT6 = np.append(RT6_gt[0:3] + noise_t, quaternion_multiply(R_quat_gt, noise_r_q[0]))  # X Y Z q
            Ts[id0] = RT6
            id0 += 1

        s1, s2 = Ts.shape

        Ts_td = np.empty([num_data - self.time_size, self.time_size, s2])
        Ts_gt_td = np.empty([num_data - self.time_size, self.time_size, s2])

        for i in range(num_data - self.time_size):
            Ts_td[i] = Ts[i:i + self.time_size]
            Ts_gt_td[i] = Ts_gt[i:i + self.time_size]
        # data_td = {'Geo_noise': Ts_td, 'gt_rel': Ts_gt_td}
        # ______________________________________________________________________________________________________________
        # Save input files for the network
        # ______________________________________________________________________________________________________________
        data_td = {'Geo_noise': Ts_td, 'gt_rel': Ts_gt[self.time_size - 1:]}
        filename = os.path.join(self.mother_folder, 'network_input_files', file_id + '.pkl')
        geo_file = open(filename, "wb")
        pickle.dump(data_td, geo_file, protocol=4)
        geo_file.close()

        G_data = data_td['Geo_noise']
        G_gt = data_td['gt_rel']
        # ______________________________________________________________________________________________________________
        # Save geo+noise for later evaluation (in result folder)
        # ______________________________________________________________________________________________________________

        save2txt(Ts, 'Ge_data_all', self.mother_folder, file_id)

        # print(Ts_td.shape)
        print("_______________________________________________________________________________________________________")
        print('Geometry input saved  file: ', file_id)
        print("_______________________________________________________________________________________________________")
        # 1: noise_rel (X, Y, Z, i, j, k, w)
        # 2: gt_rel (X, Y, Z, i, j, k, w)
        # 3: gt_rel 4X4
        # return self.Ts, self.Ts_gt, self.raw_T_rel
        return G_data

    def create_geo_icp2(self, seq=''):

        transform_icp_rel_rot = np.zeros([self.e_idx, 4, 4])
        transform_icp_gl = np.zeros([self.e_idx, 4, 4])
        Ts_dq = np.zeros([self.e_idx, 8])

        Ts_gt_dq = np.zeros([self.e_idx, 8])
        Trel_gt_all = np.zeros([self.e_idx, 4, 4])
        transform_gt_gl = np.zeros([self.e_idx, 4, 4])
        transform_icp_rel = np.zeros([self.e_idx, 4, 4])

        for idx in range(self.e_idx):
            # print("*******")
            # print("idx", idx)
            # __________________________________________________________________________________________________________
            # Ground Truth
            # __________________________________________________________________________________________________________
            Trel_gt, global_transform2, pose2 = self.loader_gt.read_gt(idx)
            # _____________GT_local 8*1
            Ts_gt_dq[idx] = pt.dual_quaternion_from_transform(Trel_gt)
            # _____________GT_local 4*4 (directly save for output)
            Trel_gt_all[idx] = Trel_gt
            # _____________GT global 4*4
            transform_gt_gl[idx] = global_transform2

            # __________________________________________________________________________________________________________
            # ICP
            # __________________________________________________________________________________________________________
            scan_path = os.path.join(self.data_folder_path, self.pcds_list[idx])
            curr_scan_pts = Ptutils.readScan(scan_path)
            curr_scan_down_pts = Ptutils.random_sampling(curr_scan_pts, num_points=7000)

            if idx == 0:
                # self.prev_node_idx = curr_node_idx
                self.prev_scan_pts = copy.deepcopy(curr_scan_pts)
                self.icp_initial = np.eye(4)
                # transform_icp_rel[0] = np.eye(4)
                transform_icp_rel[idx] = np.array([[0.0, -1.0, 0.0, 0.0],
                                                   [0.0, 0.0, -1.0, 0.0],
                                                   [1.0, 0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0, 1.0]])
                # print("ICP0: I")
                continue

            # calc odometry
            # print("ICP", "curr:", scan_path, "prev", "before")
            # print("____________")
            prev_scan_down_pts = Ptutils.random_sampling(self.prev_scan_pts, num_points=7000)
            odom_transform, _, _ = ICP.icp(curr_scan_down_pts, prev_scan_down_pts, init_pose=self.icp_initial,
                                           max_iterations=50)

            self.prev_scan_pts = copy.deepcopy(curr_scan_pts)
            transform_icp_rel[idx] = odom_transform
            self.icp_initial = copy.deepcopy(odom_transform)

        # ________________________________________________________icp to global
        T_icp_gl = np.eye(4)

        for i in range(self.e_idx):
            T_last = copy.deepcopy(T_icp_gl)
            T_i = transform_icp_rel[i]
            T_icp_gl = np.dot(T_last, T_i)
            transform_icp_gl[i] = copy.deepcopy(T_icp_gl)

        # _____________________________________Pose alignment to first frame
        # idx_0 = sorted(list(poses_result.keys()))[0]
        pred_0 = transform_icp_gl[0]
        gt_0 = transform_gt_gl[0]

        # for cnt in range(len(transform_icp_gl)):
        #     transform_icp_gl[cnt] = np.linalg.inv(pred_0) @ transform_icp_gl[cnt]
        #     transform_gt_gl[cnt] = np.linalg.inv(gt_0) @ transform_gt_gl[cnt]

        # ________________________________________
        xyz_result = transform_icp_gl[:, 0:3, 3]
        xyz_result = xyz_result.T
        xyz_gt = transform_gt_gl[:, 0:3, 3]
        xyz_gt = xyz_gt.T

        # ________________________________________

        ax = plt.axes(projection='3d')
        ax.plot3D(xyz_gt[0, :], xyz_gt[1, :], xyz_gt[2, :], color='green')
        ax.plot3D(xyz_result[0, :], xyz_result[1, :], xyz_result[2, :], color='red')
        # ax.scatter(xyz_gt[0, 80:110], xyz_gt[1, 80:110], xyz_gt[2, 80:110], color='green', linewidths=0.25)
        # ax.scatter(xyz_result[0, 80:110], xyz_result[1, 80:110], xyz_result[2, 80:110], color='red',  linewidths=1)
        # for j in range (150):
        #     if np.remainder(j, 20) == 0:
        #         ax.text(xyz_result[0, j], xyz_result[1, j], xyz_result[2, j], int(j), fontsize=8)
        #         ax.text(xyz_gt[0, j], xyz_gt[1, j], xyz_gt[2, j], int(j), fontsize=8)

        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        plt.title("Before Optimization")
        ax.set_xlim3d([-300, 400])
        ax.set_ylim3d([-200, 400])
        ax.set_zlim3d([-100, 400])
        # ax.set_xlim3d([-10, 2])
        # ax.set_ylim3d([-20, 10])
        # ax.set_zlim3d([60, 200])
        plt.show()
        # ____________________________________________________________ optimize
        r, t, scale = umeyama_alignment(xyz_result, xyz_gt, with_scale=True)
        # print('r', r)
        # print('t', t)
        align_transformation = np.eye(4)
        # align_transformation[:3, :3] = r
        # align_transformation[:3, 3] = t

        for cnt in range(len(transform_icp_gl)):
            # print(cnt)
            # transform_icp_gl[cnt][:3, 3] *= scale
            transform_icp_gl[cnt] = align_transformation @ transform_icp_gl[cnt]
        #     __________________________________________
        xyz_result = transform_icp_gl[:, 0:3, 3]
        xyz_result = xyz_result.T

        ax = plt.axes(projection='3d')
        ax.plot3D(xyz_gt[0, :], xyz_gt[1, :], xyz_gt[2, :], color='green')
        ax.plot3D(xyz_result[0, :], xyz_result[1, :], xyz_result[2, :], color='red')
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        plt.title("After Optimization")
        ax.set_xlim3d([-300, 400])
        ax.set_ylim3d([-200, 400])
        ax.set_zlim3d([-100, 400])
        plt.show()
        # ______________________________________________________________________________________________________________
        # rows = np.zeros((self.e_idx - 1, 13))
        # for i in range(self.e_idx - 1):
        #     rows[i, 0] = int(i + self.time_size)
        #     T_gl = transform_icp_gl[i]
        #     rows[i, 1:13] = np.array([T_gl[0, 0], T_gl[0, 1], T_gl[0, 2], T_gl[0, 3],
        #                               T_gl[1, 0], T_gl[1, 1], T_gl[1, 2], T_gl[1, 3],
        #                               T_gl[2, 0], T_gl[2, 1], T_gl[2, 2], T_gl[2, 3]])
        #
        # filepath = os.path.join('results/supervised_all', 'network_input_files', 'Geo_for_vis', 'Ge_data_seq_' + seq + 'g.txt')
        # with open(filepath, "w+") as f:
        #     np.savetxt(f, rows, delimiter=' ', newline='\n', fmt="%.8g")
        # f.close()
        # ________________________________________to local
        # print(len(transform_icp_gl))
        for count, rot in enumerate(transform_icp_gl):
            # print(count)
            if count == 0:
                global_transform1_inv = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            else:
                global_transform1_inv = rot_inv(transform_icp_gl[count - 1])

            T = np.dot(global_transform1_inv, transform_icp_gl[count])
            transform_icp_rel_rot[count] = T
            # if count in (95,96,97,98,99,100,101,102,103,104,105):
            #     print(count, T)
            #     print("!!!!!!!!!")
            RT8 = pt.dual_quaternion_from_transform(T)
            Ts_dq[count] = RT8

        s1_dq, s2_dq = Ts_dq.shape
        print(s1_dq, s2_dq)

        Ts_td_dq = np.empty([s1_dq - self.time_size + 1, self.time_size, s2_dq])

        for i in range(s1_dq - self.time_size + 1):
            Ts_td_dq[i] = Ts_dq[i:i + self.time_size]

        # print("______T8")
        # print(Ts_dq[95:105])
        # print("______gt8")
        # print(Ts_gt_dq[95:105])
        data_td = {'Geo_noise': Ts_td_dq, 'gt_rel': Ts_gt_dq[self.time_size-1:]}
        # data_td = {'Geo_noise': Ts_td_dq, 'gt_rel': Ts_gt_dq[self.time_size:]}
        filename = os.path.join(self.mother_folder, 'network_input_files', 'geometry_dq' + seq + '.pkl')
        geo_file = open(filename, "wb")
        pickle.dump(data_td, geo_file, protocol=4)
        geo_file.close()

        save2txt(transform_icp_rel_rot, 'Ge_data_rel', self.mother_folder, file_id=seq, start_i=0)
        save2txt(Trel_gt_all, 'Gt_data_rel_raw', self.mother_folder, file_id=seq)

        # print("icp", transform_icp_rel_rot)
        # print("gt", Trel_gt_all)
        print("_______________________________________________________________________________________________________")
        print('Geometry input saved. seq:', self.seq)
        print("_______________________________________________________________________________________________________")
    def create_loam(self, seq=''):

        transform_icp_rel = np.zeros([self.e_idx - 1, 4, 4])
        transform_icp_rel_rot = np.zeros([self.e_idx - 1, 4, 4])
        transform_icp_gl = np.zeros([self.e_idx - 1, 4, 4])
        transform_gt_gl = np.zeros([self.e_idx - 1, 4, 4])
        raw_T_rel = np.zeros([self.e_idx - 1, 4, 4])
        Trel_gt_all = np.zeros([self.e_idx - 1, 4, 4])

        Ts_dq = np.zeros([self.internal_size + self.time_size - 1, 8])
        Ts_gt_dq = np.zeros([self.internal_size + self.time_size - 1, 8])

        id0 = 0
        for idx in range(0, self.e_idx):
            # __________________________________________________________________________________________________________
            # ICP
            # __________________________________________________________________________________________________________
            scan_path = os.path.join(self.data_folder_path, self.pcds_list[idx])
            curr_scan_pts = Ptutils.readScan(scan_path)
            curr_scan_down_pts = Ptutils.random_sampling(curr_scan_pts, num_points=7000)

            # save current node
            # curr_node_idx = idx
            if idx == 0:
                # self.prev_node_idx = curr_node_idx
                self.prev_scan_pts = copy.deepcopy(curr_scan_pts)
                self.icp_initial = np.eye(4)
                continue

            # calc odometry
            prev_scan_down_pts = Ptutils.random_sampling(self.prev_scan_pts, num_points=7000)
            odom_transform, _, _ = ICP.icp(curr_scan_down_pts, prev_scan_down_pts, init_pose=self.icp_initial,
                                           max_iterations=20)
            self.prev_scan_pts = copy.deepcopy(curr_scan_pts)
            transform_icp_rel[id0] = odom_transform

            # __________________________________________________________________________________________________________
            # Ground Truth
            # __________________________________________________________________________________________________________
            Trel_gt, global_transform2, pose2 = self.loader_gt.read_gt(idx)
            # _____________GT 8*1
            Ts_gt_dq[id0] = pt.dual_quaternion_from_transform(Trel_gt)
            # _____________GT 4*4
            Trel_gt_all[id0] = Trel_gt

            # _____________GT globla 4*4
            transform_gt_gl[id0] = global_transform2

            id0 += 1
'''