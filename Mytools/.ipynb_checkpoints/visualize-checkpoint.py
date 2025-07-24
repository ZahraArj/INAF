import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr

from Mytools.myprojection import birds_eye_point_cloud
from Mytools.kitti_open_items import Loader_KITTI
from Mytools.pre_geo_data import Geometry_data_prepare


class visall:

    def __init__(self, mother_folder):
        with open("Mytools/config.yaml", 'r') as stream:
            cfg = yaml.safe_load(stream)

        ds_config = cfg['datasets']['kitti']
        # self.s_idx = ds_config.get('s_idx')
        # self.e_idx = ds_config.get('e_idx')
        self.pc_path = ds_config.get('pc_path')
        self.gt_path = ds_config.get('gt_path')
        self.seq = ds_config.get('seq')
        self.sequences = ds_config.get('sequences', [0, 2, 3, 4, 5])
        self.scans = ds_config.get('scans', [4540, 1100, 4660, 800, 270, 2760])

        Net_config = cfg['Networks']
        self.save_txt_path_result = Net_config.get('save_txt_path_result')
        self.save_txt_path2 = Net_config.get('save_txt_path2')
        self.save_txt_path_input = Net_config.get('save_txt_path_input')
        self.divided_train = Net_config.get('divided_train', 200)
        self.Epochs = Net_config.get('Epochs', 2)
        self.Saved_date = Net_config.get('saved_model')

        # T = np.loadtxt('save_txt_path_result' + ".txt")

        self.loader_KT = Loader_KITTI(self.pc_path, self.gt_path, self.seq)
        self.mother_folder = mother_folder

    def vis_bird_eye(self):
        # ______________________________________________________________________________________________________________
        # Load results
        # ______________________________________________________________________________________________________________
        id0 = 0
        for idx in range(0, 1101):
            pc = self.loader_KT.get_item_pc(idx)
            
            plt_path = os.path.join(self.mother_folder, 'Fisheyed', self.seq)

            # __________________________________________________________________________________________________________
            birds_eye_point_cloud(pc[0], idx,
                                  side_range=(-50, 50),
                                  fwd_range=(-70, 70),
                                  res=0.1,
                                  min_height=-2.73,
                                  max_height=1.27,
                                  saveto=self.mother_folder)
            # __________________________________________________________________________________________________________

    def vis_traj(self):
        # __________________________
        # Ground Truth
        # __________________________
        filepath = os.path.join(self.mother_folder, 'result_train', '05.txt')
        gt_data = np.loadtxt(filepath, dtype=float)
        g0 = np.array([[gt_data[self.s_idx, 0], gt_data[self.s_idx, 1], gt_data[self.s_idx, 2], gt_data[self.s_idx, 3]],
                       [gt_data[self.s_idx, 4], gt_data[self.s_idx, 5], gt_data[self.s_idx, 6], gt_data[self.s_idx, 7]],
                       [gt_data[self.s_idx, 8], gt_data[self.s_idx, 9], gt_data[self.s_idx, 10],
                        gt_data[self.s_idx, 11]],
                       [0, 0, 0, 1]])

        # __________________________
        filepath = os.path.join(self.mother_folder, 'result_train', 'Ge_data.txt')
        G_data = np.loadtxt(filepath, dtype=float)

        # __________________________
        filepath = os.path.join(self.mother_folder, 'result_train', 'result.txt')
        result = np.loadtxt(filepath, dtype=float)

        fig, ax = plt.subplots(figsize=(12, 6))
        id0 = 0

        gt = ax.plot(gt_data[:, 3], gt_data[:, 11], color='green', label="map", linewidth=1)
        G_transformed_all = np.array([gt_data[self.s_idx, 3], gt_data[self.s_idx, 7], gt_data[self.s_idx, 11]])
        R_transformed_all = np.array([gt_data[self.s_idx, 3], gt_data[self.s_idx, 7], gt_data[self.s_idx, 11]])

        for idx in range(self.e_idx - self.s_idx - 1):
            plt.xlim([-300, 300])
            plt.ylim([-100, 400])
            # __________________________________
            # Geometry
            # __________________________________
            G_not_transformed = np.array([G_data[idx, 4], G_data[idx, 8], G_data[idx, 12], 1])
            G_transformed = np.dot(g0, G_not_transformed.T)
            G_transformed_all = np.vstack((G_transformed_all, G_transformed[0:3]))

            x = G_transformed_all[:, 0]
            y = G_transformed_all[:, 1]
            z = G_transformed_all[:, 2]
            # __________________________________
            # Result
            # __________________________________
            R_not_transformed = np.array([result[idx, 4], result[idx, 8], result[idx, 12], 1])
            R_transformed = np.dot(g0, R_not_transformed.T)
            R_transformed_all = np.vstack((R_transformed_all, R_transformed[0:3]))

            x2 = R_transformed_all[:, 0]
            y2 = R_transformed_all[:, 1]
            z2 = R_transformed_all[:, 2]

            # __________________________________
            # PLOT
            # __________________________________

            ax.plot(x2, z2, color='black', label="map", linewidth=1)
            # ax.plot(x, z, color='red', label="map", linewidth=1)

            # plt.legend(['Ground_truth', 'Our_method','Geometry_method'])
            plt.legend(['Ground_truth', 'Our_method'])

            traj_path = os.path.join(self.mother_folder, 'trajectory')
            dir_i = os.path.join(traj_path, str(idx)) + '.png'
            plt.savefig(dir_i)

    def vis_history(self):
        his_path = os.path.join(self.mother_folder, 'results', 'history1.npy')
        history = np.load(his_path, allow_pickle=True).item()
        for key, value in history.items():
            print(key)
        plt.plot(history['Translation_loss'])
        plt.plot(history['Quaternion_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Translation_loss', 'Quaternion_loss'], loc='upper left')

        his_path = os.path.join(self.mother_folder, 'results', 'TQ_loss.png')
        plt.savefig(his_path)
        plt.show()
        # ________________________________________________________
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.ylim([0.0,0.001 ])

        his_path = os.path.join(self.mother_folder, 'results', 'loss.png')
        plt.savefig(his_path)
        plt.show()

        # ________________________________________________________
        his_path = os.path.join(self.mother_folder, 'results', 'history1.npy')
        history = np.load(his_path, allow_pickle=True).item()
        plt.plot(history['Quaternion_loss'])
        plt.title('Quaternion_loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Quaternion_loss'], loc='upper left')
        # plt.ylim([0.0, 0.0008])
        plt.ylim([0.0, 0.001])

        his_path = os.path.join(self.mother_folder, 'results', 'Q_loss.png')
        plt.savefig(his_path)
        plt.show()

    def vis_history_all(self):
        train_loss_path = os.path.join(self.mother_folder, 'saved_model', self.Saved_date,'train_loss.npy')
        train_loss = np.load(train_loss_path, allow_pickle=True)
        n_epochs = list(len(train_loss[i]) for i in range(len(train_loss)))
        # train_loss = train_loss.flatten()
        train_loss = [i for ind in train_loss for i in ind]

        # print(len(train_loss))
        # plt.ylim([0.0, 10])
        # plt.plot(train_loss, label='train_loss')
        # plt.show()

        # trans_loss_path = os.path.join(self.mother_folder, 'results', 'trans_loss.npy')
        # trans_loss = np.load(trans_loss_path, allow_pickle=True)
        # trans_loss = trans_loss.flatten()
        #
        # quart_loss_path = os.path.join(self.mother_folder, 'results', 'quart_loss.npy')
        # quart_loss = np.load(quart_loss_path, allow_pickle=True)
        # quart_loss = quart_loss.flatten()

        val_loss_path = os.path.join(self.mother_folder, 'saved_model', self.Saved_date, 'val_loss.npy')
        val_loss = np.load(val_loss_path, allow_pickle=True)
        # val_loss = val_loss.flatten()
        val_loss = [i for ind in val_loss for i in ind]

        print(n_epochs)
        s = 0
        num_epochs = np.zeros([len(self.sequences), 1])
        # for idx in range(len(self.sequences) - 1):
        #     print(idx)
        #     s_last = s
        #     s = np.int(np.ceil(np.divide(self.scans[int(self.sequences[idx])], self.divided_train))) + s_last
        #     num_epochs[idx + 1] = np.sum(n_epochs[s_last:s]) + num_epochs[idx]
        if 1:
            # plt.ylim([0.0, 0.0001])
            # plt.ylim([0.0, 0.01])
            plt.ylim([0.0, 0.002])
            plt.plot(train_loss, label='train_loss')
            plt.plot(val_loss, label='validation_loss')
            plt.legend(loc="upper right")

            # for idx in range(len(self.sequences)):
            #     plt.annotate('seq' + self.sequences[idx], (num_epochs[idx], 0),
            #                  textcoords="offset points", xytext=(0, -40), ha='left', rotation=90)
            # plt.show()
            plt_path = os.path.join(self.mother_folder, 'saved_model', self.Saved_date, 'train_valid_loss.png')
            plt.savefig(plt_path)


class visbar:
    def __init__(self, mother_folder, seq):
        self.mother_folder = mother_folder
        self.seq = seq

    def bar_all(self, data_w, name='geo'):
        # run the animation
        fig, ax = plt.subplots()
        data_in = np.abs(data_w[:, -1])

        ymin = np.abs(np.amin(data_in, axis=0))
        ymax = np.abs(np.amax(data_in, axis=0))

        def animate(i):
            data_in1 = data_in[i]
            ax.clear()
            plt.bar(np.arange(64), data_in1, alpha=0.7, color='blue')
            plt.bar(np.arange(64), ymin, alpha=0.5, edgecolor='red', color='None', linewidth=1)
            plt.bar(np.arange(64), ymax, alpha=0.5, edgecolor='cyan', color='None', linewidth=1)
            plt.suptitle('scan:' + str(i))
            plt.ylim([0, 0.6])

        anim = FuncAnimation(fig, animate, repeat=False, save_count=50)
        if name == 'geo':
            filepath = os.path.join(self.mother_folder, 'results', 'geo_weights_' + self.seq + '.mp4')
        elif name == 'lidar':
            filepath = os.path.join(self.mother_folder, 'results', 'lidar_weights_' + self.seq + '.mp4')
        else:
            filepath = os.path.join(self.mother_folder, 'results', 'weights_' + self.seq + '.mp4')

        anim.save(filepath)
        # plt.show()


class vis_degree:
    def __init__(self, mother_folder, seq):
        self.mother_folder = mother_folder
        self.seq = seq

    def degree_all(self, gt_8, out_8):
        Euler_out = np.zeros([len(out_8), 3])
        Euler_gt = np.zeros([len(out_8), 3])
        # error_out = np.zeros(len(out_8))
        # print(out_8.shape, gt_8.shape)
        for i in range(len(out_8)):
            # print(i)
            T_out = pt.transform_from_dual_quaternion(out_8[i])
            T_gt = pt.transform_from_dual_quaternion(gt_8[i, -1])

            Euler_out[i] = np.array(pr.extrinsic_euler_xyx_from_active_matrix(T_out[0:3, 0:3]))
            Euler_gt[i] = np.array(pr.extrinsic_euler_xyx_from_active_matrix(T_gt[0:3, 0:3]))

        error_out = np.subtract(Euler_out, Euler_gt)
        print(Euler_out.shape, error_out.shape)
        plt.scatter(np.degrees(Euler_out[:, 0]), np.degrees(error_out[:, 0]))
        plt.show()
        plt.scatter(np.degrees(Euler_out[:, 1]), np.degrees(error_out[:, 1]))
        plt.show()
        plt.scatter(np.degrees(Euler_out[:, 2]), np.degrees(error_out[:, 2]))
        plt.show()
