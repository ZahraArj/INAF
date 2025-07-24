import copy

import numpy as np
from scipy.spatial.transform import Rotation as Rot_tr
import yaml
import os
from pytransform3d import transformations as pt
from Mytools.kitti_open_items import Loader_KITTI
from Mytools.pre_gt_loader import gt_loader
# from matplotlib import pyplot as plt
from datetime import datetime
import pytz

# output X,Y,Z,i,j,k,w
with open("Mytools/config.yaml", 'r') as stream:
    cfg = yaml.safe_load(stream)

ds_config = cfg['datasets']['kitti']
# s_idx = ds_config.get('s_idx')
# e_idx = ds_config.get('e_idx')
pc_path = ds_config.get('pc_path')
gt_path = ds_config.get('gt_path')
seq = ds_config.get('seq')
sequences = ds_config.get('sequences')
image_width = ds_config.get('image-width')
image_height = ds_config.get('image-height')
channels = ds_config.get('channels')

Net_config = cfg['Networks']
save_txt_path_result = Net_config.get('save_txt_path_result')
save_txt_path2 = Net_config.get('save_txt_path2')
save_txt_path_input = Net_config.get('save_txt_path_input')
# Saved_date = Net_config.get('saved_model')
Saved_date_model = Net_config.get('saved_model')
Saved_date_param = Net_config.get('saved_param')

method = Net_config.get('method')
Batch_size = Net_config.get('Batch_size')
Epochs = Net_config.get('Epochs')
loss_weights = Net_config.get('loss_weights')
time_size = Net_config.get('time_size')
fusion = Net_config.get('fusion')

scans = ds_config.get('scans')
seq_int = int(seq)
s_idx = 5
e_idx = int(scans[seq_int])
internal_size = e_idx - s_idx

loader_gt = gt_loader(pc_path, gt_path, seq)


def save2txt(output, typef, mother_folder, file_id=None, part=None, start_i=None, end_i=None, Tlast_counter=np.eye(4)):
    # T_gl = np.eye(4)
    if part == 0:
        print("start", start_i)
        Trel_gt, global_transform2, pose2 = loader_gt.read_gt(start_i)
        # Trel_gt, global_transform2, pose2 = loader_gt.read_gt(start_i-1)
        T_gl = global_transform2
    else:
        # Trel_gt, global_transform2, pose2 = loader_gt.read_gt(start_i)
        # T_gl = global_transform2
        T_gl = Tlast_counter

    if typef == 'result':
        # T_gl = Tlast
        rows = np.zeros((e_idx - s_idx, 13))
        for i in range(e_idx - s_idx):
            # _________________________________________________________________________________save for Kitti Evaluation
            R_quat = Rot_tr.from_quat(output[1][i])
            R_33 = R_quat.as_matrix()

            T_last = T_gl

            Trnsl = output[0][i]
            T_i34 = np.concatenate((R_33, Trnsl[:, None]), axis=1)
            T_i = np.concatenate((T_i34, [[0, 0, 0, 1]]), axis=0)

            T_gl = np.dot(T_last, T_i)

            rows[i, 0] = int(i + s_idx)
            rows[i, 1:13] = np.array([T_gl[0, 0], T_gl[0, 1], T_gl[0, 2], T_gl[0, 3],
                                      T_gl[1, 0], T_gl[1, 1], T_gl[1, 2], T_gl[1, 3],
                                      T_gl[2, 0], T_gl[2, 1], T_gl[2, 2], T_gl[2, 3]])

        filepath = os.path.join(mother_folder, 'results', 'result.txt')
        with open(filepath, "w+") as f:
            np.savetxt(f, rows, delimiter=' ', newline='\n', fmt="%.8g")
        f.close()

        print("_______________________________________________________________________________________________________")
        print('Result is saved')
        print("_______________________________________________________________________________________________________")

    elif typef == 'result_all':

        num_data = len(output)
        rows_all = np.zeros((end_i - start_i, 13))
        rows_all_rel = np.zeros((end_i - start_i, 13))
        for i in range(end_i - start_i):
            # _________________________________________________________________________________save for Kitti Evaluation
            T_last = T_gl
            # print(T_i)
            # print(np.round(output[i],5))
            T_i = pt.transform_from_exponential_coordinates(output[i], check=True)

            print(i)
            print(output[i][0:3])
            print(output[i][3:])
            print("____________________________________________________________")

            T_gl = np.dot(T_last, T_i)

            rows_all[i, 0] = int(i + start_i)
            rows_all[i, 1:13] = np.array([T_gl[0, 0], T_gl[0, 1], T_gl[0, 2], T_gl[0, 3],
                                          T_gl[1, 0], T_gl[1, 1], T_gl[1, 2], T_gl[1, 3],
                                          T_gl[2, 0], T_gl[2, 1], T_gl[2, 2], T_gl[2, 3]])

            rows_all_rel[i, 0] = int(i + start_i)
            rows_all_rel[i, 1:13] = np.array([T_i[0, 0], T_i[0, 1], T_i[0, 2], T_i[0, 3],
                                              T_i[1, 0], T_i[1, 1], T_i[1, 2], T_i[1, 3],
                                              T_i[2, 0], T_i[2, 1], T_i[2, 2], T_i[2, 3]])

        filepath = os.path.join(mother_folder, 'outputs', 'output_trajectory_parts', file_id + '.txt')
        filepath_rel = os.path.join(mother_folder, 'outputs', 'output_trajectory_parts', file_id + '_rel.txt')
        if part == 0:
            with open(filepath, "w+") as f:
                np.savetxt(f, rows_all, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
            with open(filepath_rel, "w+") as f_rel:
                np.savetxt(f_rel, rows_all_rel, delimiter=' ', newline='\n', fmt="%.8g")
            f_rel.close()
        else:
            with open(filepath, "a") as f:
                np.savetxt(f, rows_all, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
            with open(filepath_rel, "a") as f_rel:
                np.savetxt(f_rel, rows_all_rel, delimiter=' ', newline='\n', fmt="%.8g")
            f_rel.close()
        print("_______________________________________________________________________________________________________")
        print('Result and relative result is saved: seq', file_id, 'part:', part)
        print("_______________________________________________________________________________________________________")

    elif typef == 'result_all_dq':       
        num_data = len(output)
        # rows_all = np.zeros((end_i - start_i-4, 13))
        # rows_all_rel = np.zeros((end_i - start_i-4, 13))
        rows_all = np.zeros((num_data, 13))
        rows_all_rel = np.zeros((num_data, 13))
        
        print("num",num_data)
        for i in range(num_data):
        # for i in range(end_i - start_i-4):
        
            # _________________________________________________________________________________save for Kitti Evaluation
            T_last = copy.deepcopy(T_gl)
            T_i = pt.transform_from_dual_quaternion(output[i])

            T_gl = np.dot(T_last, T_i)

            rows_all[i, 0] = int(i + start_i)
            rows_all[i, 1:13] = np.array([T_gl[0, 0], T_gl[0, 1], T_gl[0, 2], T_gl[0, 3],
                                          T_gl[1, 0], T_gl[1, 1], T_gl[1, 2], T_gl[1, 3],
                                          T_gl[2, 0], T_gl[2, 1], T_gl[2, 2], T_gl[2, 3]])

            rows_all_rel[i, 0] = int(i + start_i)
            rows_all_rel[i, 1:13] = np.array([T_i[0, 0], T_i[0, 1], T_i[0, 2], T_i[0, 3],
                                              T_i[1, 0], T_i[1, 1], T_i[1, 2], T_i[1, 3],
                                              T_i[2, 0], T_i[2, 1], T_i[2, 2], T_i[2, 3]])
        
        
        filepath = os.path.join(mother_folder, 'outputs', 'output_trajectory_parts', Saved_date_model, file_id + '.txt')
        filepath_rel = os.path.join(mother_folder, 'outputs', 'output_trajectory_parts', Saved_date_model, file_id + '_rel.txt')
        if part == 0:
            with open(filepath, "w+") as f:
                np.savetxt(f, rows_all, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
            with open(filepath_rel, "w+") as f_rel:
                np.savetxt(f_rel, rows_all_rel, delimiter=' ', newline='\n', fmt="%.8g")
            f_rel.close()
        else:
            with open(filepath, "a") as f:
                np.savetxt(f, rows_all, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
            with open(filepath_rel, "a") as f_rel:
                np.savetxt(f_rel, rows_all_rel, delimiter=' ', newline='\n', fmt="%.8g")
            f_rel.close()
        print("_______________________________________________________________________________________________________")
        print('Result and relative result is saved: seq', file_id, 'part:', part)
        print("_______________________________________________________________________________________________________")

    elif typef == 'result_all_dq_onego':

        num_data = len(output)
        rows_all = np.zeros((end_i - start_i, 13))
        rows_all_rel = np.zeros((end_i - start_i, 13))
        for i in range(end_i - start_i):
            # _________________________________________________________________________________save for Kitti Evaluation
            T_last = T_gl

            T_i = pt.transform_from_dual_quaternion(output[i])

            T_gl = np.dot(T_last, T_i)

            rows_all[i, 0] = int(i + start_i)
            rows_all[i, 1:13] = np.array([T_gl[0, 0], T_gl[0, 1], T_gl[0, 2], T_gl[0, 3],
                                          T_gl[1, 0], T_gl[1, 1], T_gl[1, 2], T_gl[1, 3],
                                          T_gl[2, 0], T_gl[2, 1], T_gl[2, 2], T_gl[2, 3]])

            rows_all_rel[i, 0] = int(i + start_i)
            rows_all_rel[i, 1:13] = np.array([T_i[0, 0], T_i[0, 1], T_i[0, 2], T_i[0, 3],
                                              T_i[1, 0], T_i[1, 1], T_i[1, 2], T_i[1, 3],
                                              T_i[2, 0], T_i[2, 1], T_i[2, 2], T_i[2, 3]])

        filepath = os.path.join(mother_folder, 'outputs', 'output_trajectory_parts', file_id + '.txt')
        filepath_rel = os.path.join(mother_folder, 'outputs', 'output_trajectory_parts', file_id + '_rel.txt')
        if part == 0:
            with open(filepath, "w+") as f:
                np.savetxt(f, rows_all, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
            with open(filepath_rel, "w+") as f_rel:
                np.savetxt(f_rel, rows_all_rel, delimiter=' ', newline='\n', fmt="%.8g")
            f_rel.close()
        else:
            with open(filepath, "a") as f:
                np.savetxt(f, rows_all, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
            with open(filepath_rel, "a") as f_rel:
                np.savetxt(f_rel, rows_all_rel, delimiter=' ', newline='\n', fmt="%.8g")
            f_rel.close()
        print("_______________________________________________________________________________________________________")
        print('Result and relative result is saved: seq', file_id, 'part:', part)
        print("_______________________________________________________________________________________________________")

    elif typef == 'Ge_data_rel':
        T_gl_pose = []
        rows = np.zeros((e_idx, 13))
        for i in range(e_idx):
            # _________________________________________________________________________________save for Kitti Evaluation
            T_last = T_gl
            T_i = output[i]
            T_gl = np.dot(T_last, T_i)

            # rows[i, 0] = int(i + s_idx)
            rows[i, 0] = int(i)
            rows[i, 1:13] = np.array([T_gl[0, 0], T_gl[0, 1], T_gl[0, 2], T_gl[0, 3],
                                      T_gl[1, 0], T_gl[1, 1], T_gl[1, 2], T_gl[1, 3],
                                      T_gl[2, 0], T_gl[2, 1], T_gl[2, 2], T_gl[2, 3]])

            T_gl_pose.append(T_gl[0:3, 3])

        T_gl_pose = np.asarray(T_gl_pose)
        '''
        ax = plt.axes(projection='3d')
        ax.plot3D(T_gl_pose[:, 0], T_gl_pose[:, 1], T_gl_pose[:, 2], color='red')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('ICP_save')
        plt.show()

        plt.figure()
        plt.plot(T_gl_pose[:, 0], T_gl_pose[:, 2], color='red')
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.title('ICP_save_2D')
        plt.show()
        '''
        filepath = os.path.join(mother_folder, 'network_input_files', 'Geo_for_vis', 'Ge_data_seq_' + seq + '.txt')
        with open(filepath, "w+") as f:
            np.savetxt(f, rows, delimiter=' ', newline='\n', fmt="%.8g")
        f.close()

        filepath = os.path.join(mother_folder, 'network_input_files', 'Geo_for_vis', 'Ge_data_seq_' + seq)
        # plt.savefig(filepath)

    elif typef == 'Gt_data_rel_raw':
        T_gl_pose = []
        rows = np.zeros((e_idx - s_idx, 13))
        for i in range(e_idx - s_idx):
            # _________________________________________________________________________________save for Kitti Evaluation
            T_last = T_gl
            T_i = output[i]
            T_gl = np.dot(T_last, T_i)

            rows[i, 0] = int(i + s_idx)
            rows[i, 1:13] = np.array([T_gl[0, 0], T_gl[0, 1], T_gl[0, 2], T_gl[0, 3],
                                      T_gl[1, 0], T_gl[1, 1], T_gl[1, 2], T_gl[1, 3],
                                      T_gl[2, 0], T_gl[2, 1], T_gl[2, 2], T_gl[2, 3]])
            T_gl_pose.append(T_gl[0:3, 3])

        T_gl_pose = np.asarray(T_gl_pose)
        '''        
        ax = plt.axes(projection='3d')
        ax.plot3D(T_gl_pose[:, 0], T_gl_pose[:, 1], T_gl_pose[:, 2], color='green')

        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('GT_save')
        plt.show()

        plt.figure()
        plt.plot(T_gl_pose[:, 0], T_gl_pose[:, 2], color='green')
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.title('GT_save_2D')
        plt.show()
        '''

        filepath = os.path.join(mother_folder, 'network_input_files', 'Geo_for_vis', 'gt_for_vis_' + seq + '.txt')
        with open(filepath, "w+") as f:
            np.savetxt(f, rows, delimiter=' ', newline='\n', fmt="%.8g")
        f.close()

    elif typef == 'Ge_data':
        rows = np.zeros((e_idx - s_idx, 13))
        for i in range(e_idx - s_idx):
            # _________________________________________________________________________________save for Kitti Evaluation
            R_quat = Rot_tr.from_quat(output[i, 3:7])
            R_33 = R_quat.as_matrix()

            T_last = T_gl

            Trnsl = output[i, 0:3]
            T_i34 = np.concatenate((R_33, Trnsl[:, None]), axis=1)
            T_i = np.concatenate((T_i34, [[0, 0, 0, 1]]), axis=0)

            T_gl = np.dot(T_last, T_i)

            rows[i, 0] = int(i + s_idx)
            rows[i, 1:13] = np.array([T_gl[0, 0], T_gl[0, 1], T_gl[0, 2], T_gl[0, 3],
                                      T_gl[1, 0], T_gl[1, 1], T_gl[1, 2], T_gl[1, 3],
                                      T_gl[2, 0], T_gl[2, 1], T_gl[2, 2], T_gl[2, 3]])

        filepath = os.path.join(mother_folder, 'outputs', 'Ge_data.txt')
        with open(filepath, "w+") as f:
            np.savetxt(f, rows, delimiter=' ', newline='\n', fmt="%.8g")
        f.close()

    elif typef == 'Ge_data_all':
        data_gt_path = os.path.join(gt_path, str(file_id) + '.txt')
        file_gt = open(data_gt_path, 'r')
        Lines = file_gt.readlines()
        file_gt.close()
        num_data = len(Lines)

        rows_all = np.zeros((num_data - 1, 13))

        for i in range(num_data - 1):
            # _________________________________________________________________________________save for Kitti Evaluation
            if np.linalg.norm(output[i, 3:7]) == 0:
                print(output[i, 3:7])
            R_quat = Rot_tr.from_quat(output[i, 3:7])
            R_33 = R_quat.as_matrix()

            T_last = T_gl

            Trnsl = output[i, 0:3]
            T_i34 = np.concatenate((R_33, Trnsl[:, None]), axis=1)
            T_i = np.concatenate((T_i34, [[0, 0, 0, 1]]), axis=0)

            T_gl = np.dot(T_last, T_i)

            rows_all[i, 0] = int(i)
            rows_all[i, 1:13] = np.array([T_gl[0, 0], T_gl[0, 1], T_gl[0, 2], T_gl[0, 3],
                                          T_gl[1, 0], T_gl[1, 1], T_gl[1, 2], T_gl[1, 3],
                                          T_gl[2, 0], T_gl[2, 1], T_gl[2, 2], T_gl[2, 3]])

        filepath = os.path.join(mother_folder, 'outputs', file_id + '.pkl')
        with open(filepath, "w+") as f:
            np.savetxt(f, rows_all, delimiter=' ', newline='\n', fmt="%.8g")
        f.close()

    elif typef == 'intermed_geo':

        filepath = os.path.join(mother_folder, 'outputs', 'Intermediate_results', Saved_date_model,
                                'intermed_geo_seq_' + file_id + '.txt')

        output_flat = np.reshape(output, (output.shape[0], -1))
        print('int', np.shape(output), np.shape(output_flat))
        if part == 0:
            with open(filepath, "w+") as f:
                np.savetxt(f, output_flat, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
        else:
            with open(filepath, "a") as f:
                np.savetxt(f, output_flat, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
        print("_______________________________________________________________________________________________________")
        print('GEO __ Intermed results are saved: seq', file_id, 'part:', part)
        print("_______________________________________________________________________________________________________")

    elif typef == 'intermed_w_geo':

        filepath = os.path.join(mother_folder, 'outputs', 'output_trajectory_parts', 'Intermediate_results',
                                'intermed_we_geo_seq_' + file_id + '.txt')
        output_flat = np.reshape(output, (output.shape[0], -1))

        if part == 0:
            with open(filepath, "w+") as f:
                np.savetxt(f, output_flat, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
        else:
            with open(filepath, "a") as f:
                np.savetxt(f, output_flat, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
        print("_______________________________________________________________________________________________________")
        print('GEO __ Intermed weights are saved: seq', file_id, 'part:', part)
        print("_______________________________________________________________________________________________________")

    elif typef == 'intermed_lidar':
        filepath = os.path.join(mother_folder, 'outputs', 'Intermediate_results', Saved_date_model, 
                                'intermed_lidar_seq_' + file_id + '.txt')

        output_flat = np.reshape(output, (output.shape[0], -1))
        print('int', np.shape(output), np.shape(output_flat))
        if part == 0:
            with open(filepath, "w+") as f:
                np.savetxt(f, output_flat, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
        else:
            with open(filepath, "a") as f:
                np.savetxt(f, output_flat, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
        print("_______________________________________________________________________________________________________")
        print('LiDAR __ Intermed results are saved: seq', file_id, 'part:', part)
        print("_______________________________________________________________________________________________________")

    elif typef == 'intermed_w_lidar':

        filepath = os.path.join(mother_folder, 'outputs', 'Intermediate_results', Saved_date_model,
                                'intermed_we_lidar_seq_' + file_id + '.txt')
        output_flat = np.reshape(output, (output.shape[0], -1))
        print('w', np.shape(output), np.shape(output_flat))

        if part == 0:
            with open(filepath, "w+") as f:
                np.savetxt(f, output_flat, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
        else:
            with open(filepath, "a") as f:
                np.savetxt(f, output_flat, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
        print("_______________________________________________________________________________________________________")
        print('LiDAR __ Intermed weights are saved: seq', file_id, 'part:', part)
        print("_______________________________________________________________________________________________________")

    elif typef == 'intermed_w_combined':
        filepath = os.path.join(mother_folder, 'outputs', 'output_trajectory_parts', Saved_date_model,
                                'intermed_w_combined_seq_' + file_id + '.txt')

        output_flat = np.reshape(output, (output.shape[0], -1))
        if part == 0:
            with open(filepath, "w+") as f:
                np.savetxt(f, output_flat, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
        else:
            with open(filepath, "a") as f:
                np.savetxt(f, output_flat, delimiter=' ', newline='\n', fmt="%.8g")
            f.close()
        print("_______________________________________________________________________________________________________")
        print('Combined __ Intermed results are saved: seq', file_id, 'part:', part)
        print("_______________________________________________________________________________________________________")

    return T_gl


def create_folders(mother_folder):
    if not os.path.exists(mother_folder):

        Network_input_files = os.path.join(mother_folder, 'network_input_files')
        # result_train = os.path.join(mother_folder, 'results')
        saved_model = os.path.join(mother_folder, 'saved_model')
        bird_eye = os.path.join(mother_folder, 'bird_eye_view')
        traj = os.path.join(mother_folder, 'trajectory')
        scan2d = os.path.join(mother_folder, 'scan2d')

        os.makedirs(mother_folder)
        os.makedirs(Network_input_files)
        # os.makedirs(result_train)
        os.makedirs(saved_model)
        os.makedirs(bird_eye)
        os.makedirs(traj)
        os.makedirs(scan2d)
        print("_______________________________________________________________________________________________________")
        print("Created Directory:", mother_folder)
        print("_______________________________________________________________________________________________________")
    else:
        print("Directory already existed")
        exit()


def create_folders_init():
    if method == 'supervised':
        mother_folder = ''.join(['supervised_', str(s_idx), '_', str(e_idx)])
    else:
        mother_folder = ''.join(['self_Supervised', str(s_idx), '_', str(e_idx)])

    mother_folder = os.path.join('results', mother_folder)
    return mother_folder


def save_config(mother_folder):
    now = datetime.now(pytz.timezone('America/Toronto'))
    start_time = now.strftime("%Y_%m_%d_%H_%M")
    filepath = os.path.join(mother_folder, 'saved_model', start_time + 'config.txt')
    l0 = 'Dataset_______________________________________'
    l1 = 'start index: ' + str(s_idx)
    l2 = 'end index: ' + str(e_idx)
    l3 = 'KITTI dataset: sequence' + str(sequences)
    l4 = 'Image_width: ' + str(image_width)
    l5 = 'Image_height: ' + str(image_height)
    l6 = 'Channels: ' + str(channels)

    L0 = 'Network_______________________________________'
    L1 = 'Method: ' + str(method)
    L2 = 'Batch size: ' + str(Batch_size)
    L3 = 'Epochs: ' + str(Epochs)
    L4 = 'Loss Weights: ' + str(loss_weights)
    L5 = 'Fusion:' + str(fusion)

    listtosave = "\n".join([l0, l1, l2, l3, l4, l5, l6, L0, L1, L2, L3, L4, L5])

    f = open(filepath, "w+")
    f.writelines(listtosave)
    f.close()
