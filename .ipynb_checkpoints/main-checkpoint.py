# Zahra Arjmandi
# 28-01-2022

# _________________________________________________________________________________________________________ Dependencies
# Requirements:
# tensorflow 2.3
# numpy 1.19.5
# open3d  0.12
# numpy, torch, open3d, opencv
# pip install torch, pip install open3d, pip install opencv-python, pip install keras, pip install PyYAML,
# pip install matplotlib, conda install opencv
# pip install pydot, sudo apt install graphviz

# ___________________________________________________________________________________________________ External libraries
import numpy as np
import yaml
import time as time
import os
# import tensorflow as tf


# ___________________________________________________________________________________________________________ My classes
from Mytools.My_model_g import BaseNet
# from Mytools.My_Models import BaseNet
from Mytools.make_pcfile_4network import Lidar_data_prepare
from Mytools.pre_geo_data import Geometry_data_prepare
from Mytools.output_save import save2txt, create_folders_init, create_folders, save_config
from Mytools.visualize import visall
from Mytools.tfrecord_tfread import recorder_reader
# from Mytools.vis_bars import vis_weights

# ______________________________________________________________________________________________________________________
# # Set the environment variable
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    

if __name__ == '__main__':
    
    with open("Mytools/config.yaml", 'r') as stream:
        cfg = yaml.safe_load(stream)

    ds_config = cfg['datasets']['kitti']
    seq = ds_config.get('seq')
    pc_path = ds_config.get('pc_path')
    gt_path = ds_config.get('gt_path')
    s_idx = ds_config.get('s_idx')
    e_idx = ds_config.get('e_idx')
    process_number = ds_config.get('process_number')

    Net_config = cfg['Networks']
    Save_path = Net_config.get('Save_path', './saved_model/model.h5')
    Batch_size = Net_config.get('Batch_size', 2)
    branch = Net_config.get('branch')
    time_size = Net_config.get('time_size')
    batch_gen = Net_config.get('batch_gen', False)
    data_pre = Net_config.get('data_pre', 'saved')
    rot_rep = Net_config.get('rot_rep', 'expn')

    if data_pre == 'saved_all' or data_pre == 'custom_gen':
        mother_folder_all = ''.join(['results'])
    else:
        mother_folder = create_folders_init()
        

    # __________________________________________________________________________________________________________________
    # process 1: Save Train Data
    # __________________________________________________________________________________________________________________
    if process_number == 1:
        # create_folders(mother_folder)

        if data_pre == 'saved':
            # __________________________
            # Prepare and save Geometry data (i-1, 1)
            # __________________________
            G_create = Geometry_data_prepare(mother_folder)
            G_create.create_geo_timedist()
            # G_create.load_data()
            # __________________________
            # Prepare and save Lidar data (i, i+1)
            # __________________________
            loader2 = Lidar_data_prepare(mother_folder)
            loader2.create_lidar_data_timedist()
            # loader2.create_lidar_data()

        if data_pre == 'saved_all' or data_pre == 'custom_gen':
            mother_folder_all = ''.join(['results'])
            # __________________________
            # Prepare and save Geometry data (i-1, 1)
            # __________________________
            if 1:
                G_create = Geometry_data_prepare(mother_folder_all)
                if rot_rep == 'expn':
                    G_create.create_geo_timedist_exp('_' + seq)
                elif rot_rep == 'dquart':
                    G_create.create_geo_icp('_' + seq)

            # __________________________
            # Prepare and save Lidar data (i, i+1)
            # __________________________
            if 1:
                loader2 = Lidar_data_prepare(mother_folder_all)
                loader2.create_line_by_line()

                # loader2.create_lidar_data_timedist('_'+seq)

        elif data_pre == 'tfrecord':
            recorder_reader = recorder_reader(mother_folder)
            recorder_reader.tf_recorder()

    # __________________________________________________________________________________________________________________
    # process 2: Train the Model
    # __________________________________________________________________________________________________________________
    elif process_number == 2:
        st = time.time()

        save_config(mother_folder_all)

        if data_pre == 'saved':
            # __________________________
            # Load Geometry data
            # __________________________
            G_create = Geometry_data_prepare(mother_folder)
            G_data, G_gt = G_create.load_saved_data()
            # __________________________
            # Load Lidar data
            # __________________________
            loader2 = Lidar_data_prepare(mother_folder)
            AI_data = loader2.load_saved_data()

            # __________________________
            # Start training
            # __________________________
            Basenet = BaseNet(mother_folder)
            model = Basenet.makemodel(G_data, G_gt, AI_data)

        elif data_pre == 'saved_all' or data_pre == 'custom_gen':
            mother_folder_save = ''.join(['results'])
            # mother_folder_save = os.path.join(mother_folder_save, 'supervised_all')
            # train_s = 100
            # train_end = 120
            # __________________________
            # Load Geometry data
            # __________________________
            # G_create = Geometry_data_prepare(mother_folder_save)
            # G_data, G_gt = G_create.load_saved_data_all()
            # __________________________
            # Load Lidar data
            # __________________________
            # loader2 = Lidar_data_prepare(mother_folder_save)
            # AI_data = loader2.load_saved_data_h5(train_s, train_end)

            # __________________________
            # Start training
            # __________________________
            Basenet = BaseNet(mother_folder_save)
            Basenet.makemodel()
            # model = Basenet.makemodel(G_data[train_s: train_end], G_gt[train_s: train_end], AI_data)

        else:
            Basenet = BaseNet(mother_folder)
            model = Basenet.makemodel()


        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds', '(', elapsed_time/3660, 'hours)')

    # __________________________________________________________________________________________________________________
    # process 3: Run Saved Model
    # __________________________________________________________________________________________________________________
    elif process_number == 3:
        st = time.time()
        if data_pre == 'saved':
            # __________________________
            # Load Geometry data
            # __________________________
            G_create = Geometry_data_prepare(mother_folder)
            G_data, G_gt = G_create.load_saved_data()
            # __________________________
            # Load Lidar data
            # __________________________
            loader2 = Lidar_data_prepare(mother_folder)
            AI_data = loader2.load_saved_data()
            # __________________________
            # Load model
            # __________________________
            Basenet = BaseNet(mother_folder)
            y = Basenet.load_savedmodel(G_data, AI_data)
            save2txt(y, 'result', mother_folder)

        elif data_pre == 'saved_all' or data_pre == 'custom_gen':
            # Basenet = BaseNet(mother_folder_all) #save results inside here
            # Basenet.load_savedmodel()

            Basenet = BaseNet(mother_folder_all)

            Basenet.predict_ones_fprimes()

            # Basenet.train_end()
            # Basenet.load_end()
        else:
            Basenet = BaseNet(mother_folder)  
            y = Basenet.load_savedmodel()  
            save2txt(y, 'result', mother_folder)

        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds', '(', elapsed_time/3660, 'hours)')

    # __________________________________________________________________________________________________________________
    # process 4: Visualize
    # __________________________________________________________________________________________________________________
    elif process_number == 4:
        if data_pre == 'saved_all' or data_pre == 'custom_gen':
            mother_folder_save = ''.join(['results'])
            # mother_folder_save = os.path.join(mother_folder_save, 'supervised_all')
            print(mother_folder_save)
            visclass = visall(mother_folder_save)
            # visclass.vis_bird_eye()
            # visclass.vis_traj()
            visclass.vis_history_all()
        else:
            visclass = visall(mother_folder)
            # visclass.vis_bird_eye()
            # visclass.vis_traj()
            visclass.vis_history()
            
#     elif process_number == 5:
        
#             vis_weights1 = vis_weights()
#             vis_weights1.vis_we()

    elif process_number == 6:
        Basenet = BaseNet(mother_folder_all)
        Basenet.read_tuner()
