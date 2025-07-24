

# # Enable dynamic memory growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# # Enable mixed precision training
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

import numpy as np
import yaml
import os
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import tensorflow_probability as tfp
from scipy.spatial import cKDTree
# from sklearn.neighbors import KDTree
from scipy.optimize import minimize, approx_fprime
import gc
import time
from datetime import datetime
import pytz
import pickle

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
import tensorflow.keras.backend as K
import keras_tuner as kt
# import open3d.ml.tf as ml3d
from tensorflow.keras import layers, Model, Input, backend, losses
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_graphics.geometry.transformation as tfg
import tensorflow_graphics.nn.loss as tfgn

# from typing import List
# from typing import Optional
# from typing import Tuple

from Mytools.make_pcfile_4network import Lidar_data_prepare
from Mytools.pre_geo_data import Geometry_data_prepare
from Mytools.tfrecord_tfread import recorder_reader
from Mytools.output_save import save2txt
# from Mytools.att_lstm import attention
# from Mytools.att_ww import cbam_block
# from Mytools.match_pc import chamfer_distance_tf
from Mytools.models_all import GeoLayer, GeoLayer_two, GeoLayer_intermed, GeoLayer_end
from Mytools.models_all import geo_s2e_att, geo_s2e_att_two
from Mytools.models_all import Lidar_layer, Lidar_layer_two, Lidar_layer_intermed
from Mytools.models_all import LiD_s2e_att, LiD_s2e_att_two
from Mytools.models_all import LiD_s2e_INAF, LiD_s2e_INAF_two
from Mytools.models_all import Combined_weighted, Combined_weighted_simple
from Mytools.models_all import Combined_s2e_nowe, Combined_s2e_nowe_two
from Mytools.models_all import Combined_s2e_nowe_simple, Combined_s2e_nowe_two_simple
from Mytools.models_all import Combined_s2e_att_simple, Combined_s2e_att_two_simple
from Mytools.models_all import Combined_s2e_att, Combined_s2e_att_two
from Mytools.models_all import Combined_s2e_INAF
from Mytools.models_all import Combined_s2e_soft_nowe, Combined_s2e_soft_nowe_two
from Mytools.models_all import geo_s2e_nowe, geo_s2e_nowe_two
from Mytools.models_all import LiD_s2e_nowe, LiD_s2e_nowe_two
from Mytools.visualize import visbar, vis_degree
from Mytools.my_custom_gen import file_order, My_Custom_Generator

# from Mytools.models_Li import model_LiDar_v1 ,model_LiDar_grad_v1
from Mytools.read_ct import ct_loader

from Mytools.my_tuner import build_model_geo, build_model_all, build_model_lidar, build_model_all_att


# tf.config.run_functions_eagerly(True)
# tf.compat.v1.disable_eager_execution()


    
# Clear session
tf.keras.backend.clear_session()

class BaseNet:
    def __init__(self, mother_folder, manual_id=None):

        # Read YAML file
        self.fprime = None
        with open('Mytools/config.yaml', 'r') as stream:
            cfg = yaml.safe_load(stream)

        ds_config = cfg['datasets']['kitti']
        self.image_width = ds_config.get('image-width', 1024)
        self.image_height = ds_config.get('image-height', 64)
        self.channels = ds_config['channels']
        self.channels_N = np.size(self.channels)
        self.s_idx = ds_config.get('s_idx')
        self.e_idx = ds_config.get('e_idx')
        self.sequences = ds_config.get('sequences')
        self.sequences_all = ds_config.get('sequences_all')
        self.scans = ds_config.get('scans')
        self.all_images_path = ds_config.get('all_images_path')

        Net_config = cfg['Networks']
        self.Batch_size = Net_config.get('Batch_size', 2)
        self.Epochs = Net_config.get('Epochs', 2)
        self.Save_path = Net_config.get('Save_path', './saved_model/model.h5')
        self.Saved_date_model = Net_config.get('saved_model')
        self.Saved_date_param = Net_config.get('saved_param')
        self.method = Net_config.get('method')
        self.branch = Net_config.get('branch')
        self.branch_mode = Net_config.get('branch_mode')
        self.loss_weights = Net_config.get('loss_weights')
        self.time_size = Net_config.get('time_size')
        self.batch_gen = Net_config.get('batch_gen', False)
        self.data_pre = Net_config.get('data_pre', 'saved')
        self.fusion = Net_config.get('fusion', 'simple')
        self.rot_rep = Net_config.get('rot_rep', 'expn')
        self.divided_train = Net_config.get('divided_train', 200)
        self.run_over = Net_config.get('run_over', False)
        self.pre_trained_model = Net_config.get('pre_trained_model')
        self.calculate_fprimes = Net_config.get('calculate_fprimes')
        self.hyper_overwrite = Net_config.get('hyper_overwrite', False)

        self.count = 0
        self.mother_folder = mother_folder

        self.manual_id = manual_id



        self.G_create = Geometry_data_prepare(self.mother_folder)
        self.recorder_reader = recorder_reader(self.mother_folder)
        
        # ______________________________________________________________________________________________________________
        # Make Folder
        # ______________________________________________________________________________________________________________
        now = datetime.now(pytz.timezone('America/Toronto'))
        self.folder_name = now.strftime("%Y_%m_%d_%H_%M")



        # combined2_____________________________________________________________________________________________________
        # __________________________
        # Input layers
        # __________________________
        geo_inp = layers.Input(shape=(self.time_size, 8), name='geo_input')
        lidar_inp = layers.Input(
            shape=(self.time_size, self.image_height, self.image_width, 2 * self.channels_N), name='AI_input')
        # __________________________
        # output = Function (input)
        # __________________________
        if self.branch_mode == 'geo':
            if self.fusion == 'direct':
                combined2_end = geo_s2e_nowe
            elif self.fusion == 'soft':
                combined2_end = geo_s2e_att
            # elif self.fusion == 'INAF':
            #     combined2_end = geo_s2e_nowe_INAF()
            
        elif self.branch_mode == 'lidar':
            if self.fusion == 'direct':
                combined2_end = LiD_s2e_nowe
            elif self.fusion == 'soft':
                combined2_end = LiD_s2e_att
            elif self.fusion == 'INAF':
                combined2_end = LiD_s2e_INAF
                
        elif self.branch_mode == 'all':
            if self.fusion == 'direct':
                # combined2_end = Combined_s2e_nowe
                combined2_end = Combined_s2e_nowe_simple
            elif self.fusion == 'soft':
                # combined2_end = Combined_s2e_att
                combined2_end = Combined_s2e_att_simple          
            elif self.fusion == 'INAF':
                combined2_end = Combined_s2e_INAF

        # __________________________
        # Model(inputs, outputs)
        # __________________________
        # self.Combined2 = Model(inputs=[geo_inp, lidar_inp], outputs=combined2_end)
        # self.Combined2 = Model(inputs={'geo_input': geo_inp, 'AI_input': lidar_inp}, outputs=combined2_end)
        # self.Combined2 = Model(inputs={'geo_input': geo_inp, 'AI_input': lidar_inp}, outputs=combined2_end)
        self.Combined2 = combined2_end

        
        # ___________________________________________________________________________________________________
        # Input layers
        # __________________________
        geo_inp = layers.Input(shape=(self.time_size, 8), name='geo_input')
        lidar_inp = layers.Input(
            shape=(self.time_size, self.image_height, self.image_width, 2 * self.channels_N), name='AI_input')
        # __________________________
        # output = Function (input)
        # __________________________
        if self.branch_mode == 'geo':
            if self.fusion == 'direct':
                combined2_end_two = geo_s2e_nowe_two
            elif self.fusion == 'soft':
                combined2_end_two = geo_s2e_att_two
            # elif self.fusion == 'INAF':
            #     combined2_end_two = geo_s2e_nowe_INAF()   
        elif self.branch_mode == 'lidar':
            if self.fusion == 'direct':
                combined2_end_two = LiD_s2e_nowe_two
            elif self.fusion == 'soft':
                combined2_end_two = LiD_s2e_att_two
            elif self.fusion == 'INAF':
                combined2_end_two = LiD_s2e_INAF_two
        elif self.branch_mode == 'all':
            if self.fusion == 'direct':
                # combined2_end_two = Combined_s2e_nowe_two
                combined2_end_two = Combined_s2e_nowe_two_simple
            elif self.fusion == 'soft':
                combined2_end_two = Combined_s2e_att_two_simple
            # elif self.fusion == 'INAF':
            #     combined2_end_two = LiD_s2e_INAF()
        # __________________________
        # Model(inputs, outputs)
        # __________________________
        # self.Combined2_two = Model(inputs=[geo_inp, lidar_inp], outputs=combined2_end_two)
        self.Combined2_two = combined2_end_two
    # __________________________________________________________________________________________________________________
    # Make Model
    # __________________________________________________________________________________________________________________
    # def my_dq_loss(y_true, y_pred):
    #     z = y_true * (y_pred * tf.constant([1., -1., -1., -1.000000000]))
    #     wtot = tf.reduce_sum(z,1)
    #     return tf.reduce_mean(2*tf.math.acos(K.clip(tf.math.sqrt(wtot*wtot), -1.,1.)))
    

    # __________________________________________________________________________________________________________________
    # Make Model
    # __________________________________________________________________________________________________________________
    def makemodel(self):

        model_combined2 = self.Combined2
        
        # ______________________________________________________________________________________________________________
        # Define a learning rate schedule for OPTIMIZER and CIMPILE
        # ______________________________________________________________________________________________________________
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.001,
        #     decay_steps=10000,
        #     decay_rate=0.9,
        #     staircase=True)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=False)
        
#         def combined_loss(y_true, y_pred, alpha=0.8):
#             mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
#             mse = tf.keras.losses.mean_squared_error(y_true, y_pred)

#             combined_loss = (1 - alpha) * mae + alpha * mse
#             # combined_loss = tf.clip_by_value(combined_loss, epsilon, 1e10)

#             return combined_loss
        combined_loss = tf.keras.losses.mean_squared_error
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1.0)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
        # optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

        '''
        # Create the optimizer with the learning rate schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model_combined2.compile(loss=tf.keras.losses.mean_squared_error, optimizer=optimizer, metrics=['mae'])
        # model_combined2.compile(loss=combined_loss, optimizer=optimizer, metrics=['mae'])
        '''
        if self.branch_mode == 'geo':
            if self.fusion == 'direct':
                build_model = lambda hp: build_model_geo(hp, model_combined2, combined_loss, optimizer)
        elif self.branch_mode == 'lidar':
            if self.fusion == 'direct':
                build_model = lambda hp: build_model_lidar(hp, model_combined2, combined_loss, optimizer)
        elif self.branch_mode == 'all': 
            if self.fusion == 'direct':
                build_model = lambda hp: build_model_all(hp, model_combined2, combined_loss, optimizer)
            if self.fusion == 'soft':
                build_model = lambda hp: build_model_all_att(hp, model_combined2, combined_loss, optimizer)
                    

        # ______________________________________________________________________________________________________________
        # 
        # ______________________________________________________________________________________________________________
        G_ict = ct_loader()
        G_create = Geometry_data_prepare(self.mother_folder)
        G_data, G_gt=  np.array([], dtype=np.float32).reshape(0, 4, 8), np.array([], dtype=np.float32).reshape(0, 8)
        G_data2, G_gt2=  np.array([], dtype=np.float32).reshape(0, 4, 8), np.array([], dtype=np.float32).reshape(0, 8)
        # G_data, G_gt=  np.array([], dtype=np.float32).reshape(0, 8), np.array([], dtype=np.float32).reshape(0, 8)
        AI_data = np.array([], dtype=np.float32).reshape(0, 4, 64, 720, 14)
        
        if self.branch_mode !='geo':
            loader2 = Lidar_data_prepare(self.mother_folder)
            
            
        # ______________________________________________________________________________________________________________
        # Run Over already trained network
        # ______________________________________________________________________________________________________________
        if self.run_over:
            # Call the model with some dummy data to build its variables
            dummy_geo_input = tf.zeros((2, 4, 8))
            dummy_lidar_input = tf.zeros((2, 4, 64, 720, 14))
            dummy_data = {'geo_input': dummy_geo_input, 'AI_input': dummy_lidar_input}
            dummy_output = model_combined2(dummy_data)   
            print("**********************************************************************************************")
            print("Running over a trained model")
            print("**********************************************************************************************")
            filename = os.path.join(self.mother_folder, 'saved_model', self.pre_trained_model, 'combined2.h5')
            model_combined2.load_weights(filename)
            
            
        train_loss_combined, val_loss_combined = [], []
        # ______________________________________________________________________________________________________________
        # Saved_all
        # ______________________________________________________________________________________________________________
        
        if self.data_pre == 'saved_all':

            for seqs in self.sequences_all:
                G_data_ic = G_ict.load_saved_data_all(seqs)              
                G_data_i, G_gt_i = G_create.load_saved_data_all(seqs)
                #__________
                # gt_tw = np.empty((len(G_gt_i) - self.time_size +1, self.time_size, 8))
                # for i in range(len(gt_tw)):
                #     gt_tw[i] = G_gt_i[i:i+self.time_size]
                
                # G_gt_i = G_gt_i[self.time_size-1:]
                #__________
                # G_data = np.vstack([G_data, gt_tw])
                G_data2 = np.vstack([G_data2, G_data_ic])
                G_data = np.vstack([G_data, G_data_i])
                G_gt = np.vstack([G_gt, G_gt_i])
                print(np.shape(G_data_ic), np.shape(G_data_i))
                print(np.shape(G_data), np.shape(G_data2))
                
            if self.branch_mode !='geo':
                for seqs in self.sequences_all:
                    AI_data_i = loader2.load_saved_data_h5_all(seqs)
                    print(AI_data_i.shape)
                    AI_data = np.vstack([AI_data, AI_data_i])
            
            if self.fusion == 'INAF':
                if self.branch_mode == 'all':
                    for seqs in self.sequences_all:
                        filepath = os.path.join(self.mother_folder, 'outputs', 'Intermediate_results', self.Saved_date_model, 
                                        'intermed_geo_seq_' + seqs + '.txt')
                        INAF_P_geo = np.loadtxt(filepath)
                        
                        filepath = os.path.join(self.mother_folder, 'outputs', 'Intermediate_results', self.Saved_date_model, 
                                        'intermed_lidar_seq_' + seqs + '.txt')
                        INAF_P_Li = np.loadtxt(filepath)
                        
                        filepath = os.path.join(self.mother_folder, 'outputs', 'Intermediate_results', self.Saved_date_model, 
                                        'intermed_we_lidar_seq_' + seqs + '.txt')
                        INAF_W = np.loadtxt(filepath)
                else:                    
                    for seqs in self.sequences_all:
                        filepath = os.path.join(self.mother_folder, 'outputs', 'Intermediate_results', self.Saved_date_model, 
                                        'intermed_lidar_seq_' + seqs + '.txt')
                        INAF_P = np.loadtxt(filepath)
                        filepath = os.path.join(self.mother_folder, 'outputs', 'Intermediate_results', self.Saved_date_model, 
                                        'intermed_we_lidar_seq_' + seqs + '.txt')
                        INAF_W = np.loadtxt(filepath)

            print('All_sequence', ': ', 'G_data_shape', G_data.shape, 'G_gt_shape', G_gt.shape, "___________________________")
            print('All_sequence', ': ', 'AI_dat_shape', AI_data.shape, "___________________________")

            mu = self.divided_train
            counter=0

            # temp_start = counter * mu
            # temp_end = temp_start + mu

            # if seq_scans - 5 < temp_end: temp_end = seq_scans - 5

            # ______________________________________________________________________________________________________________
            # Data Prepare
            # ______________________________________________________________________________________________________________
            # __________________________________________GEO
            
            G_data_temp = G_data
            G_gt_temp = G_gt
            print(np.shape(G_data_temp), np.shape(G_gt_temp), np.shape(AI_data))
            # __________________________________________LiDAR
            if self.branch_mode == 'geo':
                AI_data_temp = tf.zeros([G_data_temp.shape[0], G_data_temp.shape[1]])
            else:
                # AI_data_temp = loader2.load_saved_data_h5_all(seq)
                AI_data_temp = AI_data
            # print('AI_shape', tf.shape(AI_data_temp))

            x_geo, y = G_data_temp, G_gt_temp

            if self.fusion == 'INAF':
                if self.branch_mode == 'all':
                    x_combined = {'Param_geo': INAF_P_geo, 'Param_Li': INAF_P_Li, 'Weight': INAF_W}
                    print('x_combined shapes', np.shape(INAF_P_geo), np.shape(INAF_P_Li), np.shape(INAF_W))
                else:
                    x_combined = {'Param': INAF_P, 'Weight': INAF_W}
                    print('x_combined shapes', np.shape(INAF_P), np.shape(INAF_W))
            else:
                print(np.shape(x_geo), np.shape(AI_data_temp))
                x_combined = {'geo_input': x_geo, 'AI_input': AI_data_temp}
            
            #___________________________________________________________________________
            def load_and_split_data(x_dict, y, batch_size=8, validation_split=0.2, shuffle_buffer_size=1000):
                # Combine x_dict and y into a single dataset
                dataset = tf.data.Dataset.from_tensor_slices((x_dict, y))

                # Split the dataset into training and validation sets
                num_samples = sum(1 for _ in dataset)
                num_validation_samples = int(num_samples * validation_split)
                dataset = dataset.shuffle(shuffle_buffer_size)
                
                train_dataset = dataset.skip(num_validation_samples)
                validation_dataset = dataset.take(num_validation_samples)

                # Load the training and validation datasets
                train_dataset = load_data(train_dataset, batch_size)
                validation_dataset = load_data(validation_dataset, batch_size)

                return train_dataset, validation_dataset
            
            def load_data(dataset, batch_size):
                dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
                return dataset

            # ______________________________________________________________________________________________________________
            # Training starts
            # ______________________________________________________________________________________________________________
            
            # ________________________________________________________________Set up the tuner
            train_dataset, validation_dataset = load_and_split_data(x_combined, y)

            tuner = kt.Hyperband(build_model,
                                 # objective='val_mae',
                                 objective='val_loss',
                                 max_epochs=10,
                                 factor=3,
                                 directory='hyperparam',
                                 project_name=self.branch_mode+self.fusion,
                                 overwrite=self.hyper_overwrite)
            
            # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
            
            # Run the tuner
            tuner.search(train_dataset,
                         validation_data=validation_dataset,
                         epochs=10, 
                         # callbacks=[early_stopping, reduce_lr]
                         callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
                         # callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
                         )


            # Get the best hyperparameters and model
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_model = tuner.hypermodel.build(best_hps)
            
            print("The hyperparameter search is complete. The optimal hyperparameters are:")
            hps_dict = best_hps.get_config()['values']
            for hp_name, hp_value in hps_dict.items():
                print(f"- {hp_name}: {hp_value}")
                           
            
            model_dir = os.path.join(self.mother_folder, 'saved_model', self.folder_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            checkpoint_filepath = os.path.join(model_dir, 'combined2_best_weights.h5')
            
            
            checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                         monitor='val_loss',
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='min',
                                         verbose=1)

            train_history_combined = best_model.fit(
                train_dataset,
                # x=x_combined, y=y,
                validation_data=validation_dataset,
                epochs=self.Epochs,
                batch_size=self.Batch_size,
                verbose=1,
                callbacks=[checkpoint]
                )


            train_loss_combined.append(train_history_combined.history['loss'])
            val_loss_combined.append(train_history_combined.history['val_loss'])

            gc.collect()
            
    
#         elif self.data_pre == 'custom_gen':
#             # print("here___")

#             initialize_file_order = file_order(self.mother_folder)
#             initialize_file_order.forward_file_order()

#             file_names_dir = os.path.join(self.all_images_path, 'file_names_shuffles.npy')
#             file_names_dir_val = os.path.join(self.all_images_path, 'file_names_shuffles_val.npy')
#             Geo_dir = os.path.join(self.all_images_path, 'Geo_shuffles.npy')
#             Geo_dir_val = os.path.join(self.all_images_path, 'Geo_shuffles_val.npy')
#             gt_dir = os.path.join(self.all_images_path, 'gt_shuffles.npy')
#             gt_dir_val = os.path.join(self.all_images_path, 'gt_shuffles_val.npy')

#             X_train_filenames = np.load(file_names_dir)
#             X_valid_filenames = np.load(file_names_dir_val)
#             Ge_train = np.load(Geo_dir)
#             Ge_valid = np.load(Geo_dir_val)
#             gt_train = np.load(gt_dir)
#             gt_valid = np.load(gt_dir_val)

#             x_inputs = [Ge_train, X_train_filenames]
#             y_output = gt_train

#             x_inputs_valid = [Ge_valid, X_valid_filenames]
#             y_output_valid = gt_valid

#             my_training_batch_generator = My_Custom_Generator(x_inputs, y_output, self.Batch_size)
#             my_validation_batch_generator = My_Custom_Generator(x_inputs_valid, y_output_valid, self.Batch_size)
            
            
#             # ________________________________________________________________Set up the tuner
#             tuner = kt.RandomSearch(
#                 build_model,
#                 objective='val_mae',
#                 max_trials=50,  # Number of hyperparameter configurations to try
#                 executions_per_trial=2,  # Number of models to train per trial
#                 directory='hyperparam',
#                 project_name='all_s2e_nowe_tuning',
#                 overwrite=True
#             )
#             # Run the tuner
#             tuner.search(
#                 x=my_training_batch_generator, 
#                 validation_data=my_validation_batch_generator,
#                 epochs=50, 
#                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
#             )

#             # Get the best hyperparameters and model
#             best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#             best_model = tuner.hypermodel.build(best_hps)   
            
#             train_history_combined = best_model.fit(
#                 x=my_training_batch_generator,
#                 validation_data=my_validation_batch_generator,
#                 epochs=50,
#                 verbose=1)
#             # train_history_combined = model_combined2.fit(
#             #     x=my_training_batch_generator,
#             #     validation_data=my_validation_batch_generator,
#             #     epochs=self.Epochs,
#             #     verbose=1)

#             train_loss_combined.append(train_history_combined.history['loss'])
#             val_loss_combined.append(train_history_combined.history['val_loss'])

        # ______________________________________________________________________________________________________________
        # Save Model
        # ______________________________________________________________________________________________________________
        ##

#         current_time_path = os.path.join(self.mother_folder, 'saved_model', self.folder_name)
#         os.mkdir(current_time_path)
        
#         filename = os.path.join(self.mother_folder, 'saved_model', self.folder_name, 'combined2.h5')
#         best_model.save_weights(filename)
        

        print('_______________________________________________________________________________________________________')
        print('Model GEO and LiDAR saved to disk')
        print('_______________________________________________________________________________________________________')

        # ______________________________________________________________________________________________________________
        # PLOT History
        # ______________________________________________________________________________________________________________
        
        filename = os.path.join(self.mother_folder, 'saved_model', self.folder_name, 'train_loss.npy')
        np.save(filename, train_loss_combined)
        filename = os.path.join(self.mother_folder, 'saved_model', self.folder_name, 'val_loss.npy')
        np.save(filename, val_loss_combined)
        # filename = os.path.join(self.mother_folder, 'results', 'n_epochs_best.npy')
        # np.save(filename, n_epochs_best)

        # plt.show()
        # return model_geo
        
        
    def read_tuner(self):
        tuner = kt.Hyperband(
            # project_name=self.branch_mode+self.fusion,
            project_name='geo_s2e_nowe_tuning_all',
            hypermodel=None, 
            objective='val_accuracy',
            max_epochs=10,
            directory='hyperparam')


        tuner.reload()
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        hps_dict = best_hps.get_config()['values']

        print("The best hyperparameters are:")
        for hp_name, hp_value in hps_dict.items():
            print(f"- {hp_name}: {hp_value}")

    def predict_ones_fprimes(self):    
        
        # filename = os.path.join(self.mother_folder, 'saved_model', self.Saved_date_model, 'combined2.h5')
        filename = os.path.join(self.mother_folder, 'saved_model', self.Saved_date_model, 'combined2_best_weights.h5')
        if self.fusion == 'INAF':
            model_combined2_two = self.Combined2_two
            model_combined2_two.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam', metrics=['mae'])
            # Call the model with some dummy data to build its variables
            dummy_param_input = tf.zeros((1, 32))
            dummy_w_input = tf.zeros((1, 32))
            dummy_data = {'Param': dummy_param_input, 'Weight': dummy_w_input}
            dummy_output = model_combined2_two(dummy_data) 
            model_combined2_two.load_weights(filename)
        else:

            model_combined2_two = self.Combined2_two
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=10000,
                decay_rate=0.9,
                staircase=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            
            def combined_loss(y_true, y_pred, alpha=0.8):
                mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
                mse = tf.keras.losses.mean_squared_error(y_true, y_pred)

                combined_loss = (1 - alpha) * mae + alpha * mse

                return combined_loss
            
            if self.branch_mode == 'geo':
                build_model = lambda hp: build_model_geo(hp, model_combined2_two, combined_loss, optimizer)
            elif self.branch_mode == 'lidar':
                build_model = lambda hp: build_model_lidar(hp, model_combined2_two, combined_loss, optimizer)
            elif self.branch_mode == 'all':    
                build_model = lambda hp: build_model_all(hp, model_combined2_two, combined_loss, optimizer)
            
            tuner = kt.Hyperband(build_model,
                                 # objective='val_mae',
                                 objective='loss',
                                 max_epochs=50,
                                 factor=3,
                                 directory='hyperparam',
                                 project_name=self.branch_mode+self.fusion,
                                 overwrite = False)
            
            # Get the best hyperparameters and model
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model_combined2_two = tuner.hypermodel.build(best_hps)

            # model_combined2_two.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam', metrics=['mae'])
            # Call the model with some dummy data to build its variables
            dummy_geo_input = tf.zeros((2, 4, 8))
            dummy_lidar_input = tf.zeros((2, 4, 64, 720, 14))
            dummy_data = {'geo_input': dummy_geo_input, 'AI_input': dummy_lidar_input}
            dummy_output = model_combined2_two(dummy_data) 
            model_combined2_two.load_weights(filename)
            

        
        G_create = Geometry_data_prepare(self.mother_folder)
        loader2 = Lidar_data_prepare(self.mother_folder)
        
        save_directory = os.path.join(self.mother_folder, 'outputs', 'output_trajectory_parts', self.Saved_date_model)
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        
        if self.calculate_fprimes:            
            save_directory2 = os.path.join(self.mother_folder, 'outputs', 'Intermediate_results', self.Saved_date_model)
            if not os.path.exists(save_directory2):
                os.mkdir(save_directory2)
        if self.fusion == 'INAF':
            print('ff00')
            for seqs in self.sequences_all:
                filepath = os.path.join(self.mother_folder, 'outputs', 'Intermediate_results', self.Saved_date_param, 
                                'intermed_lidar_seq_' + seqs + '.txt')
                INAF_P = np.loadtxt(filepath)
                filepath = os.path.join(self.mother_folder, 'outputs', 'Intermediate_results', self.Saved_date_param, 
                                'intermed_we_lidar_seq_' + seqs + '.txt')
                INAF_W = np.loadtxt(filepath)

        for seq in self.sequences:
            seq_int = int(seq)
            seq_scans = int(self.scans[seq_int])

            print("_______________________________________________________________________________________________"
                  "_______________________________________________________________________________")
            print('Sequence started: ', seq)
            print("_______________________________________________________________________________________________"
                  "_______________________________________________________________________________")

            # G_data, G_gt = G_create.load_saved_data_all(seq)

            G_data_ic = G_ict.load_saved_data_all(seqs)              
                
            gt_tw = np.empty((len(G_gt) - self.time_size +1, self.time_size, 8))
            for i in range(len(gt_tw)):
                gt_tw[i] = G_gt[i:i+self.time_size]

            # G_gt = G_gt[self.time_size-1:]
            
            # mu = self.divided_train
            mu = seq_scans
            seq_scans = seq_scans
            inside_loop = int(np.ceil(np.divide(seq_scans, mu)))
            
            if self.data_pre == 'saved_all' or self.data_pre == 'custom_gen':
                if self.fusion == 'INAF':
                    print('ff')
                    x_combined = {'Param': INAF_P, 'Weight': INAF_W}
                    y_combined2 = model_combined2_two.predict(x=x_combined, verbose=1)

                    Tlast = np.eye(4)
                    counter=0
                    temp_start = counter * mu
                    temp_end = seq_scans - self.time_size
                    
                    Tlast_next = save2txt(y_combined2, 'result_all_dq', self.mother_folder, file_id=seq, part=0,
                                      start_i=temp_start + self.time_size, end_i=temp_end + self.time_size,
                                      Tlast_counter=Tlast)

                    gc.collect()                    
                else:
                    
                    for counter in range(inside_loop):
                        temp_start = counter * mu
                        # if counter == 0: temp_start = 5
                        temp_end = temp_start + mu
                        # if seq_scans - 5 < temp_end: temp_end = seq_scans - 5
                        if seq_scans < temp_end: temp_end = seq_scans

                        print('Sequence', seq, 'part', counter, ': ', temp_start, 'to', temp_end, 'of', seq_scans + 5,
                              '***************************************************************')
                        # print(temp_start, temp_end, np.shape(G_data), temp_end-self.time_size+1)
                        # G_data_temp = G_data[temp_start+1: temp_end-self.time_size+1]
                        G_data_temp = G_data
                        # G_data_temp = gt_tw

                        if self.branch_mode == 'geo':
                            AI_data_temp = tf.zeros([G_data_temp.shape[0], G_data_temp.shape[1]])
                        else:
                            AI_data_temp = loader2.load_saved_data_h5(seq, temp_start, temp_end+1)
                            # AI_data_temp = loader2.load_saved_data_h5_all(seq)

                        print("___", np.shape(G_data_temp), np.shape(AI_data_temp))
                        x_geo = G_data_temp
                        x_lidar = AI_data_temp

                        print(np.shape(x_geo), np.shape(x_lidar))
                        x_combined = {'geo_input': x_geo, 'AI_input': x_lidar}
                        
                        
                        def load_test_data(x_dict, batch_size=8):
                            dataset = tf.data.Dataset.from_tensor_slices(x_dict)
                            test_dataset = load_data(dataset, batch_size)
                            return test_dataset

                        def load_data(dataset, batch_size):
                            dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
                            return dataset
                        
                        test_dataset = load_test_data(x_combined, batch_size=self.Batch_size)

                        # ______________________________________________________________________________________________________
                        # Intermediate features
                        # ______________________________________________________________________________________________________

                        # intermed_geo = model_geo_3.predict(x=x_geo, verbose=1)
                        # intermed_lidar = model_lidar_3.predict(x=x_lidar, verbose=1)

                        # ______________________________________________________________________________________________________
                        # Calculate gradients
                        # ______________________________________________________________________________________________________


                        #'''
                        y_combined2, c_geo_intermed, c_lidar_intermed, \
                            my_grad0, my_grad1, my_grad2, my_grad3, my_grad4, my_grad5, my_grad6, my_grad7 = \
                            model_combined2_two.predict(x=test_dataset, verbose=1)
                        # print("cc", np.shape(c_lidar_intermed))
                        combined_grad_np = np.squeeze(
                            np.array([my_grad0, my_grad1, my_grad2, my_grad3, my_grad4, my_grad5, my_grad6, my_grad7]))
                        combined_grad_np = combined_grad_np.transpose((1, 0, 2))   # Batch* 4* 64 * 8
                        print(np.shape(combined_grad_np))                        


                        # ______________________________________________________________________________________________________
                        # Calculate FPRIME
                        # ______________________________________________________________________________________________________
                        # geo_fprime = np.empty((0, 8), np.float)
                        # lidar_fprime = np.empty((0, 8), np.float)

                        if self.calculate_fprimes:
                            combined_fprime = np.empty((0, 8), float)
                            for j in range(len(y_combined2)):
                                if j % 50 == 0: print('fprime at', j + 1, 'of', len(y_combined2))
                                # y0_g, y0_l = y_geo[j], y_lidar[j]
                                y0_c = y_combined2[j]
                                AI_data_temp0 = AI_data_temp[j]

                                # geo_fprime_temp = np.array(
                                #     approx_fprime(y0_g, cal_grad_one, np.sqrt(np.finfo(float).eps), AI_data_temp0))
                                # geo_fprime = np.vstack([geo_fprime, geo_fprime_temp])
                                #
                                # lidar_fprime_temp = np.array(
                                #     approx_fprime(y0_l, cal_grad_one, np.sqrt(np.finfo(float).eps), AI_data_temp0))
                                # lidar_fprime = np.vstack([lidar_fprime, lidar_fprime_temp])

                                combined_fprime_temp = np.array(
                                    approx_fprime(y0_c, cal_grad_one, np.sqrt(np.finfo(float).eps), AI_data_temp0))
                                combined_fprime = np.vstack([combined_fprime, combined_fprime_temp])
                                # print(np.shape(combined_fprime))

                            combined_fprime = tf.expand_dims(combined_fprime, axis=1)
                            # print(np.shape(combined_fprime), np.shape(combined_grad_np))
                            layer_grad_combined = tf.squeeze(tf.matmul(combined_fprime, combined_grad_np))
                            print(np.shape(layer_grad_combined))
                            
                            if self.branch_mode == 'geo':
                                save2txt(c_geo_intermed, 'intermed_geo', self.mother_folder, file_id=seq, part=counter,
                                         start_i=temp_start + self.time_size, end_i=temp_end + self.time_size)
                            elif self.branch_mode == 'lidar':
                                save2txt(c_lidar_intermed, 'intermed_lidar', self.mother_folder, file_id=seq, part=counter,
                                         start_i=temp_start + self.time_size, end_i=temp_end + self.time_size)
                            elif self.branch_mode == 'all':
                                save2txt(c_geo_intermed, 'intermed_geo', self.mother_folder, file_id=seq, part=counter,
                                         start_i=temp_start + self.time_size, end_i=temp_end + self.time_size)
                                save2txt(c_lidar_intermed, 'intermed_lidar', self.mother_folder, file_id=seq, part=counter,
                                         start_i=temp_start + self.time_size, end_i=temp_end + self.time_size)
                                
                            save2txt(layer_grad_combined, 'intermed_w_lidar', self.mother_folder, file_id=seq, part=counter,
                                     start_i=temp_start + self.time_size, end_i=temp_end + self.time_size)

                        if counter == 0:
                            Tlast = np.eye(4)
                        else:
                            Tlast = Tlast_next
                            # print(Tlast_next)
                        print(temp_start + self.time_size-1,temp_end + self.time_size)
                        Tlast_next = save2txt(y_combined2, 'result_all_dq', self.mother_folder, file_id=seq, part=counter,
                                          start_i=temp_start + self.time_size-1, end_i=temp_end + self.time_size,
                                          Tlast_counter=Tlast)

                        gc.collect()
                
#             elif self.data_pre == 'custom_gen':
#                 # print("here")

#                 initialize_file_order = file_order(self.mother_folder)
#                 initialize_file_order.file_order()

#                 file_names_dir = os.path.join(self.all_images_path, 'file_names_shuffles.npy')
#                 file_names_dir_val = os.path.join(self.all_images_path, 'file_names_shuffles_val.npy')
#                 Geo_dir = os.path.join(self.all_images_path, 'Geo_shuffles.npy')
#                 Geo_dir_val = os.path.join(self.all_images_path, 'Geo_shuffles_val.npy')
#                 gt_dir = os.path.join(self.all_images_path, 'gt_shuffles.npy')
#                 gt_dir_val = os.path.join(self.all_images_path, 'gt_shuffles_val.npy')

#                 X_train_filenames = np.load(file_names_dir)
#                 X_valid_filenames = np.load(file_names_dir_val)
#                 Ge_train = np.load(Geo_dir)
#                 Ge_valid = np.load(Geo_dir_val)
#                 gt_train = np.load(gt_dir)
#                 gt_valid = np.load(gt_dir_val)

#                 x_inputs = [Ge_train, X_train_filenames]
#                 y_output = gt_train

#                 x_inputs_valid = [Ge_valid, X_valid_filenames]
#                 y_output_valid = gt_valid

#                 my_training_batch_generator = My_Custom_Generator(x_inputs, y_output, self.Batch_size)
#                 my_validation_batch_generator = My_Custom_Generator(x_inputs_valid, y_output_valid, self.Batch_size)
                
#                 y_combined2, c_geo_intermed, c_lidar_intermed, my_grad0, my_grad1, my_grad2, my_grad3, my_grad4, my_grad5, my_grad6, my_grad7 = \
#                 model_combined2_two.predict(x=my_training_batch_generator, verbose=1)

#                 combined_grad_np = np.squeeze(
#                     np.array([my_grad0, my_grad1, my_grad2, my_grad3, my_grad4, my_grad5, my_grad6, my_grad7]))
#                 combined_grad_np = combined_grad_np.transpose((1, 2, 3, 0))


                # geo_fprime = np.repeat(geo_fprime, 4, axis=0)
                # geo_fprime = np.reshape(geo_fprime, (-1, 4, 8))
                # geo_fprime = np.expand_dims(geo_fprime, axis=3)
                # layer_grad_geo = tf.squeeze(tf.matmul(geo_grad_np, geo_fprime))
                #
                # lidar_fprime = np.repeat(lidar_fprime, 4, axis=0)
                # lidar_fprime = np.reshape(lidar_fprime, (-1, 4, 8))
                # lidar_fprime = np.expand_dims(lidar_fprime, axis=3)
                # layer_grad_lidar = tf.squeeze(tf.matmul(lidar_grad_np, lidar_fprime))

                # ______________________________________________________________________________________________________

                # save2txt(intermed_geo, 'intermed_geo', self.mother_folder, file_id=seq, part=counter,
                #          start_i=temp_start + 5, end_i=temp_end + 5)
                # save2txt(layer_grad_geo, 'intermed_w_geo', self.mother_folder, file_id=seq, part=counter,
                #          start_i=temp_start + 5, end_i=temp_end + 5)
                #
                # save2txt(intermed_lidar, 'intermed_lidar', self.mother_folder, file_id=seq, part=counter,
                #          start_i=temp_start + 5, end_i=temp_end + 5)
                # save2txt(layer_grad_lidar, 'intermed_w_lidar', self.mother_folder, file_id=seq, part=counter,
                #          start_i=temp_start + 5, end_i=temp_end + 5)


                # Tlast = np.eye(4)
                # Tlast_next = save2txt(y_combined2, 'result_all_dq', self.mother_folder, file_id=seq, part=0,
                #                       start_i=0, end_i=temp_end + self.time_size,
                #                       Tlast_counter=Tlast)

        gc.collect()

    def train_end(self):
        what_model = 'all'
        G_create = Geometry_data_prepare(self.mother_folder)
        # ______________________________________________________________________________________________________________
        # Prepare model end
        # ______________________________________________________________________________________________________________
        if what_model == 'geo':
            model_end = self.geo_model_end
            model_end.summary()
            filename = os.path.join(self.mother_folder, 'model_geo_end.png')
            tf.keras.utils.plot_model(model_end, show_shapes=True, to_file=filename)
            model_end.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam')

        elif what_model == 'all':
            model_end = self.Combined_weighted
            model_end.summary()
            filename = os.path.join(self.mother_folder, 'model_combined_end.png')
            tf.keras.utils.plot_model(model_end, show_shapes=True, to_file=filename)
            model_end.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam')

        for seq in self.sequences:
            seq_int = int(seq)
            # seq_scans = int(self.scans[seq_int])

            G_data, G_gt = G_create.load_saved_data_all(seq)
            print("_______________________________________________________________________________________________"
                  "_______________________________________________________________________________")
            print('Sequence started: ', seq)
            print("_______________________________________________________________________________________________"
                  "_______________________________________________________________________________")
            #  _________________________________________________________________________________________________________
            # Load Intermed and weights
            #  _________________________________________________________________________________________________________
            filepath = os.path.join(self.mother_folder, 'outputs', 'output_trajectory_parts', 'Intermediate_results',
                                    'intermed_geo_seq_' + seq + '.txt')
            geo_intermed_flat = np.loadtxt(filepath)
            geo_intermed = geo_intermed_flat.reshape((geo_intermed_flat.shape[0], self.time_size, 64))

            filepath = os.path.join(self.mother_folder, 'outputs', 'output_trajectory_parts', 'Intermediate_results',
                                    'intermed_we_geo_seq_' + seq + '.txt')
            geo_layer_grad_flat = np.loadtxt(filepath)
            geo_layer_grad = geo_layer_grad_flat.reshape((geo_layer_grad_flat.shape[0], self.time_size, 64))

            #  _________________________________________________________________________________________________________

            filepath = os.path.join(self.mother_folder, 'outputs', 'output_trajectory_parts', 'Intermediate_results',
                                    'intermed_lidar_seq_' + seq + '.txt')
            lidar_intermed_flat = np.loadtxt(filepath)
            lidar_intermed = lidar_intermed_flat.reshape((lidar_intermed_flat.shape[0], self.time_size, 64))

            filepath = os.path.join(self.mother_folder, 'outputs', 'output_trajectory_parts', 'Intermediate_results',
                                    'intermed_we_lidar_seq_' + seq + '.txt')
            lidar_layer_grad_flat = np.loadtxt(filepath)
            lidar_layer_grad = lidar_layer_grad_flat.reshape((lidar_layer_grad_flat.shape[0], self.time_size, 64))

            #  _________________________________________________________________________________________________________

            Visbar = visbar(self.mother_folder, seq)
            Visbar.bar_all(geo_layer_grad, name='geo')
            Visbar.bar_all(lidar_layer_grad, name='lidar')
            # for j in range(layer_grad_flat.shape[0]):
            #     intermed_1 = intermed[j, -1]
            #     plt.bar(np.arange(64), intermed_1, alpha=0.3)
            #     plt.show(block=False)
            #     plt.pause(0.1)
            # plt.close()

            #  _________________________________________________________________________________________________________
            # Save and Compile Geo_end
            #  _________________________________________________________________________________________________________
            # x_intermed = {'geo_i': intermed, 'geo_w': layer_grad}
            # y_end = G_gt
            # train_history = model_end.fit(
            #     x=x_intermed,
            #     y=y_end,
            #     epochs=self.Epochs, batch_size=self.Batch_size,
            #     validation_split=0.1,
            #     verbose=1)

            x_intermed = {'geo_intermed': geo_intermed, 'geo_weight': geo_layer_grad,
                          'lidar_intermed': lidar_intermed, 'lidar_weight': lidar_layer_grad, }
            y_end = G_gt
            train_history = model_end.fit(
                x=x_intermed,
                y=y_end,
                epochs=self.Epochs, batch_size=self.Batch_size,
                validation_split=0.1,
                verbose=1)
        # ______________________________________________________________________________________________________________
        # Save Model
        # ______________________________________________________________________________________________________________
        filename = os.path.join(self.mother_folder, 'saved_model', 'model_end.h5')
        # model_end.save(filename)
        model_end.save_weights(filename)
        print('_______________________________________________________________________________________________________')
        print('Model saved to disk')
        print('_______________________________________________________________________________________________________')

        gc.collect()

    def load_end(self):

        # filename = os.path.join(self.mother_folder, 'results', 'saved_model', 'model_geo_end.h5')
        # model_geo_end = self.geo_model_end
        # model_geo_end.load_weights(filename, by_name=True)

        filename = os.path.join(self.mother_folder, 'saved_model', 'model_end.h5')
        model_combined = self.Combined_weighted
        model_combined.load_weights(filename, by_name=True)

        G_create = Geometry_data_prepare(self.mother_folder)

        for seq in self.sequences:
            seq_int = int(seq)
            # seq_scans = int(self.scans[seq_int])

            G_data, G_gt = G_create.load_saved_data_all(seq)
            print("_______________________________________________________________________________________________"
                  "_______________________________________________________________________________")
            print('Sequence started: ', seq)
            print("_______________________________________________________________________________________________"
                  "_______________________________________________________________________________")

            #  _________________________________________________________________________________________________________
            # Load Intermed and weights
            #  _________________________________________________________________________________________________________
            filepath = os.path.join(self.mother_folder, 'outputs', 'output_trajectory_parts', 'Intermediate_results',
                                    'intermed_geo_seq_' + seq + '.txt')
            geo_intermed_flat = np.loadtxt(filepath)
            geo_intermed = geo_intermed_flat.reshape((geo_intermed_flat.shape[0], self.time_size, 64))

            filepath = os.path.join(self.mother_folder, 'outputs', 'output_trajectory_parts', 'Intermediate_results',
                                    'intermed_we_geo_seq_' + seq + '.txt')
            geo_layer_grad_flat = np.loadtxt(filepath)
            geo_layer_grad = geo_layer_grad_flat.reshape((geo_layer_grad_flat.shape[0], self.time_size, 64))

            #  _________________________________________________________________________________________________________

            filepath = os.path.join(self.mother_folder, 'outputs', 'output_trajectory_parts', 'Intermediate_results',
                                    'intermed_lidar_seq_' + seq + '.txt')
            lidar_intermed_flat = np.loadtxt(filepath)
            lidar_intermed = lidar_intermed_flat.reshape((lidar_intermed_flat.shape[0], self.time_size, 64))

            filepath = os.path.join(self.mother_folder, 'outputs', 'output_trajectory_parts', 'Intermediate_results',
                                    'intermed_we_lidar_seq_' + seq + '.txt')
            lidar_layer_grad_flat = np.loadtxt(filepath)
            lidar_layer_grad = lidar_layer_grad_flat.reshape((lidar_layer_grad_flat.shape[0], self.time_size, 64))

            #  _________________________________________________________________________________________________________

            #  _________________________________________________________________________________________________________
            # Save and Compile Geo_end
            #  _________________________________________________________________________________________________________
            x_intermed = {'geo_intermed': geo_intermed, 'geo_weight': geo_layer_grad,
                          'lidar_intermed': lidar_intermed, 'lidar_weight': lidar_layer_grad, }
            y = model_combined.predict(x=x_intermed, verbose=1)
            print(y.shape)
            gc.collect()

            Tlast_next = save2txt(y, 'result_all_dq', self.mother_folder, file_id=seq, part=0,
                                  start_i=5, end_i=2750, Tlast=np.eye(4))

            Vis_err = vis_degree(self.mother_folder, seq)
            Vis_err.degree_all(G_data, y)


# ______________________________________________________________________________________________________________________
# Tools
# ______________________________________________________________________________________________________________________


# class my_custom_model(Model):
class my_custom_model:
    def __init__(self, geo_branch, li_branch, fusion, *args, **kwargs):
        # super(my_custom_model, self).__init__(*args, **kwargs)

        self.Geo_model = geo_branch
        self.Lidar_model = li_branch
        self.fusion = fusion

    @tf.function
    def custom_grad(self, inputs, last_layer):
        with tf.GradientTape() as g:
            g.watch(last_layer)
            pc1_raw, pc2_raw = tf.split(inputs, num_or_size_splits=2, axis=4)
            # B x T x W x H x Channels
            s0, s1, s2, s3, s4 = pc1_raw.shape[0], pc1_raw.shape[1], pc1_raw.shape[2], pc1_raw.shape[3], pc1_raw.shape[
                4]

            pc1 = tf.reshape(pc1_raw[:, -1, :, :, 0:3], shape=[-1, s2 * s3, 3])
            pc2 = tf.reshape(pc2_raw[:, -1, :, :, 0:3], shape=[-1, s2 * s3, 3])

            # normal2 = tf.reshape(pc2_raw[:, -1, :, :, 3:6], [-1, s2 * s3, 3])
            # normal1 = tf.reshape(pc1_raw[:, -1, :, :, 3:6], [-1, s2 * s3, 3])

            output = layers.LSTM(8,
                                 activation="tanh",
                                 recurrent_activation="sigmoid",
                                 # recurrent_dropout=0.2,
                                 unroll=False,
                                 use_bias=True,
                                 name='exp_map'
                                 )(last_layer)

            Rq, Tr3 = tfg.dual_quaternion.to_rotation_translation(output)
            R33 = tfg.rotation_matrix_3d.from_quaternion(Rq)
            RT = tf.concat([R33, tf.expand_dims(Tr3, axis=2)], -1)
            RT = tf.pad(RT, [[0, 0], [0, 1], [0, 0]], constant_values=[0.0, 0.0, 0.0, 1.0])

            pc1 = tf.pad(pc1, [[0, 0], [0, 0], [0, 1]], constant_values=1)
            pc1 = tf.transpose(pc1, perm=[0, 2, 1])
            pc1_tr = tf.linalg.matmul(RT, pc1)
            pc1_tr = pc1_tr[:, 0:3]
            pc1_tr = tf.transpose(pc1_tr, perm=[0, 2, 1])  # B x WH x 3

            print(pc1_tr.shape, pc2.shape)

            # for epoch in range(self.Epochs):
            pc2e = pc2
            print(pc2e)

            tree2 = cKDTree(pc2e, leafsize=500, balanced_tree=False)
            dist_in, ind = tree2.query(pc1_tr, k=1)
            # ind_all = ind
            nonempty = np.count_nonzero(dist_in)
            dist_in = np.sum(np.abs(dist_in))
            if nonempty != 0:
                dist_in = np.divide(dist_in, nonempty)
            dist_p2p = dist_in

            # __________________________________________grad
            c_grad = g.gradient(dist_p2p, last_layer)

        return c_grad

    # def train_step(self, data):
    #     with tf.GradientTape() as tape:
    #         x, y = data
    #
    #         geo_out = self.Geo_model(x['geo_input'])
    #         lidar_out = self.Lidar_model(x['AI_input'])
    #
    #         In_fusion = {'inp_fusion_geo': geo_out, 'inp_fusion_li': lidar_out}
    #         dq_out = self.fusion(In_fusion)
    #
    #         total_loss = self.compiled_loss(y, dq_out, regularization_losses=self.losses)
    #
    #     grads = tape.gradient(total_loss, self.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    #     return {"loss": total_loss}
    #
    # @tf.function
    # def call(self, data, training=False):
    #     geo_input = data['geo_input']
    #     li_input = data['AI_input']
    #
    #     geo_out = self.Geo_model(geo_input)
    #     lidar_out = self.Lidar_model(li_input)
    #
    #     In_fusion = {'inp_fusion_geo': geo_out, 'inp_fusion_li': lidar_out}
    #
    #     dq_out = self.fusion(In_fusion)
    #     return dq_out


@tf.function
def pcs_dis(pc2, pc1_tr):
    s1, s2, s3 = pc2.shape[0], pc2.shape[1], pc2.shape[2]
    pc2 = tf.cast(pc2, dtype=tf.float16)
    pc1_tr = tf.cast(pc1_tr, dtype=tf.float16)
    print(s1, s2, s3)
    i, cham_dist = tf.constant(0), tf.constant([0], shape=[s1], dtype=tf.float16)
    # i, cham_dist = tf.constant(0), tf.constant(0, dtype=tf.float32)

    while i < s2:  # AutoGraph converts while-loop to tf.while_loop().
        pc2i = pc2[:, i]
        # print(pc2i.shape)
        pc2i = tf.reshape(pc2i, shape=[-1, 1, 3])
        print("pc2i", pc2i.shape, pc2.shape)
        cham_dist_temp = tfgn.chamfer_distance.evaluate(pc2i, pc1_tr)
        cham_dist += cham_dist_temp
        i += 1
    return cham_dist


def pcs_cent_dis(source, destination):
    # Find nearest destination point for each source point
    # Set up variables
    src = tf.Variable(source)
    dst = tf.constant(destination)

    # Get distances
    src_squ = tf.reduce_sum(tf.square(source), axis=1, keepdims=True)
    dst_squ = tf.reduce_sum(tf.square(destination), axis=1, keepdims=True)
    dist = src_squ - 2 * tf.matmul(src, dst, transpose_b=True) + tf.transpose(dst_squ)
    print("dist", dist.shape)
    ind_nn = tf.argmin(dist, axis=1)

    dst_gth = tf.gather(dst, ind_nn, axis=0)
    dst_mean = tf.reduce_mean(dst_gth, axis=0, keepdims=True)
    dst_cnt = dst_gth - dst_mean

    return dst_cnt


def cal_grad(y, AI_data_temp):
    pc1_raw, pc2_raw = np.split(AI_data_temp, 2, axis=4)
    # B x T x W x H x Channels (500 4 64 720 7)
    s0, s1, s2, s3, s4 = pc1_raw.shape[0], pc1_raw.shape[1], pc1_raw.shape[2], pc1_raw.shape[3], pc1_raw.shape[4]
    # print(s0, s1, s2, s3, s4)

    pc1 = np.reshape(pc1_raw[:, -1, :, :, 0:3], [-1, s2 * s3, 3])
    pc2 = np.reshape(pc2_raw[:, -1, :, :, 0:3], [-1, s2 * s3, 3])

    # normal2 = tf.reshape(pc2_raw[:, :, :, 3:6], [s1, s2 * s3, 3])
    # normal1 = tf.reshape(pc1_raw[:, :, :, 3:6], [s1, s2 * s3, 3])
    # non_zero_order = ~np.all(pc1 == 0, axis=2)

    Rq, Tr3 = tfg.dual_quaternion.to_rotation_translation(y)
    R33 = tfg.rotation_matrix_3d.from_quaternion(Rq)
    RT = np.concatenate([R33, np.expand_dims(Tr3, axis=2)], -1)
    one_row = np.tile([0.0, 0.0, 0.0, 1.0], (s0, 1, 1))
    RT = np.concatenate((RT, one_row), axis=1)

    pc1 = np.pad(pc1, [[0, 0], [0, 0], [0, 1]], constant_values=1)
    pc1 = np.transpose(pc1, (0, 2, 1))
    pc1_tr = np.matmul(RT, pc1)
    pc1_tr = pc1_tr[:, 0:3]
    pc1_tr = np.transpose(pc1_tr, (0, 2, 1))  # B x WH x 3

    # remove zero values
    dist_p2p = np.zeros(s0, dtype=float)
    # ind_all = np.zeros([s0, s3 * s2])
    for i in range(s0):
        tree2 = cKDTree(pc2[i], leafsize=500, balanced_tree=False)
        dist_in, ind = tree2.query(pc1_tr[i], k=1)
        # ind_all[i, :] = ind
        nonempty = np.count_nonzero(dist_in)
        dist_in = np.sum(np.abs(dist_in))
        if nonempty != 0:
            dist_in = np.divide(dist_in, nonempty)
        dist_p2p[i] = dist_in

    dist_all = dist_p2p

    return dist_all


def cal_grad_one(y, AI_data_temp):
    pc1_raw, pc2_raw = np.split(AI_data_temp, 2, axis=3)
    # T x W x H x Channels (4 64 720 7)
    s1, s2, s3, s4 = pc1_raw.shape[0], pc1_raw.shape[1], pc1_raw.shape[2], pc1_raw.shape[3]
    # print(s0, s1, s2, s3, s4)

    pc1 = np.reshape(pc1_raw[-1, :, :, 0:3], [s2 * s3, 3])
    pc2 = np.reshape(pc2_raw[-1, :, :, 0:3], [s2 * s3, 3])

    # normal2 = tf.reshape(pc2_raw[:, :, :, 3:6], [s1, s2 * s3, 3])
    # normal1 = tf.reshape(pc1_raw[:, :, :, 3:6], [s1, s2 * s3, 3])
    # non_zero_order = ~np.all(pc1 == 0, axis=2)

    Rq, Tr3 = tfg.dual_quaternion.to_rotation_translation(y)
    R33 = tfg.rotation_matrix_3d.from_quaternion(Rq)
    RT = np.concatenate([R33, np.expand_dims(Tr3, axis=1)], -1)
    one_row = np.array([[0.0, 0.0, 0.0, 1.0]])
    RT = np.concatenate((RT, one_row), axis=0)

    pc1 = np.pad(pc1, [[0, 0], [0, 1]], constant_values=1)
    pc1 = np.transpose(pc1, (1, 0))
    pc1_tr = np.matmul(RT, pc1)
    pc1_tr = pc1_tr[0:3]
    pc1_tr = np.transpose(pc1_tr, (1, 0))  # WH x 3

    tree2 = cKDTree(pc2, leafsize=500, balanced_tree=False)
    dist_in, ind = tree2.query(pc1_tr, k=1)
    nonempty = np.count_nonzero(dist_in)
    dist_in = np.sum(np.abs(dist_in))
    if nonempty != 0:
        dist_in = np.divide(dist_in, nonempty)
    dist_p2p = dist_in

    dist_all = dist_p2p
    return dist_all


# ________________________________
# Loss
# ________________________________


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1_loss_my = 2 * p * r / (p + r + K.epsilon())
    f1_loss_my = tf.where(tf.math.is_nan(f1_loss_my), tf.zeros_like(f1_loss_my), f1_loss_my)
    return 1 - K.mean(f1_loss_my)
# _____________________________________________________________________________________________
# pw, px, py, pz, qw, qx, qy, qz


class DualQuaternionLoss(tf.keras.losses.Loss):
    def __init__(self, name="dual_quaternion_loss", **kwargs):
        super(DualQuaternionLoss, self).__init__(name=name, **kwargs)

    def call(self, DQ_target, DQ_predicted):
        # Extract real and dual parts
        Q0_target = DQ_target[:, :4]
        Qe_target = DQ_target[:, 4:]

        Q0_predicted = DQ_predicted[:, :4]
        Qe_predicted = DQ_predicted[:, 4:]

        # Compute the dual quaternion dot product
        dot_product = tf.reduce_sum(Q0_target * Q0_predicted + Qe_target * Qe_predicted, axis=-1) + \
                      tf.reduce_sum(Q0_target * Qe_predicted - Qe_target * Q0_predicted, axis=-1)
        
        # Normalize by the number of samples
        num_samples = tf.cast(tf.shape(DQ_target)[0], dtype=tf.float32)
        average_dot_product = tf.reduce_mean(dot_product)
        
        # The loss is the absolute value of the difference from 1
        loss = tf.abs(1.0 - average_dot_product)
        
        return loss
    

def dual_quaternion_loss(y_true, y_pred):
    # Ensure dual quaternions are normalized
    y_true_normalized = tf.math.l2_normalize(y_true, axis=-1)
    y_pred_normalized = tf.math.l2_normalize(y_pred, axis=-1)

    # Compute the dot product between true and predicted dual quaternions
    dot_product = tf.reduce_sum(y_true_normalized * y_pred_normalized, axis=-1)

    # Compute the loss as the square of the geodesic distance on the dual quaternion manifold
    loss = tf.reduce_mean((1.0 - tf.square(dot_product)))
    return loss