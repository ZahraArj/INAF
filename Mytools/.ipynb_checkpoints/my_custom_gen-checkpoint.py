import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import yaml
import os
import shutil
from sklearn.model_selection import train_test_split

from Mytools.pre_geo_data import Geometry_data_prepare
from Mytools.make_pcfile_4network import Lidar_data_prepare


class file_order:

    def __init__(self, mother_folder_all, **kwargs):
        # Read YAML file
        self.images = None
        with open("Mytools/config.yaml", 'r') as stream:
            cfg = yaml.safe_load(stream)
        ds_config = cfg['datasets']['kitti']
        self.scans = ds_config.get('scans')
        self.pc_path = ds_config.get('pc_path')
        self.gt_path = ds_config.get('gt_path')
        self.all_images_path = ds_config.get('all_images_path')
        self.seq_input = ds_config.get('seq')
        self.sequences = ds_config.get('sequences')
        nt_config = cfg['Networks']
        self.Batch_size = nt_config.get('Batch_size')
        self.time_size = nt_config.get('time_size')
        self.data_pre_mode = nt_config.get('data_pre')

        self.train_dir = self.pc_path
        self.dest_dir = self.all_images_path
        self.counter = 0

        self.mother_folder = mother_folder_all
        self.images_dir = self.all_images_path

        self.G_create = Geometry_data_prepare(self.mother_folder)

    def forward_file_order(self):
        # print(self.mother_folder)


        filenames = []
        Geo_input = []
        gt_output = []

        filenames_counter = 0

        for seq in self.sequences:
            print('seq_queue', seq)

            pc_path_seq = os.path.join(self.pc_path, seq, 'velodyne')

            subdirs, dirs, files = os.walk(pc_path_seq).__next__()
            m = len(files)
            # print(m)

            G_create = Geometry_data_prepare(self.mother_folder)
            G_data, G_gt = G_create.load_saved_data_all(seq)
            # print('here0', G_data.shape, G_gt.shape, m)
            filenames_counter_in = 0
            for subdir, dirs, files in os.walk(pc_path_seq):
                sorted_files = sorted(files)
                # print(files)

                for files_i4, files_i3, files_i2, files_i1, files_i0 in \
                        (zip(sorted_files, sorted_files[1:], sorted_files[2:], sorted_files[3:], sorted_files[4:])):
                    # print('here1',[files_i4, files_i3, files_i2, files_i1, seq])

                    filenames.append([files_i4, files_i3, files_i2, files_i1, files_i0, seq])
                    Geo_input.append(G_data[filenames_counter_in])
                    gt_output.append(G_gt[filenames_counter_in])

                    # labels[filenames_counter, 0] = labels_counter
                    filenames_counter_in = filenames_counter_in + 1

        # saving the filename array as .npy file
        file_names_dir = os.path.join(self.all_images_path, 'file_names.npy')
        np.save(file_names_dir, filenames)

        temp = (shuffle(list(zip(filenames, Geo_input, gt_output))))
        filenames_shuffled, Geo_shuffled, gt_shuffled = zip(*temp)
        filenames_shuffled, Geo_shuffled, gt_shuffled = list(filenames_shuffled), list(Geo_shuffled), list(gt_shuffled)
        filenames_shuffled, Geo_shuffled, gt_shuffled = np.array(filenames_shuffled), np.array(Geo_shuffled), np.array(
            gt_shuffled)

        x_train_filenames, x_val_filenames, x_train_geo, x_val_geo, y_train, y_val = \
            train_test_split(filenames_shuffled, Geo_shuffled, gt_shuffled, test_size=0.2, random_state=1)

        # saving the shuffled file.
        # you can load them later using np.load().
        file_names_dir = os.path.join(self.all_images_path, 'file_names_shuffles.npy')
        file_names_dir_val = os.path.join(self.all_images_path, 'file_names_shuffles_val.npy')
        Geo_dir = os.path.join(self.all_images_path, 'Geo_shuffles.npy')
        Geo_dir_val = os.path.join(self.all_images_path, 'Geo_shuffles_val.npy')
        gt_dir = os.path.join(self.all_images_path, 'gt_shuffles.npy')
        gt_dir_val = os.path.join(self.all_images_path, 'gt_shuffles_val.npy')

        np.save(file_names_dir, x_train_filenames)
        np.save(file_names_dir_val, x_val_filenames)
        np.save(Geo_dir, x_train_geo)
        np.save(Geo_dir_val, x_val_geo)
        np.save(gt_dir, y_train)
        np.save(gt_dir_val, y_val)


class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self, inputs, output, batch_size):
        self.Geo, self.filenames = inputs[0], inputs[1]
        self.gt = output
        self.batch_size = batch_size

        self.mother_folder_all = ''.join(['results'])
        # self.mother_folder_all = os.path.join(mother_folder_all, 'supervised_all')

    def __len__(self):
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        # print(idx)
        batch_x_Li = self.filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x_Geo = self.Geo[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.gt[idx * self.batch_size: (idx + 1) * self.batch_size]
        
        # print(0, batch_x_Li)
        # print(1,batch_x_Geo )
        # print(2, batch_y )
        
        # loader2 = Lidar_data_prepare(self.mother_folder_all)
        # Lidar_data_batch = np.array([loader2.create_lidar_data_gen(filenames) for filenames in batch_x_Li])

        Lidar_data_batch = np.array([Lidar_data_prepare(self.mother_folder_all, manual_id=filenames[-1] ).create_lidar_data_gen(filenames[:-1]) for filenames in batch_x_Li])
        
        # Lidar_data_batch = np.zeros([256,4,14, 64, 720 ])
        
        # x_input = [np.array(batch_x_Geo), Lidar_data_batch]
        x_input = {'geo_input': np.array(batch_x_Geo), 'AI_input': np.transpose(Lidar_data_batch, (0, 1, 3, 4,2))}
        print("hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", np.shape(x_input['AI_input']))

        y_output = np.array(batch_y)

        return x_input, y_output
