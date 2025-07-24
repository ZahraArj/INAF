# ___________________________________________________________________________________________________ External libraries
import numpy as np
from Mytools.pre_pc_project import LaserScan
import yaml
import pickle
from scipy.spatial import cKDTree
import _pickle as cPickle
import os
# import matplotlib.pyplot as plt
import tables
import json
import hickle as hkl
import time

# ___________________________________________________________________________________________________________ My classes
from Mytools.kitti_open_items import Loader_KITTI


class Lidar_data_prepare:

    def __init__(self, mother_folder, manual_id=None):

        # Read YAML file
        self.images = None
        with open("Mytools/config.yaml", 'r') as stream:
            cfg = yaml.safe_load(stream)

        ds_config = cfg['datasets']['kitti']
        self.scans = ds_config.get('scans')
        self.image_width = ds_config.get('image-width', 1024)
        self.image_height = ds_config.get('image-height', 64)
        self.fov_up = ds_config.get('fov-up', 3)
        self.fov_down = ds_config.get('fov-down', -25)
        self.max_depth = ds_config.get('max-depth', 80)
        self.min_depth = ds_config.get('min-depth', 2)

        self.mean_img = ds_config['mean-image']
        self.std_img = ds_config['std-image']
        self.channels = ds_config['channels']

        crop_factors = ds_config.get('crop-factors', [0, 0])
        self.crop_top = crop_factors[0]
        self.crop_left = crop_factors[1]

        self.pc_path = ds_config.get('pc_path')
        self.gt_path = ds_config.get('gt_path')
        self.seq = ds_config.get('seq')
        if manual_id is not None:
            self.loader_KT = Loader_KITTI(self.pc_path, self.gt_path, manual_id)
        else:
            self.loader_KT = Loader_KITTI(self.pc_path, self.gt_path, self.seq)

        self.mother_folder = mother_folder

        Net_config = cfg['Networks']
        self.batch_size = Net_config.get('Batch_size', 2)
        self.time_size = Net_config.get('time_size')
        self.batch_gen = Net_config.get('batch_gen', False)
        self.data_pre = Net_config.get('data_pre', 'saved')
        self.s_idx = self.time_size

        # self.seq_size = self.e_idx - self.s_idx + self.time_size - 1
        self.manual_id = manual_id

        seq_int = int(self.seq)
        # self.s_idx = ds_config.get('s_idx')
        # self.e_idx = ds_config.get('e_idx')
        self.e_idx = int(self.scans[seq_int])

    def transform_images(self):
        temp_channel = np.append(np.array(self.channels), np.array(self.channels) + 8)
        imgs_normalized = [img.transpose(2, 0, 1) for img in self.images]
        imgs_normalized = np.stack(imgs_normalized)
        imgs_normalized = imgs_normalized[:, temp_channel]
        return imgs_normalized

    def get_velo_image(self, idx):


        scan = LaserScan(H=self.image_height, W=self.image_width, fov_up=self.fov_up,
                         fov_down=self.fov_down,
                         min_depth=self.min_depth, max_depth=self.max_depth,
                         mother_folder=self.mother_folder, idx=idx)

        scan.open_scan(self.loader_KT.get_item_pc(idx)[0])
        proj_xyz = scan.proj_xyz / self.max_depth
        proj_remission = scan.proj_remission
        proj_range = scan.proj_range
        proj_normal = scan.proj_normal
        image = np.dstack((proj_xyz, proj_remission, proj_normal, proj_range))

        return image

    def load_image(self, ds_index1, ds_index2, img_index):
        img1 = self.get_velo_image(ds_index1)
        img2 = self.get_velo_image(ds_index2)
        img12 = np.concatenate((img1, img2), axis=2)
        self.images[img_index] = img12

    def load_images(self, indices):
        # print('start')
        for i in range(indices.shape[0] - 1):
            # print('ii',indices[i], indices[i + 1])
            self.load_image(indices[i], indices[i + 1], i)

    def create_lidar_data_timedist(self, batch_s=None):
        self.images = [None] * self.time_size
        # indices = np.array([i for i in range(batch_s - self.time_size + 1, batch_s + 2)])
        if batch_s == self.time_size-1:
            indices = np.array([0,0,1,2,3])
        else:
            indices = np.array([i for i in range(batch_s - self.time_size, batch_s + 1)])
            # indices = np.array([i for i in range(batch_s - self.time_size, batch_s + 1)])
        self.load_images(indices)
        proc_images = self.transform_images()
        data = proc_images

        s1, s2, s3, s4 = data.shape
        data_td = np.empty([1, self.time_size, s2, s3, s4])
        data_td[0] = data

        return data

    def create_lidar_data_gen(self, idxs):
        self.images = [None] * self.time_size

        i4_int, i3_int, i2_int, i1_int, i0_int = (
            int(idxs[0][:6]), int(idxs[1][:6]), int(idxs[2][:6]), int(idxs[3][:6]), int(idxs[4][:6]))
        indices = np.array([i for i in [i4_int, i3_int, i2_int, i1_int, i0_int]])
        # print('in',indices)
        self.load_images(indices)
        # print('im',self.images)
        proc_images = self.transform_images()
        data = proc_images

        s1, s2, s3, s4 = data.shape
        data_td = np.empty([1, self.time_size, s2, s3, s4])
        data_td[0] = data

        return data

    def create_line_by_line(self):
        filename = os.path.join('results', 'network_input_files',
                                'lidar_data_seq_' + self.seq + '.h5')

        f = tables.open_file(filename, mode='w')
        atom = tables.Float32Atom()

        array_c = f.create_earray(f.root, 'data', atom, (0, self.time_size, self.image_height, self.image_width, 14))

        for idx in range(self.s_idx-1, self.e_idx+1):
            # print("___", idx)
            AI_data_batch = self.create_lidar_data_timedist(batch_s=idx)
            AI_data_batch = np.transpose(AI_data_batch, (0, 2, 3, 1))
            AI_data_batch = np.expand_dims(AI_data_batch, axis=0)

            array_c.append(AI_data_batch)

            print('lidar scan created', idx)

        f.close()
        print('_______________________________________________________________________________________________________')
        print('Lidar Data Saved')
        print('_______________________________________________________________________________________________________')

    def load_saved_data(self):
        filename = os.path.join(self.mother_folder, 'network_input_files', 'lidar_data.pkl')
        lidar_file = open(filename, "rb")
        data_LI = pickle.load(lidar_file)
        lidar_file.close()
        # AI_data = np.transpose(data_LI, (0, 2, 3, 1))
        AI_data = np.transpose(data_LI, (0, 1, 3, 4, 2))
        # print('AI_data.shape:', AI_data.shape)
        return AI_data

    def load_saved_data_h5(self, seq, train_s, train_end):
        filename = os.path.join('results', 'network_input_files',
                                'lidar_data_seq_' + seq + '.h5')
        lidar_file = tables.open_file(filename, mode='r')
        AI_data = lidar_file.root.data[train_s: train_end]
        lidar_file.close()

        return AI_data
    
    def load_saved_data_h5_all(self, seq):
        filename = os.path.join('results', 'network_input_files',
                                'lidar_data_seq_' + seq + '.h5')
        lidar_file = tables.open_file(filename, mode='r')
        AI_data = lidar_file.root.data[:]
        lidar_file.close()

        return AI_data


'''
def visualize_(pixelvalue, img):
    proj_xyz = img[0:3]
    proj_remission = img[3]
    proj_normal = img[4:7]
    proj_range = img[7]

    plt.close('all')
    # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
    if pixelvalue == 'reflectance':
        pixel_values = proj_remission
    elif pixelvalue == 'range':
        pixel_values = proj_range
    elif pixelvalue == 'xyz':
        pixel_values = proj_xyz[1, :, :]
    elif pixelvalue == 'normals':
        # print(self.proj_normal[30])
        pixel_values = proj_normal[2, :, :]

    dpi = 500  # Image resolution
    fig, ax = plt.subplots(dpi=dpi)
    fig.set_size_inches(16, 2)
    indices = np.nonzero(pixel_values)
    u = indices[0]
    v = indices[1]
    ax.scatter(v, u, s=1, c=pixel_values[indices], linewidths=0, alpha=1, cmap='gist_ncar')
    ax.set_facecolor((0, 0, 0))  # Set regions with no points to black
    ax.axis('scaled')
    # plt.xlim([0, 720])  # prevent drawing empty space outside of horizontal FOV
    # plt.ylim([0, 64])

    plt.show()
    
'''
'''
    def create_lidar_data(self):
        self.images = [None] * self.seq_size
        indices = np.array([i for i in range(self.s_idx, self.e_idx + 1)])
        self.load_images(indices)
        proc_images = self.transform_images()

        # data = {'images': proc_images, 'untrans-images': org_images}
        data = proc_images

        filename = os.path.join(self.mother_folder, 'network_input_files', 'lidar_data.pkl')
        lidar_file = open(filename, "wb")
        pickle.dump(data, lidar_file, protocol=4)
        lidar_file.close()

        print("_______________________________________________________________________________________________________")
        print('Lidar input saved')
        print("_______________________________________________________________________________________________________")

    def create_lidar_data_timedist(self, seq=''):
        self.images = [None] * (self.e_idx - self.s_idx + self.time_size - 1)
        indices = np.array([i for i in range(self.s_idx - self.time_size + 1, self.e_idx + 1)])
        self.load_images(indices)
        print('aa')
        proc_images = self.transform_images()
        self.images = None
        print('proc_images_size', proc_images.nbytes)

        s1, s2, s3, s4 = proc_images.shape
        data_td = np.empty([self.e_idx - self.s_idx, self.time_size, s2, s3, s4])
        print('data_td_empty_size', data_td.nbytes)
        for i in range(self.e_idx - self.s_idx):
            data_td[i] = proc_images[i:i + self.time_size]

        print('data_td_size', data_td.nbytes)

        # filename = os.path.join(self.mother_folder, 'network_input_files', 'lidar_data' + seq + '.pkl')
        # filename = os.path.join(self.mother_folder, 'network_input_files', 'lidar_data' + seq + '.hkl')
        # lidar_file = open(filename, "wb")
        # pickle.dump(data_td, lidar_file, protocol=2)
        # hkl.dump(data_td, lidar_file, mode='w')
        # lidar_file.close()

        print("_______________________________________________________________________________________________________")
        print('Lidar input saved')
        print("_______________________________________________________________________________________________________")

    def create_lidar_data_timedist_tfrecord(self):
        self.images = [None] * (self.e_idx - self.s_idx + self.time_size - 1)
        indices = np.array([i for i in range(self.s_idx - self.time_size + 1, self.e_idx + 1)])
        self.load_images(indices)

        proc_images = self.transform_images()
        data = proc_images

        s1, s2, s3, s4 = data.shape
        data_td = np.empty([self.e_idx - self.s_idx, self.time_size, s2, s3, s4])
        for i in range(self.e_idx - self.s_idx):
            data_td[i] = data[i:i + self.time_size]

        AI_data = np.transpose(data_td, (0, 1, 3, 4, 2))

        return AI_data
        
        def create_trees(self):
        proc_images = self.transform_images()
        AI_data = np.transpose(proc_images, (0, 2, 3, 1))

        pc1_raw, pc2_raw = np.split(AI_data, 2, axis=1)
        s1, s2, s3, s4 = pc1_raw.shape

        s_b = self.batch_size

        for i in range(self.e_idx - self.s_idx):
            # print(i)
            # pc1 = np.reshape(pc1_raw[i:i+1, :, :, 0:3], [s2 * s3, 3])
            pc2 = np.reshape(pc2_raw[i, :, :, 0:3], [s2 * s3, 3])

            tree2 = cKDTree(pc2, leafsize=1024, balanced_tree=False)
            print("here")
            raw_tree = cPickle.dumps(tree2)
            # print(raw_tree)
'''
