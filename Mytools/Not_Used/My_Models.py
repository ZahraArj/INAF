import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
# import tensorflow_probability as tfp
from scipy.spatial import cKDTree
# from sklearn.neighbors import KDTree
import gc
import time
import pickle
from Mytools.att_ww import cbam_block

import tensorflow as tf
from tensorflow.keras import layers, Model, Input, backend, losses
from tensorflow.keras.models import load_model
import tensorflow_graphics.geometry.transformation as tfg

from Mytools.make_pcfile_4network import Lidar_data_prepare
from Mytools.pre_geo_data import Geometry_data_prepare
from Mytools.tfrecord_tfread import recorder_reader
from Mytools.output_save import save2txt
from Mytools.att_lstm import attention

# tf.compat.v1.disable_eager_execution()

# tf.config.run_functions_eagerly(True)

# physical_devices = tf.config.list_logical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)

# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))


class BaseNet:

    def __init__(self, mother_folder, manual_id=None):
        # Read YAML file
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
        self.scans = ds_config.get('scans')

        Net_config = cfg['Networks']
        self.Batch_size = Net_config.get('Batch_size', 2)
        self.Epochs = Net_config.get('Epochs', 2)
        self.Save_path = Net_config.get('Save_path', './saved_model/model.h5')
        self.method = Net_config.get('method')
        self.branch = Net_config.get('branch')
        self.loss_weights = Net_config.get('loss_weights')
        self.time_size = Net_config.get('time_size')
        self.batch_gen = Net_config.get('batch_gen', False)
        self.data_pre = Net_config.get('data_pre', 'saved')
        self.fusion = Net_config.get('fusion', 'simple')
        self.rot_rep = Net_config.get('rot_rep', 'expn')
        self.divided_train = Net_config.get('divided_train', 200)

        self.count = 0
        self.mother_folder = mother_folder

        self.manual_id = manual_id

        # print('1:not none:', manual_id)
        self.loader_pre = Lidar_data_prepare(self, manual_id=manual_id)

        self.li_create = Lidar_data_prepare(self.mother_folder)
        self.G_create = Geometry_data_prepare(self.mother_folder)

        self.recorder_reader = recorder_reader(self.mother_folder)

    # __________________________________________________________________________________________________________________
    # Resnet
    # __________________________________________________________________________________________________________________
    def relu_bn(self, inputs):
        relu = layers.TimeDistributed(layers.ReLU())(inputs)
        bn = layers.TimeDistributed(layers.BatchNormalization())(relu)
        return bn

    def Conv_Layer(self, filters):
        Conv_Layer = layers.Conv2D(filters=filters,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding='valid',
                                   activation='relu')
        return Conv_Layer

    def residual_block(self, in_x, downsample, filters, kernel_size, strides=1):
        y = layers.TimeDistributed(layers.Conv2D(kernel_size=kernel_size,
                                                 strides=(1 if not downsample else 2),
                                                 # strides=strides,
                                                 filters=filters,
                                                 padding='same'))(in_x)
        y = self.relu_bn(y)
        y = layers.TimeDistributed(layers.Conv2D(kernel_size=kernel_size,
                                                 strides=1,
                                                 filters=filters,
                                                 padding='same'))(y)

        if downsample:
            in_x = layers.TimeDistributed(layers.Conv2D(kernel_size=1,
                                                        strides=2,
                                                        filters=filters,
                                                        padding='same'))(in_x)

        # y = cbam_block(y)
        out = layers.Add()([in_x, y])
        out = self.relu_bn(out)
        return out

    def create_res_net(self, in_x, num_filters=64):

        t = layers.TimeDistributed(layers.BatchNormalization())(in_x)

        paddings = tf.constant([[0, 0], [0, 0], [0, 0], [1, 1], [0, 0]])
        t = tf.pad(t, paddings, mode='REFLECT')
        paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]])
        t = tf.pad(t, paddings, mode='CONSTANT')
        t = layers.TimeDistributed(layers.Conv2D(kernel_size=3,
                                                 strides=(1, 2),
                                                 filters=num_filters,
                                                 padding='valid'))(t)

        paddings = tf.constant([[0, 0], [0, 0], [0, 0], [1, 1], [0, 0]])
        t = tf.pad(t, paddings, mode='REFLECT')
        paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]])
        t = tf.pad(t, paddings, mode='CONSTANT')
        t = layers.TimeDistributed(layers.MaxPool2D(pool_size=3,
                                                    strides=(1, 2),
                                                    padding='valid'))(t)

        t = self.relu_bn(t)

        num_blocks_list = [2, 2, 2, 2]
        # num_blocks_list = [2, 3, 2, 1]
        # num_blocks_list = [2, 5, 5, 2]
        # for i in range(len(num_blocks_list)):
        #     num_blocks = num_blocks_list[i]
        #     for j in range(num_blocks):
        #         t = self.residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters, kernel_size=3)
        #     num_filters *= 2

        # _____________________________________________________________________________________________________8 Res Net
        # print('t:', t.shape)
        t1 = self.residual_block(t, downsample=False, filters=64, kernel_size=3)
        t1 = self.residual_block(t1, downsample=False, filters=64, kernel_size=3)
        # print('t1:', t1.shape)

        t2 = self.residual_block(t1, downsample=True, filters=128, kernel_size=3, strides=(1, 2))
        t2 = self.residual_block(t2, downsample=False, filters=128, kernel_size=3)
        # print('t2:', t2.shape)

        t3 = self.residual_block(t2, downsample=True, filters=256, kernel_size=3, strides=(1, 2))
        t3 = self.residual_block(t3, downsample=False, filters=256, kernel_size=3)
        # print('t3:', t3.shape)

        t4 = self.residual_block(t3, downsample=True, filters=512, kernel_size=3, strides=(2, 2))
        t4 = self.residual_block(t4, downsample=False, filters=512, kernel_size=3)
        # print('t4:', t4.shape)

        t = layers.TimeDistributed(layers.AveragePooling2D(strides=(t4.shape[2], t4.shape[3]),
                                                           pool_size=(t4.shape[2], t4.shape[3])))(t4)

        t = layers.TimeDistributed(layers.Flatten())(t)
        outputs = layers.TimeDistributed(layers.Dense(1000, activation='softmax'))(t)

        # features = [t1, t2, t3, t4, outputs]
        return outputs

    def Geo_branch(self, geo_inp):
        Fully_Connected1 = layers.TimeDistributed(layers.Dense(128, activation='tanh'))(geo_inp)
        Fully_Connected2 = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(Fully_Connected1)

        # Fully_Connected1 = layers.LSTM(128,
        #                                activation="tanh",
        #                                recurrent_activation="sigmoid",
        #                                return_sequences=True,
        #                                unroll=False,
        #                                use_bias=True,
        #                                )(geo_inp)
        #
        # Fully_Connected2 = layers.LSTM(128,
        #                                activation="tanh",
        #                                recurrent_activation="sigmoid",
        #                                return_sequences=True,
        #                                unroll=False,
        #                                use_bias=True,
        #                                )(Fully_Connected1)
        return Fully_Connected2

    def Lidar_branch(self, AI_input):
        Layer_Resnet = self.create_res_net(AI_input)
        lidar_branch = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(Layer_Resnet)
        # lidar_branch = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(lidar_branch)
        return lidar_branch

    def simple_fusion(self, geo_branch, lidar_branch):

        # concatted = layers.Concatenate(axis=2)([geo_branch, lidar_branch])
        # Fully_Connected = layers.TimeDistributed(layers.Dense(100, activation='tanh'))(concatted)

        Fully_Connected = layers.Concatenate(axis=2)([geo_branch, lidar_branch])

        Fully_Connected_T = layers.LSTM(3,
                                        activation="tanh",
                                        recurrent_activation="sigmoid",
                                        # recurrent_dropout=0.2,
                                        unroll=False,
                                        use_bias=True,
                                        name='Translation'
                                        )(Fully_Connected)
        Fully_Connected_Q = layers.LSTM(4,
                                        activation="tanh",
                                        recurrent_activation="sigmoid",
                                        unroll=False,
                                        use_bias=True,
                                        name='Quaternion'
                                        )(Fully_Connected)
        # Fully_Connected_T = layers.TimeDistributed(layers.Dense(3,
        #                                                         activation='elu',
        #                                                         name='Translation'))(Fully_Connected4)
        # Fully_Connected_Q = layers.TimeDistributed(layers.Dense(4,
        #                                                         activation='elu',
        #                                                         name='Quaternion'))(Fully_Connected4)

        return Fully_Connected_T, Fully_Connected_Q

    def simple_fusion_exp(self, geo_branch, lidar_branch):

        Fully_Connected = layers.Concatenate(axis=2)([geo_branch, lidar_branch])

        Fully_Connected_all = layers.LSTM(6,
                                          activation="tanh",
                                          recurrent_activation="sigmoid",
                                          # recurrent_dropout=0.2,
                                          unroll=False,
                                          use_bias=True,
                                          name='exp_map'
                                          )(Fully_Connected)
        return Fully_Connected_all

    def simple_fusion_dq(self, geo_branch, lidar_branch):

        Fully_Connected = layers.Concatenate(axis=2)([geo_branch, lidar_branch])

        output = layers.LSTM(8,
                             activation="tanh",
                             recurrent_activation="sigmoid",
                             # recurrent_dropout=0.2,
                             unroll=False,
                             use_bias=True,
                             name='exp_map'
                             )(Fully_Connected)

        # output = layers.Bidirectional(8,
        #                               activation="tanh",
        #                               recurrent_activation="sigmoid",
        #                               # recurrent_dropout=0.2,
        #                               unroll=False,
        #                               use_bias=True,
        #                               name='exp_map'
        #                               )(Fully_Connected)

        # print('before lstm', Fully_Connected.shape)
        # output = layers.Bidirectional(layers.LSTM(8,
        #                                           activation="tanh",
        #                                           recurrent_activation="sigmoid",
        #                                           return_sequences=True,
        #                                           # recurrent_dropout=0.2,
        #                                           unroll=False,
        #                                           use_bias=True,
        #                                           name='Transformation'
        #                                           ))(Fully_Connected)
        # print('before att', output.shape)
        # output = attention(output)
        return output

    def geo_extension(self, geo_branch):
        Fully_Connected = layers.TimeDistributed(layers.Dense(100, activation='tanh'))(geo_branch)
        geo_ext = layers.LSTM(6,
                              activation="tanh",
                              recurrent_activation="sigmoid",
                              unroll=False,
                              use_bias=True,
                              name='Translation'
                              )(Fully_Connected)
        return geo_ext

    @tf.function
    def geo_extension_grad(self, geo_branch):
        # with tf.GradientTape() as tape_g:
        #     tape_g.watch(geo_branch)
        Fully_Connected = layers.TimeDistributed(layers.Dense(100, activation='tanh'))(geo_branch)
        geo_ext = layers.LSTM(6,
                              activation="tanh",
                              recurrent_activation="sigmoid",
                              unroll=False,
                              use_bias=True,
                              name='Translation'
                              )(Fully_Connected)

        # g_grad = tape_g.gradient(geo_ext, geo_branch)
        g_grad = tf.gradients(geo_ext, geo_branch)
        return g_grad

    def Li_extension(self, Li_branch):
        with tf.GradientTape() as tape_L:
            tape_L.watch(Li_branch)
            # Fully_Connected = layers.TimeDistributed(layers.Dense(100, activation='tanh'))(Li_branch)
            Li_ext = layers.LSTM(6,
                                 activation="tanh",
                                 recurrent_activation="sigmoid",
                                 unroll=False,
                                 use_bias=True,
                                 name='Translation'
                                 )(Li_branch)

            L_grad = tape_L.gradient(Li_ext, Li_branch)
            # print('li', c_grad)
        return Li_ext, L_grad

    def soft_fusion(self, geo_branch, lidar_branch):

        conc1 = layers.Concatenate(axis=2)([geo_branch, lidar_branch])

        SS1 = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc1)
        S1 = tf.keras.activations.sigmoid(SS1)

        SS2 = layers.TimeDistributed(layers.Dense(100, activation='tanh'))(conc1)
        S2 = tf.keras.activations.sigmoid(SS2)

        AS1 = tf.math.multiply(geo_branch, S1)
        AS2 = tf.math.multiply(lidar_branch, S2)
        conc2 = layers.Concatenate(axis=2)([AS1, AS2])

        Fully_Connected = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc2)

        Fully_Connected_T = layers.LSTM(3,
                                        activation="tanh",
                                        recurrent_activation="sigmoid",
                                        # recurrent_dropout=0.2,
                                        unroll=False,
                                        use_bias=True,
                                        name='Translation'
                                        )(Fully_Connected)
        Fully_Connected_Q = layers.LSTM(4,
                                        activation="tanh",
                                        recurrent_activation="sigmoid",
                                        unroll=False,
                                        use_bias=True,
                                        name='Quaternion'
                                        )(Fully_Connected)

        return Fully_Connected_T, Fully_Connected_Q

    def soft_fusion_exp(self, geo_branch, lidar_branch):

        conc1 = layers.Concatenate(axis=2)([geo_branch, lidar_branch])

        SS1 = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc1)
        S1 = tf.keras.activations.sigmoid(SS1)

        SS2 = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc1)
        S2 = tf.keras.activations.sigmoid(SS2)

        AS1 = tf.math.multiply(geo_branch, S1)
        AS2 = tf.math.multiply(lidar_branch, S2)
        conc2 = layers.Concatenate(axis=2)([AS1, AS2])

        Fully_Connected = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc2)

        output = layers.LSTM(6,
                             activation="tanh",
                             recurrent_activation="sigmoid",
                             # recurrent_dropout=0.2,
                             unroll=False,
                             use_bias=True,
                             name='Translation'
                             )(Fully_Connected)

        return output

    def soft_fusion_dq(self, geo_branch, lidar_branch):

        conc1 = layers.Concatenate(axis=2)([geo_branch, lidar_branch])

        SS1 = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc1)
        S1 = tf.keras.activations.sigmoid(SS1)

        SS2 = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc1)
        S2 = tf.keras.activations.sigmoid(SS2)

        AS1 = tf.math.multiply(geo_branch, S1)
        AS2 = tf.math.multiply(lidar_branch, S2)
        conc2 = layers.Concatenate(axis=2)([AS1, AS2])

        Fully_Connected = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc2)

        output = layers.LSTM(8,
                             activation="tanh",
                             recurrent_activation="sigmoid",
                             # recurrent_dropout=0.2,
                             unroll=False,
                             use_bias=True,
                             name='Translation'
                             )(Fully_Connected)
        # print('before lstm', Fully_Connected.shape)
        # output = layers.Bidirectional(layers.LSTM(8,
        #                                           activation="tanh",
        #                                           recurrent_activation="sigmoid",
        #                                           return_sequences=True,
        #                                           # recurrent_dropout=0.2,
        #                                           unroll=False,
        #                                           use_bias=True,
        #                                           name='Transformation'
        #                                           ))(Fully_Connected)
        # print('before att', output.shape)
        # output = attention(output)

        return output

    def grad_fusion(self, geo_branch, lidar_branch, g_grad, L_grad):
        print(geo_branch.dtype)
        print(g_grad.dtype)

        geo_fc = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(g_grad)
        Li_fc = layers.TimeDistributed(layers.Dense(100, activation='tanh'))(L_grad)

        conc1 = layers.Concatenate(axis=2)([geo_fc, Li_fc])

        SS1 = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc1)
        S1 = tf.keras.activations.sigmoid(SS1)

        SS2 = layers.TimeDistributed(layers.Dense(100, activation='tanh'))(conc1)
        S2 = tf.keras.activations.sigmoid(SS2)

        AS1 = tf.math.multiply(geo_branch, S1)
        AS2 = tf.math.multiply(lidar_branch, S2)
        conc2 = layers.Concatenate(axis=2)([AS1, AS2])

        Fully_Connected = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc2)

        output = layers.LSTM(6,
                             activation="tanh",
                             recurrent_activation="sigmoid",
                             # recurrent_dropout=0.2,
                             unroll=False,
                             use_bias=True,
                             name='Translation'
                             )(Fully_Connected)

        return output

    def grad2_fusion(self, geo_branch, lidar_branch, inputs):
        # conc1 = layers.Concatenate(axis=2)([geo_branch, lidar_branch])

        Fully_Connected_geo = layers.Dense(64, activation='tanh')(geo_branch)
        out_geo = layers.LSTM(8,
                              activation='tanh',
                              recurrent_activation="sigmoid",
                              # recurrent_dropout=0.2,
                              unroll=False,
                              use_bias=True,
                              name='Translation'
                              )(Fully_Connected_geo)
        # print(inputs.shape)
        # print(out_geo.shape)
        S1 = self.var_layer(inputs, out_geo)

        Fully_Connected_li = layers.Dense(64, activation='tanh')(lidar_branch)
        out_li = layers.LSTM(8,
                             activation='tanh',
                             recurrent_activation="sigmoid",
                             # recurrent_dropout=0.2,
                             unroll=False,
                             use_bias=True,
                             name='Translation'
                             )(Fully_Connected_li)
        S2 = self.var_layer(inputs, out_li)

        AS1 = tf.math.multiply(geo_branch, S1)
        AS2 = tf.math.multiply(lidar_branch, S2)

        conc2 = layers.Concatenate(axis=2)([AS1, AS2])
        Fully_Connected = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc2)

        output = layers.LSTM(8,
                             activation="tanh",
                             recurrent_activation="sigmoid",
                             # recurrent_dropout=0.2,
                             unroll=False,
                             use_bias=True,
                             name='Translation'
                             )(Fully_Connected)
        return output

    # def ent_fusion(self, geo_branch, lidar_branch):
    #     Fully_Connected_geo = layers.Dense(64, activation='tanh')(geo_branch)
    #     out_geo = layers.LSTM(8,
    #                           activation='tanh',
    #                           recurrent_activation="sigmoid",
    #                           # recurrent_dropout=0.2,
    #                           unroll=False,
    #                           use_bias=True,
    #                           name='Translation'
    #                           )(Fully_Connected_geo)
    #     S1 =
    #
    #     Fully_Connected_li = layers.Dense(64, activation='tanh')(lidar_branch)
    #     out_li = layers.LSTM(8,
    #                          activation='tanh',
    #                          recurrent_activation="sigmoid",
    #                          # recurrent_dropout=0.2,
    #                          unroll=False,
    #                          use_bias=True,
    #                          name='Translation'
    #                          )(Fully_Connected_li)
    #     S2 = self.var_layer(inputs, out_li)
    #
    #     AS1 = tf.math.multiply(geo_branch, S1)
    #     AS2 = tf.math.multiply(lidar_branch, S2)
    #
    #     conc2 = layers.Concatenate(axis=2)([AS1, AS2])
    #     Fully_Connected = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc2)

    # def shannon_entropy_func(self, p):
    #     """Calculates the shannon entropy.
    #     Arguments:      p (int)        : probability of event.
    #     Returns:        shannon entropy.
    #     """
    #
    #     return -tf.math.log(p.mean())
    #
    #     # Create a Bernoulli distribution
    #     bernoulli_distribution = tfd.Bernoulli(probs=.5)
    #
    #     # Use TFPs entropy method to calculate the entropy of the distribution
    #     shannon_entropy = bernoulli_distribution.entropy()

    def no_fuse(self, lstm_AI2):

        Fully_Connected4 = layers.Dense(100,
                                        activation='tanh')(lstm_AI2)

        Fully_Connected_T = layers.LSTM(3,
                                        activation='tanh',
                                        recurrent_activation="sigmoid",
                                        # recurrent_dropout=0.2,
                                        unroll=False,
                                        use_bias=True,
                                        name='Translation'
                                        )(Fully_Connected4)
        Fully_Connected_Q = layers.LSTM(4,
                                        activation='tanh',
                                        recurrent_activation="sigmoid",
                                        unroll=False,
                                        use_bias=True,
                                        name='Quaternion'
                                        )(Fully_Connected4)

        return Fully_Connected_T, Fully_Connected_Q

    def no_fuse_exp(self, lstm_AI2):

        Fully_Connected4 = layers.Dense(64,
                                        activation='tanh')(lstm_AI2)

        Fully_Connected_T = layers.LSTM(6,
                                        activation='tanh',
                                        recurrent_activation="sigmoid",
                                        # recurrent_dropout=0.2,
                                        unroll=False,
                                        use_bias=True,
                                        name='Translation'
                                        )(Fully_Connected4)

        return Fully_Connected_T

    def no_fuse_dq(self, lstm_AI2):

        Fully_Connected4 = layers.Dense(64,
                                        activation='tanh')(lstm_AI2)

        Fully_Connected_T = layers.LSTM(8,
                                        activation='tanh',
                                        recurrent_activation="sigmoid",
                                        # recurrent_dropout=0.2,
                                        unroll=False,
                                        use_bias=True,
                                        name='Translation'
                                        )(Fully_Connected4)

        return Fully_Connected_T

    @tf.custom_gradient
    def li_loss6(self, inputs, output):  # output: x y z i j k w
        pc1_raw, pc2_raw = tf.split(inputs, num_or_size_splits=2, axis=4)
        # B x T x W x H x Channels
        s0, s1, s2, s3, s4 = pc1_raw.shape[0], pc1_raw.shape[1], pc1_raw.shape[2], pc1_raw.shape[3], pc1_raw.shape[4]
        print(s0, s1, s2, s3, s4)

        pc1 = tf.reshape(pc1_raw[:, -1, :, :, 0:3], shape=[-1, s2 * s3, 3])
        pc2 = tf.reshape(pc2_raw[:, -1, :, :, 0:3], shape=[-1, s2 * s3, 3])

        # normal2 = tf.reshape(pc2_raw[:, :, :, 3:6], [s1, s2 * s3, 3])
        # normal1 = tf.reshape(pc1_raw[:, :, :, 3:6], [s1, s2 * s3, 3])

        # non_zero_order = ~np.all(pc1 == 0, axis=2)

        Rq, Tr3 = tfg.dual_quaternion.to_rotation_translation(output)
        R33 = tfg.rotation_matrix_3d.from_quaternion(Rq)
        RT = tf.concat([R33, tf.expand_dims(Tr3, axis=2)], -1)
        RT = tf.pad(RT, [[0, 0], [0, 1], [0, 0]], constant_values=[0.0, 0.0, 0.0, 1.0])

        pc1 = tf.pad(pc1, [[0, 0], [0, 0], [0, 1]], constant_values=1)
        pc1 = tf.transpose(pc1, perm=[0, 2, 1])
        pc1_tr = tf.linalg.matmul(RT, pc1)
        pc1_tr = pc1_tr[:, 0:3]
        pc1_tr = tf.transpose(pc1_tr, perm=[0, 2, 1])  # B x WH x 3

        # remove zero values
        dist_p2p = np.zeros(self.Batch_size, dtype=float)
        ind_all = np.zeros([s1, self.image_height * self.image_width])
        for i in range(self.Batch_size):
            tree2 = cKDTree(pc2[i], leafsize=500, balanced_tree=False)
            dist_in, ind = tree2.query(pc1_tr[i], k=1)
            ind_all[i, :] = ind
            nonempty = np.count_nonzero(dist_in)
            dist_in = np.sum(np.abs(dist_in))
            if nonempty != 0:
                dist_in = np.divide(dist_in, nonempty)
            dist_p2p[i] = dist_in

        # ind_all = ind_all.astype(int)
        pc2_g = tf.gather(pc2, ind_all, batch_dims=1)
        # normal2_g = tf.gather(normal2, ind_all, batch_dims=1)

        # __________________________________________________________point 2 plane distance
        # dist_p2pl = tf.norm(tf.math.multiply(tf.subtract(pc1_tr, pc2_g), normal2_g), axis=2)
        # nonempty = tf.math.count_nonzero(dist_p2pl, axis=1)
        # dist_p2pl = tf.math.reduce_sum(dist_p2pl, axis=1)
        # dist_p2pl = tf.math.divide_no_nan(tf.cast(dist_p2pl, tf.float64), tf.cast(nonempty, tf.float64))
        # # __________________________________________________________plane 2 plane distance
        # normal1_trnsp = tf.transpose(normal1, perm=[0, 2, 1])
        # normal1_tr = tf.linalg.matmul(R33, normal1_trnsp)
        # normal1_tr = tf.transpose(normal1_tr, perm=[0, 2, 1])
        #
        # N1check = tf.norm(normal1, axis=2)
        # N2check = tf.norm(normal2_g, axis=2)
        # N1num = tf.not_equal(N1check, 0)
        # # print(N1num.shape)
        # N2num = tf.not_equal(N2check, 0)
        # N1N2 = tf.math.logical_and(N1num, N2num)
        # mask = tf.expand_dims(tf.cast(N1N2, dtype=tf.float32), axis=len(N1N2.shape))
        #
        # dist_pl2pl = tf.norm(tf.subtract(mask * normal1_tr, mask * normal2_g), axis=2)
        # nonempty = tf.math.count_nonzero(dist_pl2pl, axis=1)
        # dist_pl2pl = tf.math.reduce_sum(dist_pl2pl, axis=1)
        # dist_pl2pl = tf.math.divide_no_nan(tf.cast(dist_pl2pl, tf.float64), tf.cast(nonempty, tf.float64))
        # _____________________________________________________________________________all
        # dist_all = tf.add(tf.add(dist_p2p, dist_p2pl), dist_pl2pl)
        # dist_all = tf.add(dist_p2p, dist_p2pl)
        dist_all = dist_p2p

        # print('_____')
        # print(dist_p2p)
        # print(dist_p2pl)
        # print(dist_pl2pl)
        # print(dist_all)

        # ________________________________________________________________________________
        # @tf.function
        def grad(*upstream):
            with tf.GradientTape() as g:
                g.watch(output)

                # R33_c = tfg.rotation_matrix_3d.from_quaternion(
                #     tf.gather(output, [3, 4, 5, 6], axis=1))

                Rq_c, Tr3_c = tfg.dual_quaternion.to_rotation_translation(output)
                R33_c = tfg.rotation_matrix_3d.from_quaternion(Rq_c)
                RT_c = tf.concat([R33_c, tf.expand_dims(Tr3_c, axis=2)], -1)
                RT_c = tf.pad(RT_c, [[0, 0], [0, 1], [0, 0]], constant_values=[0.0, 0.0, 0.0, 1.0])

                pc1_tr_c = tf.linalg.matmul(RT_c, pc1)
                pc1_tr_c = pc1_tr_c[:, 0:3]
                pc1_tr_c = tf.transpose(pc1_tr_c, perm=[0, 2, 1])

                # __________________________________________p2p
                d_p2p = tf.norm(tf.abs(tf.subtract(pc1_tr_c, pc2_g)), axis=2)
                nonempty_g = tf.math.count_nonzero(d_p2p, axis=1)
                d_p2p = tf.norm(d_p2p, axis=1, ord=1)
                d_p2p = tf.math.divide_no_nan(tf.cast(d_p2p, tf.float64), tf.cast(nonempty_g, tf.float64))

                # __________________________________________p2pl
                # d_p2pl = tf.norm(tf.math.multiply(tf.subtract(pc1_tr_c, pc2_g), normal2_g), axis=2)
                # nonempty_g = tf.math.count_nonzero(d_p2pl, axis=1)
                # d_p2pl = tf.norm(d_p2pl, axis=1, ord=1)
                # d_p2pl = tf.math.divide_no_nan(tf.cast(d_p2pl, tf.float64), tf.cast(nonempty_g, tf.float64))
                #
                # # _________________________________________pl2pl
                # N_tp = tf.transpose(normal1, perm=[0, 2, 1])
                # N_tr = tf.linalg.matmul(R33_c, N_tp)
                # N_tr = tf.transpose(N_tr, perm=[0, 2, 1])
                # d_pl2pl = tf.norm(tf.subtract(mask * N_tr, mask * normal2_g), axis=2)
                # nonempty_g = tf.math.count_nonzero(d_pl2pl, axis=1)
                # d_pl2pl = tf.norm(d_pl2pl, axis=1, ord=1)
                # d_pl2pl = tf.math.divide_no_nan(tf.cast(d_pl2pl, tf.float64), tf.cast(nonempty_g, tf.float64))
                # __________________________________________all
                # dist_all_c = tf.add(tf.add(d_p2p, d_p2pl), d_pl2pl)
                # dist_all_c = tf.add(d_p2p, d_p2pl)
                dist_all_c = d_p2p
                # print('_____')
                # print(d_p2p)
                # print(d_p2pl)
                # print(d_pl2pl)
                # print(dist_all_c)

                # __________________________________________grad
                c_grad = g.gradient(dist_all_c, output)
                c_grad = tf.convert_to_tensor(c_grad, dtype=tf.dtypes.float32)
                upstream = tf.convert_to_tensor(upstream, dtype=tf.dtypes.float32)
                upstream = tf.reshape(upstream, [s1, 1])

            all_grad = upstream * c_grad
            all_grad = tf.reshape(all_grad, [s1, 7])

            return None, all_grad

        dist_p2p_tf = tf.reshape(dist_all, [s0, 1])
        # print(dist_in_all_tf)
        return dist_p2p_tf, grad

    @tf.function
    def var_layer(self, inputs, output):  # output: x y z i j k w

        inputs_v = tf.Variable(inputs)
        # inputs_v = inputs.numpy()
        # inputs_c = tf.constant(inputs)
        # inputs_np = inputs_c.numpy()
        pc1_raw, pc2_raw = tf.split(inputs_v, num_or_size_splits=2, axis=4)
        # B x T x W x H x Channels
        s0, s1, s2, s3, s4 = pc1_raw.shape[0], pc1_raw.shape[1], pc1_raw.shape[2], pc1_raw.shape[3], pc1_raw.shape[4]

        pc1 = tf.reshape(pc1_raw[:, -1, :, :, 0:3], shape=[-1, s2 * s3, 3])
        pc2 = tf.reshape(pc2_raw[:, -1, :, :, 0:3], shape=[-1, s2 * s3, 3])

        # normal2 = tf.reshape(pc2_raw[:, -1, :, :, 3:6], [-1, s2 * s3, 3])
        # normal1 = tf.reshape(pc1_raw[:, -1, :, :, 3:6], [-1, s2 * s3, 3])

        Rq, Tr3 = tfg.dual_quaternion.to_rotation_translation(output)
        R33 = tfg.rotation_matrix_3d.from_quaternion(Rq)
        RT = tf.concat([R33, tf.expand_dims(Tr3, axis=2)], -1)
        RT = tf.pad(RT, [[0, 0], [0, 1], [0, 0]], constant_values=[0.0, 0.0, 0.0, 1.0])

        pc1 = tf.pad(pc1, [[0, 0], [0, 0], [0, 1]], constant_values=1)
        pc1 = tf.transpose(pc1, perm=[0, 2, 1])
        pc1_tr = tf.linalg.matmul(RT, pc1)
        pc1_tr = pc1_tr[:, 0:3]
        pc1_tr = tf.transpose(pc1_tr, perm=[0, 2, 1])  # B x WH x 3

        # remove zero values
        # dist_p2p = np.zeros([s1], dtype=float)
        # ind_all = np.zeros([self.image_height * self.image_width])
        print(pc1_tr.shape, pc2.shape)

        # for epoch in range(self.Epochs):
        pc2e = pc2
        print(pc2e)

        tree2 = cKDTree(pc2e, leafsize=500, balanced_tree=False)
        # tree2 = KDTree(pc2e, leafsize=500)
        dist_in, ind = tree2.query(pc1_tr, k=1)
        # ind_all = ind
        nonempty = np.count_nonzero(dist_in)
        dist_in = np.sum(np.abs(dist_in))
        if nonempty != 0:
            dist_in = np.divide(dist_in, nonempty)
        dist_p2p = dist_in

        # ind_all = ind_all.astype(int)
        # pc2_g = tf.gather(pc2, ind_all, batch_dims=1)
        # normal2_g = tf.gather(normal2, ind_all, batch_dims=1)

        # # __________________________________________________________point 2 plane distance
        # dist_p2pl = tf.norm(tf.math.multiply(tf.subtract(pc1_tr, pc2_g), normal2_g), axis=2)
        # nonempty = tf.math.count_nonzero(dist_p2pl, axis=1)
        # dist_p2pl = tf.math.reduce_sum(dist_p2pl, axis=1)
        # dist_p2pl = tf.math.divide_no_nan(tf.cast(dist_p2pl, tf.float64), tf.cast(nonempty, tf.float64))
        # # __________________________________________________________plane 2 plane distance
        # normal1_trnsp = tf.transpose(normal1, perm=[0, 2, 1])
        # normal1_tr = tf.linalg.matmul(R33, normal1_trnsp)
        # normal1_tr = tf.transpose(normal1_tr, perm=[0, 2, 1])
        #
        # N1check = tf.norm(normal1, axis=2)
        # N2check = tf.norm(normal2_g, axis=2)
        # N1num = tf.not_equal(N1check, 0)
        # # print(N1num.shape)
        # N2num = tf.not_equal(N2check, 0)
        # N1N2 = tf.math.logical_and(N1num, N2num)
        # mask = tf.expand_dims(tf.cast(N1N2, dtype=tf.float32), axis=len(N1N2.shape))
        #
        # dist_pl2pl = tf.norm(tf.subtract(mask * normal1_tr, mask * normal2_g), axis=2)
        # nonempty = tf.math.count_nonzero(dist_pl2pl, axis=1)
        # dist_pl2pl = tf.math.reduce_sum(dist_pl2pl, axis=1)
        # dist_pl2pl = tf.math.divide_no_nan(tf.cast(dist_pl2pl, tf.float64), tf.cast(nonempty, tf.float64))
        # _____________________________________________________________________________all
        # dist_all = tf.add(tf.add(dist_p2p, dist_p2pl), dist_pl2pl)
        # dist_all = tf.add(dist_p2p, dist_p2pl)

        print('_____')
        # print(dist_p2p)
        # print(dist_p2pl)
        # print(dist_pl2pl)
        print(dist_p2p)

        # dist_p2p_tf = tf.reshape(dist_all, [s1, 1])
        return dist_p2p

    def makemodel(self, G_data=None, G_gt=None, AI_data=None):
        # global x
        #  _____________________________________________________________________________________________________________
        # Geo Branch
        #  _____________________________________________________________________________________________________________
        # inp_geo = Input(shape=7, name='geo_input')
        if self.rot_rep == 'expn':
            inp_geo = layers.Input(shape=(self.time_size, 6), name='geo_input')

        elif self.rot_rep == 'dquart':
            inp_geo = layers.Input(shape=(self.time_size, 8), name='geo_input')

        else:
            inp_geo = layers.Input(shape=(self.time_size, 7), name='geo_input')

        Geo_branch = self.Geo_branch(inp_geo)
        #  _____________________________________________________________________________________________________________
        # Lidar Branch
        #  _____________________________________________________________________________________________________________
        if not self.branch == 'geo':
            inp_lidar_stacked = layers.Input(
                shape=(self.time_size, self.image_height, self.image_width, 2 * self.channels_N),
                name='AI_input')
            Lidar_branch = self.Lidar_branch(inp_lidar_stacked)

        #  _____________________________________________________________________________________________________________
        # Extension Module
        #  _____________________________________________________________________________________________________________
        if self.fusion == 'grad':
            # Geo_branch_v = tf.Variable(Geo_branch)
            geo_ext = self.geo_extension(Geo_branch)
            g_grad = self.geo_extension_grad(Geo_branch)
            Li_ext, L_grad = self.Li_extension(Lidar_branch)

        #  _____________________________________________________________________________________________________________
        # Fusion
        #  _____________________________________________________________________________________________________________
        if self.rot_rep == 'expn':
            if self.fusion == 'simple':
                fusion_method = self.simple_fusion_exp
            elif self.fusion == 'soft':
                fusion_method = self.soft_fusion_exp
            elif self.fusion == 'grad':
                fusion_method = self.grad_fusion
        elif self.rot_rep == 'dquart':
            if self.fusion == 'simple':
                fusion_method = self.simple_fusion_dq
            elif self.fusion == 'soft':
                fusion_method = self.soft_fusion_dq
            elif self.fusion == 'grad':
                fusion_method = self.grad_fusion
            elif self.fusion == 'grad2':
                fusion_method = self.grad2_fusion
        else:
            if self.fusion == 'simple':
                fusion_method = self.simple_fusion
            elif self.fusion == 'soft':
                fusion_method = self.soft_fusion

        # elif self.fusion == 'prob': fusion_method = self.simple_fusion

        #  _____________________________________________________________________________________________________________
        # Build Model
        #  _____________________________________________________________________________________________________________
        if self.branch == 'all':
            if self.rot_rep == 'expn':
                if self.fusion == 'grad':
                    Fully_Connected_all = fusion_method(Geo_branch, Lidar_branch, g_grad, L_grad)
                    model = Model(inputs=[inp_lidar_stacked, inp_geo],
                                  outputs=[Fully_Connected_all, geo_ext, Li_ext])
                else:
                    Fully_Connected_all = fusion_method(Geo_branch, Lidar_branch)
                    model = Model(inputs=[inp_lidar_stacked, inp_geo],
                                  outputs=Fully_Connected_all)
            elif self.rot_rep == 'dquart':
                if self.fusion == 'grad':
                    Fully_Connected_all = fusion_method(Geo_branch, Lidar_branch, g_grad, L_grad)
                    model = Model(inputs=[inp_lidar_stacked, inp_geo],
                                  outputs=[Fully_Connected_all, geo_ext, Li_ext])
                elif self.fusion == 'grad2':
                    Fully_Connected_all = fusion_method(Geo_branch, Lidar_branch, inp_lidar_stacked)
                    model = Model(inputs=[inp_lidar_stacked, inp_geo],
                                  outputs=[Fully_Connected_all])
                else:
                    Fully_Connected_all = fusion_method(Geo_branch, Lidar_branch)
                    model = Model(inputs=[inp_lidar_stacked, inp_geo],
                                  outputs=Fully_Connected_all)

            else:
                Fully_Connected_T, Fully_Connected_Q = fusion_method(Geo_branch, Lidar_branch)
                model = Model(inputs=[inp_lidar_stacked, inp_geo],
                              outputs=[Fully_Connected_T, Fully_Connected_Q])

        elif self.branch == 'lidar':
            Fully_Connected_T, Fully_Connected_Q = self.no_fuse(Lidar_branch)
            model = Model(inputs=[inp_lidar_stacked],
                          outputs=[Fully_Connected_T, Fully_Connected_Q])

        elif self.branch == 'geo':
            if self.rot_rep == 'expn':
                Fully_Connected_all = self.no_fuse_exp(Geo_branch)
                model = Model(inputs=[inp_geo],
                              outputs=[Fully_Connected_all])

            elif self.rot_rep == 'dquart':
                Fully_Connected_all = self.no_fuse_dq(Geo_branch)
                model = Model(inputs=[inp_geo],
                              outputs=[Fully_Connected_all])

            else:
                Fully_Connected_T, Fully_Connected_Q = self.no_fuse(Geo_branch)
                model = Model(inputs=[inp_geo],
                              outputs=[Fully_Connected_T, Fully_Connected_Q])

        #  _____________________________________________________________________________________________________________
        # Save Model
        #  _____________________________________________________________________________________________________________
        model.summary()
        filename = os.path.join(self.mother_folder, 'results', 'model1.png')
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=filename)

        # ______________________________________________________________________________________________________________
        # SELF SUPERVISED
        # ______________________________________________________________________________________________________________
        if self.method == 'self_supervised':
            model.compile(loss=self.li_loss6, optimizer='adam')
            # model.fit(
            #     x={'geo_input': G_data, 'AI_input': AI_data},
            #     y=AI_data,
            #     epochs=self.Epochs, batch_size=self.Batch_size,
            #     verbose=1)

        # ______________________________________________________________________________________________________________
        # SUPERVISED
        # ______________________________________________________________________________________________________________
        if self.method == 'supervised':
            if self.rot_rep == 'expn' or self.rot_rep == 'dquart':
                model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam',
                              loss_weights=100)
            else:
                model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam',
                              loss_weights={'Translation': self.loss_weights[0], 'Quaternion': self.loss_weights[1]})

        # ______________________________________________________________________________________________________________
        # Data preparation
        # ______________________________________________________________________________________________________________
        if self.data_pre == 'saved' or 'saved_all':
            if self.method == 'self_supervised':
                x = {'geo_input': G_data, 'AI_input': AI_data}
            else:
                if self.branch == 'all':
                    x = {'geo_input': G_data, 'AI_input': AI_data}
                elif self.branch == 'lidar':
                    x = {'AI_input': AI_data}
                elif self.branch == 'geo':
                    x = {'geo_input': G_data}

        # ______________________________________________________________________________________________________________
        # Training
        # ______________________________________________________________________________________________________________
        if self.data_pre == 'tfdata':
            steps = np.int(np.floor(np.divide((self.e_idx - self.s_idx), self.Batch_size)))
            history = model.fit(self._input_fn(),
                                steps_per_epoch=steps,
                                epochs=self.Epochs,
                                verbose=1)
        elif self.data_pre == 'tfrecord':
            # steps = np.int(np.floor(np.divide((self.e_idx - self.s_idx), self.Batch_size)))
            # filename = os.path.join(self.mother_folder, 'saved_model', 'model1.h5')
            # model = load_model(filename, custom_objects={'li_loss6': self.li_loss6})
            history = model.fit(self.get_dataset(),
                                # steps_per_epoch=steps,
                                epochs=self.Epochs,
                                verbose=1)
        elif self.data_pre == 'saved_all':
            G_create = Geometry_data_prepare(self.mother_folder)

            loader2 = Lidar_data_prepare(self.mother_folder)
            train_loss = []
            val_loss = []
            trans_loss = []
            quart_loss = []
            n_epochs_best = []

            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)

            for seq in self.sequences:
                seq_int = int(seq)
                seq_scans = np.int(self.scans[seq_int])

                print("_______________________________________________________________________________________________"
                      "_______________________________________________________________________________")
                print('Sequence started: ', seq)
                print("_______________________________________________________________________________________________"
                      "_______________________________________________________________________________")

                G_data, G_gt = G_create.load_saved_data_all(seq)

                mu = self.divided_train
                seq_scans = seq_scans
                inside_loop = np.int(np.ceil(np.divide(seq_scans, mu)))

                for counter in range(inside_loop):
                    temp_start = counter * mu
                    # if counter == 0: temp_start = 5
                    temp_end = temp_start + mu
                    if seq_scans - 5 < temp_end: temp_end = seq_scans - 5

                    print('Sequence', seq, 'part', counter, ': ', temp_start, 'to', temp_end, 'of', seq_scans + 5,
                          '_____________________________________________________')

                    AI_data_temp = loader2.load_saved_data_h5(seq, temp_start, temp_end)
                    G_data_temp = G_data[temp_start: temp_end]
                    G_gt_temp = G_gt[temp_start: temp_end]

                    if self.method == 'supervised':
                        x = {'geo_input': G_data_temp, 'AI_input': AI_data_temp}
                        print('here', x['geo_input'].shape, x['AI_input'].shape)
                        if self.fusion == 'grad':
                            y = [G_gt_temp, G_gt_temp, G_gt_temp]
                        else:
                            y = G_gt_temp
                        train_history = model.fit(x=x,
                                                  # y={'Translation': G_gt_temp[:, 0:3], 'Quaternion': G_gt_temp[:,
                                                  # 3:7]},
                                                  y=y,
                                                  epochs=self.Epochs,
                                                  batch_size=self.Batch_size,
                                                  validation_split=0.1,
                                                  # callbacks=[callback],
                                                  verbose=1)
                    else:
                        x = {'geo_input': G_data_temp, 'AI_input': AI_data_temp}
                        y = AI_data_temp
                        train_history = model.fit(
                            x=x,
                            y=y,
                            epochs=self.Epochs, batch_size=self.Batch_size,
                            verbose=1)
                    n_epochs_best_temp = np.argmax(train_history.history['val_loss'])
                    print(n_epochs_best_temp)
                    n_epochs_best.append(n_epochs_best_temp)
                    train_loss.append(train_history.history['loss'])
                    val_loss.append(train_history.history['val_loss'])
                    # trans_loss.append(train_history.history['Translation_loss'])
                    # quart_loss.append(train_history.history['Quaternion_loss'])
                    gc.collect()
        else:
            history = model.fit(x=x,
                                y={'Translation': G_gt[:, 0:3], 'Quaternion': G_gt[:, 3:7]},
                                epochs=self.Epochs,
                                batch_size=self.Batch_size,
                                validation_split=0.1,
                                verbose=1)

        # ______________________________________________________________________________________________________________
        # Save Model
        # ______________________________________________________________________________________________________________
        filename = os.path.join(self.mother_folder, 'results', 'saved_model', 'model1.h5')
        model.save(filename)
        print('_______________________________________________________________________________________________________')
        print('Model saved to disk')
        print('_______________________________________________________________________________________________________')

        # ______________________________________________________________________________________________________________
        # PLOT History
        # ______________________________________________________________________________________________________________
        if self.data_pre == 'saved_all':
            filename = os.path.join(self.mother_folder, 'results', 'train_loss.npy')
            np.save(filename, train_loss)
            filename = os.path.join(self.mother_folder, 'results', 'val_loss.npy')
            np.save(filename, val_loss)
            # filename = os.path.join(self.mother_folder, 'results', 'trans_loss.npy')
            # np.save(filename, trans_loss)
            # filename = os.path.join(self.mother_folder, 'results', 'quart_loss.npy')
            # np.save(filename, quart_loss)
            filename = os.path.join(self.mother_folder, 'results', 'n_epochs_best.npy')
            np.save(filename, n_epochs_best)

            # plt.plot(train_loss)
            # plt.plot(val_loss)
            # plt.plot(trans_loss)
            # plt.plot(quart_loss)
            plt.show()
        else:
            filename = os.path.join(self.mother_folder, 'results', 'history1.npy')
            np.save(filename, history.history)
            # print(history.history.keys())

            plt.plot(history.history['Translation_loss'])
            plt.plot(history.history['Quaternion_loss'])
            plt.title('Translation_loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['Translation_loss', 'Quaternion_loss'], loc='upper left')
            plt.show()

            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()

        return model

    def load_savedmodel(self, G_data=None, AI_data=None):

        filename = os.path.join(self.mother_folder, 'results', 'saved_model', 'model1.h5')
        model = load_model(filename, custom_objects={'li_loss6': self.li_loss6})
        # model.summary()

        if self.data_pre == 'tfdata':
            steps = np.int(np.floor(np.divide((self.e_idx - self.s_idx), self.Batch_size)))
            y = model.predict(x=self._input_fn(),
                              steps=steps,
                              batch_size=self.Batch_size,
                              verbose=1)

        elif self.data_pre == 'tfrecord':
            y = model.predict(self.get_dataset(),
                              verbose=1)

        elif self.data_pre == 'saved_all':
            G_create = Geometry_data_prepare(self.mother_folder)
            loader2 = Lidar_data_prepare(self.mother_folder)

            for seq in self.sequences:
                seq_int = int(seq)
                seq_scans = np.int(self.scans[seq_int])
                print("_______________________________________________________________________________________________"
                      "_______________________________________________________________________________")
                print('Sequence started: ', seq)
                print("_______________________________________________________________________________________________"
                      "_______________________________________________________________________________")

                G_data, G_gt = G_create.load_saved_data_all(seq)

                mu = self.divided_train
                seq_scans = seq_scans
                inside_loop = np.int(np.ceil(np.divide(seq_scans, mu)))

                for counter in range(inside_loop):
                    temp_start = counter * mu
                    # if counter == 0: temp_start = 5
                    temp_end = temp_start + mu
                    if seq_scans - 5 < temp_end: temp_end = seq_scans - 5

                    print('Sequence', seq, 'part', counter, ': ', temp_start, 'to', temp_end, 'of', seq_scans + 5,
                          '***************************************************************')

                    if not self.branch == 'geo':
                        AI_data_temp = loader2.load_saved_data_h5(seq, temp_start, temp_end)
                    G_data_temp = G_data[temp_start: temp_end]
                    G_gt_temp = G_gt[temp_start: temp_end]

                    if not self.branch == 'geo':
                        x = {'geo_input': G_data_temp, 'AI_input': AI_data_temp}
                    else:
                        x = {'geo_input': G_data_temp}

                    y = model.predict(x=x, verbose=1)
                    gc.collect()
                    if counter == 0:
                        Tlast = np.eye(4)
                    else:
                        Tlast = Tlast_next
                        # print(Tlast_next)

                    if self.rot_rep == 'expn':
                        Tlast_next = save2txt(y, 'result_all', self.mother_folder, file_id=seq, part=counter,
                                              start_i=temp_start + 5, end_i=temp_end + 5, Tlast=Tlast)
                    elif self.rot_rep == 'dquart':
                        Tlast_next = save2txt(y, 'result_all_dq', self.mother_folder, file_id=seq, part=counter,
                                              start_i=temp_start + 5, end_i=temp_end + 5, Tlast=Tlast)

        else:
            if self.branch == 'all':
                x_input = {'geo_input': G_data, 'AI_input': AI_data},
            elif self.branch == 'lidar':
                x_input = {'AI_input': AI_data},
            elif self.branch == 'geo':
                x_input = {'geo_input': G_data}
            y = model.predict(x_input, batch_size=self.Batch_size)

        return y

    def tf_generator(self):
        count = 0
        while True:
            # start_time = time.time()
            # _____________________________________________________________________________________________________Batch
            batch_start = self.s_idx + count
            # print(batch_start)
            # ________________________________________________________________________________________________Load_Lidar
            AI_data_batch = self.li_create.create_lidar_data_timedist(batch_s=batch_start)
            AI_data_batch = np.transpose(AI_data_batch, (0, 2, 3, 1))
            # __________________________________________________________________________________________________Geo_data
            G_data_batch, gt_data = self.G_create.create_geo_timedist_tfdata(batch_start)
            # __________________________________________________________________________________________________________

            x_batch = {'geo_input': G_data_batch, 'AI_input': AI_data_batch}
            y_batch = {'Translation': gt_data[0:3], 'Quaternion': gt_data[3:7]}

            count += 1
            if count >= tf.floor(tf.divide((self.e_idx - self.s_idx), self.Batch_size)):
                count = 0

            # print('time: ', (time.time() - start_time))
            yield x_batch, y_batch

    def _input_fn(self):
        # start_time = time.time()
        dataset = tf.data.Dataset.from_generator(self.tf_generator,
                                                 output_types=(
                                                     {'geo_input': tf.float64, 'AI_input': tf.float64},
                                                     {'Translation': tf.float64, 'Quaternion': tf.float64}),
                                                 output_shapes=(
                                                     {'geo_input': (4, 7), 'AI_input': (4, 64, 720, 14)},
                                                     {'Translation': 3, 'Quaternion': 4})
                                                 )
        # it = iter(dataset)
        dataset = dataset.repeat(self.Epochs)
        dataset = dataset.batch(self.Batch_size, drop_remainder=True)
        # dataset = dataset.cache()
        dataset = dataset.prefetch(1)

        # iterator = dataset.make_one_shot_iterator()
        # while True:
        #     batch_features, batch_labels = iterator.get_next()
        #     yield batch_features, batch_labels
        # print('time_input: ', (time.time() - start_time))
        return dataset

    def get_dataset(self):
        filename = os.path.join('tfrecorded_files', str(self.s_idx) + '_' + str(self.e_idx) + '.tfrecords')
        dataset = self.recorder_reader.load_tfrecord(filename)
        dataset = dataset.prefetch(1)
        dataset = dataset.batch(self.Batch_size)
        return dataset

