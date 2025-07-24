import numpy as np
import tensorflow as tf
import yaml
import os

from Mytools.make_pcfile_4network import Lidar_data_prepare
from Mytools.pre_geo_data import Geometry_data_prepare


class recorder_reader:

    def __init__(self, mother_folder):
        self.mother_folder = mother_folder
        self.li_create = Lidar_data_prepare(self.mother_folder)
        self.G_create = Geometry_data_prepare(self.mother_folder)

        with open('Mytools/config.yaml', 'r') as stream:
            cfg = yaml.safe_load(stream)

        ds_config = cfg['datasets']['kitti']
        self.image_width = ds_config.get('image-width', 1024)
        self.image_height = ds_config.get('image-height', 64)
        self.channels = ds_config['channels']
        self.channels_N = np.size(self.channels)
        self.s_idx = ds_config.get('s_idx')
        self.e_idx = ds_config.get('e_idx')

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

    def tf_recorder(self):
        filename = os.path.join('tfrecorded_files', str(self.s_idx) + '_' + str(self.e_idx) + '.tfrecords')
        options = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP)
        writer = tf.io.TFRecordWriter(filename, options=options)

        for index in range(self.s_idx, self.e_idx):
            out = self.parse_single_image(index)
            writer.write(out.SerializeToString())
            print('tf recorded for scan= ', index)

        writer.close()

    def parse_single_image(self, idx):
        # ________________________________________________________________________________________________Load_Lidar
        AI_data_batch = self.li_create.create_lidar_data_timedist(batch_s=idx)
        AI_data_batch = np.transpose(AI_data_batch, (0, 2, 3, 1))
        # __________________________________________________________________________________________________Geo_data
        G_data_batch, gt_data = self.G_create.create_geo_timedist_tfdata(idx)
        # __________________________________________________________________________________________________________

        # define the dictionary -- the structure -- of our single example
        data = {
            'geo_input': _bytes_feature(serialize_array(G_data_batch)),
            'AI_input': _bytes_feature(serialize_array(AI_data_batch)),
            'Translation': _bytes_feature(serialize_array(gt_data[0:3])),
            'Quaternion': _bytes_feature(serialize_array(gt_data[3:7]))
        }
        # create an Example, wrapping the single features
        out = tf.train.Example(features=tf.train.Features(feature=data))

        return out

    def load_tfrecord(self, filename):
        # create the dataset
        dataset = tf.data.TFRecordDataset(filename, compression_type='GZIP')

        # pass every single feature through our mapping function
        dataset = dataset.map(parse_tfr_element)

        return dataset


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'geo_input': tf.io.FixedLenFeature([], tf.string),
        'AI_input': tf.io.FixedLenFeature([], tf.string),
        'Translation': tf.io.FixedLenFeature([], tf.string),
        'Quaternion': tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(element, data)

    geo_input = content['geo_input']
    AI_input = content['AI_input']
    Translation = content['Translation']
    Quaternion = content['Quaternion']

    # get our 'feature'-- our image -- and reshape it appropriately
    feature_geo = tf.io.parse_tensor(geo_input, out_type=tf.float64)
    feature_geo = tf.reshape(feature_geo, shape=[4, 7])

    feature_AI_input = tf.io.parse_tensor(AI_input, out_type=tf.float64)
    feature_AI_input = tf.reshape(feature_AI_input, shape=[4, 64, 720, 14])

    feature_Translation = tf.io.parse_tensor(Translation, out_type=tf.float64)
    feature_Translation = tf.reshape(feature_Translation, shape=[3, ])

    feature_Quaternion = tf.io.parse_tensor(Quaternion, out_type=tf.float64)
    feature_Quaternion = tf.reshape(feature_Quaternion, shape=[4, ])

    x_batch = {'geo_input': feature_geo, 'AI_input': feature_AI_input}
    y_batch = {'Translation': feature_Translation, 'Quaternion': feature_Quaternion}

    return x_batch, y_batch
