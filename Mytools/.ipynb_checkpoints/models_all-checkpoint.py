import tensorflow as tf
from tensorflow.keras import layers, Model, Input, backend, activations, initializers
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from typing import Tuple
# from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import ResNet50

from Mytools.MultuHeadAtt import MultiHeadAttention

momentum_p = 0.9
# act_geo = 'tanh'
# act_lstm = 'tanh'
# act_lstm = 'tanh'
alpha_relu = 0.3



drop_rate = 0.5


class DualQuaternionNormalization(tf.keras.layers.Layer):
    def call(self, inputs):
        qr, qd = tf.split(inputs, 2, axis=-1)
        qr = tf.linalg.l2_normalize(qr, axis=-1)
        return tf.concat([qr, qd], axis=-1)

# --------------------------------------------------------------------------------------------------
# Model: geo_s2e_nowe — GEO-only network (No Attention or Fusion), Dual Quaternion normalized output
# --------------------------------------------------------------------------------------------------
class geo_s2e_nowe(Model):
    
    def __init__(self, 
                 act_geo1='relu', act_geo2='relu', act_geo3='relu', act_geo4='relu',
                 dropout_rate_1=0.1, dropout_rate_2=0.1, 
                 e1=32, e2=64, e3=128, 
                 d3=128, d2=64, 
                 act_lstm1='tanh', act_lstm2="sigmoid", 
                 do_lstm_1=0.2, do_lstm_2=0.2, do_lstm_3=0.2, do_lstm_4=0.2,
                 act16='tanh',
                 **kwargs):
        super().__init__(**kwargs)

        # Input normalization
        self.BN_geo_input = layers.TimeDistributed(layers.BatchNormalization(name='BN_geo_inp'))

        # Encoder: 3 fully connected layers
        self.Fully_Connected_en_1 = layers.TimeDistributed(
            layers.Dense(e1, activation=act_geo1, name='FC_geo_en_1', kernel_regularizer=l2(1e-4)))
        self.Fully_Connected_en_2 = layers.TimeDistributed(
            layers.Dense(e2, activation=act_geo2, name='FC_geo_en_2', kernel_regularizer=l2(1e-4)))
        self.Fully_Connected_en_3 = layers.TimeDistributed(
            layers.Dense(e3, activation=act_geo3, name='FC_geo_en_3'))
        self.drop_1 = layers.TimeDistributed(layers.Dropout(rate=dropout_rate_1, name='drop_geo_1'))

        # Decoder: 1 dense layer
        self.Fully_Connected_de_2 = layers.TimeDistributed(
            layers.Dense(d3, activation=act_geo4, name='FC_geo_de_1'))
        self.drop_2 = layers.TimeDistributed(layers.Dropout(rate=dropout_rate_2, name='drop_geo_2'))

        # LSTM layers
        self.RNN_LSTM_1 = layers.LSTM(d2, activation=act_lstm1, recurrent_activation=act_lstm2,
                                      return_sequences=True, unroll=False, use_bias=True,
                                      dropout=do_lstm_1, recurrent_dropout=do_lstm_2,
                                      kernel_regularizer=l2(1e-4), name='LSTM_end_1')
        self.RNN_LSTM_2 = layers.LSTM(32, activation=act_lstm1, recurrent_activation=act_lstm2,
                                      unroll=False, use_bias=True,
                                      dropout=do_lstm_3, recurrent_dropout=do_lstm_4,
                                      kernel_regularizer=l2(1e-4), name='LSTM_end_2')

        # Output layers
        self.Fully_Connected_end1 = layers.Dense(16, activation=act16,
                                                 kernel_regularizer=l2(1e-4), name='FC_all_end0')
        self.Fully_Connected_end = layers.Dense(8, activation=None, name='FC_all_end')

        # Output normalization (custom)
        self.DualQuaternionNormalization = DualQuaternionNormalization()

    @tf.function
    def call(self, geo_lidar_input, training: bool = True):
        geo_inp, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        # Forward pass
        x1_geo = self.BN_geo_input(geo_inp, training=training)
        x2_geo = self.Fully_Connected_en_1(x1_geo, training=training)
        x2_geo = self.Fully_Connected_en_2(x2_geo, training=training)
        x2_geo = self.Fully_Connected_en_3(x2_geo, training=training)
        x2_geo = self.drop_1(x2_geo, training=training)

        x_mid = self.Fully_Connected_de_2(x2_geo, training=training)
        x_mid = self.drop_2(x_mid, training=training)

        outp = self.RNN_LSTM_1(x_mid, training=training)
        outp = self.RNN_LSTM_2(outp, training=training)
        outp = self.Fully_Connected_end1(outp, training=training)
        outp = self.Fully_Connected_end(outp, training=training)

        # Normalize output
        outp = self.DualQuaternionNormalization(outp)
        return outp

# --------------------------------------------------------------------------------------------------
# Model: geo_s2e_nowe_two — Same as geo_s2e_nowe but returns intermediate states + gradients
# --------------------------------------------------------------------------------------------------
class geo_s2e_nowe_two(Model):

    def __init__(self, 
                 act_geo1='relu', act_geo2='relu', act_geo3='relu', act_geo4='relu',
                 dropout_rate_1=0.1, dropout_rate_2=0.1, 
                 e1=32, e2=64, e3=128, 
                 d3=128, d2=64, 
                 act_lstm1='tanh', act_lstm2="sigmoid", 
                 do_lstm_1=0.2, do_lstm_2=0.2, do_lstm_3=0.2, do_lstm_4=0.2,
                 act16='tanh',
                 **kwargs):
        super().__init__(**kwargs)

        # Same architecture as geo_s2e_nowe
        self.BN_geo_input = layers.TimeDistributed(layers.BatchNormalization(name='BN_geo_inp'))

        self.Fully_Connected_en_1 = layers.TimeDistributed(
            layers.Dense(e1, activation=act_geo1, name='FC_geo_en_1', kernel_regularizer=l2(1e-4)))
        self.Fully_Connected_en_2 = layers.TimeDistributed(
            layers.Dense(e2, activation=act_geo2, name='FC_geo_en_2', kernel_regularizer=l2(1e-4)))
        self.Fully_Connected_en_3 = layers.TimeDistributed(
            layers.Dense(e3, activation=act_geo3, name='FC_geo_en_3'))
        self.drop_1 = layers.TimeDistributed(layers.Dropout(rate=dropout_rate_1, name='drop_geo_1'))

        self.Fully_Connected_de_2 = layers.TimeDistributed(
            layers.Dense(d3, activation=act_geo4, name='FC_geo_de_1'))
        self.drop_2 = layers.TimeDistributed(layers.Dropout(rate=dropout_rate_2, name='drop_geo_2'))

        self.RNN_LSTM_1 = layers.LSTM(d2, activation=act_lstm1, recurrent_activation=act_lstm2,
                                      return_sequences=True, unroll=False, use_bias=True,
                                      dropout=do_lstm_1, recurrent_dropout=do_lstm_2,
                                      kernel_regularizer=l2(1e-4), name='LSTM_end_1')
        self.RNN_LSTM_2 = layers.LSTM(32, activation=act_lstm1, recurrent_activation=act_lstm2,
                                      unroll=False, use_bias=True,
                                      dropout=do_lstm_3, recurrent_dropout=do_lstm_4,
                                      kernel_regularizer=l2(1e-4), name='LSTM_end_2')

        self.Fully_Connected_end1 = layers.Dense(16, activation=act16,
                                                 kernel_regularizer=l2(1e-4), name='FC_all_end0')
        self.Fully_Connected_end = layers.Dense(8, activation=None, name='FC_all_end')

        self.DualQuaternionNormalization = DualQuaternionNormalization()

    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = False):
        geo_inp, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        x1_geo = self.BN_geo_input(geo_inp, training=training)
        x2_geo = self.Fully_Connected_en_1(x1_geo, training=training)
        x2_geo = self.Fully_Connected_en_2(x2_geo, training=training)
        x2_geo = self.Fully_Connected_en_3(x2_geo, training=training)
        x2_geo = self.drop_1(x2_geo, training=training)

        x_mid = self.Fully_Connected_de_2(x2_geo, training=training)
        x_mid = self.drop_2(x_mid, training=training)

        outp = self.RNN_LSTM_1(x_mid, training=training)
        outp = self.RNN_LSTM_2(outp, training=training)
        outp = self.Fully_Connected_end1(outp, training=training)
        outp = self.Fully_Connected_end(outp, training=training)
        outp = self.DualQuaternionNormalization(outp)

        # Return gradient per output dimension (deprecated style)
        self.layer_grad0 = tf.gradients(outp[:, 0], outp)
        self.layer_grad1 = tf.gradients(outp[:, 1], outp)
        self.layer_grad2 = tf.gradients(outp[:, 2], outp)
        self.layer_grad3 = tf.gradients(outp[:, 3], outp)
        self.layer_grad4 = tf.gradients(outp[:, 4], outp)
        self.layer_grad5 = tf.gradients(outp[:, 5], outp)
        self.layer_grad6 = tf.gradients(outp[:, 6], outp)
        self.layer_grad7 = tf.gradients(outp[:, 7], outp)

        return outp, x_mid, x_mid, \
            self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3, \
            self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7
# --------------------------------------------------------------------------------------------------
# Model: geo_s2e_att — GEO-only model with Attention after LSTM
# --------------------------------------------------------------------------------------------------
class geo_s2e_att(Model):
    def __init__(self,
                 act_geo1='relu', act_geo2='relu', act_geo3='relu', act_geo4='relu',
                 dropout_rate_1=0.1, dropout_rate_2=0.1,
                 e1=32, e2=64, e3=128,
                 d3=128, d2=64,
                 act_lstm1='tanh', act_lstm2="sigmoid",
                 do_lstm_1=0.2, do_lstm_2=0.2, do_lstm_3=0.2, do_lstm_4=0.2,
                 act16='tanh',
                 **kwargs):
        super().__init__(**kwargs)

        self.BN_geo_input = layers.TimeDistributed(layers.BatchNormalization(name='BN_geo_inp'))

        self.Fully_Connected_en_1 = layers.TimeDistributed(
            layers.Dense(e1, activation=act_geo1, name='FC_geo_en_1', kernel_regularizer=l2(1e-4)))
        self.Fully_Connected_en_2 = layers.TimeDistributed(
            layers.Dense(e2, activation=act_geo2, name='FC_geo_en_2', kernel_regularizer=l2(1e-4)))
        self.Fully_Connected_en_3 = layers.TimeDistributed(
            layers.Dense(e3, activation=act_geo3, name='FC_geo_en_3'))
        self.drop_1 = layers.TimeDistributed(layers.Dropout(rate=dropout_rate_1, name='drop_geo_1'))

        self.Fully_Connected_de_2 = layers.TimeDistributed(
            layers.Dense(d3, activation=act_geo4, name='FC_geo_de_1'))
        self.drop_2 = layers.TimeDistributed(layers.Dropout(rate=dropout_rate_2, name='drop_geo_2'))

        self.RNN_LSTM_1 = layers.LSTM(d2,
                                      activation=act_lstm1,
                                      recurrent_activation=act_lstm2,
                                      return_sequences=True,
                                      unroll=False,
                                      use_bias=True,
                                      dropout=do_lstm_1,
                                      recurrent_dropout=do_lstm_2,
                                      kernel_regularizer=l2(1e-4),
                                      name='LSTM_end_1')

        self.RNN_LSTM_2 = layers.LSTM(32,
                                      activation=act_lstm1,
                                      recurrent_activation=act_lstm2,
                                      return_sequences=False,  # Output shape: [batch, features]
                                      unroll=False,
                                      use_bias=True,
                                      dropout=do_lstm_3,
                                      recurrent_dropout=do_lstm_4,
                                      kernel_regularizer=l2(1e-4),
                                      name='LSTM_end_2')

        # MultiHeadAttention (no TimeDistributed needed since output is [batch, time, feat])
        self.MultiHeadAtt = layers.MultiHeadAttention(num_heads=4, key_dim=8, name='MHA')

        self.LayerNorm = layers.LayerNormalization()
        self.drop_att = layers.Dropout(rate=0.1, name='drop_att')

        self.Fully_Connected_end1 = layers.Dense(16, activation=act16,
                                                 kernel_regularizer=l2(1e-4), name='FC_all_end0')
        self.Fully_Connected_end = layers.Dense(8, activation=None, name='FC_all_end')

        self.DualQuaternionNormalization = DualQuaternionNormalization()

    @tf.function
    def call(self, geo_lidar_input, training: bool = True):
        geo_inp, _ = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        x = self.BN_geo_input(geo_inp, training=training)
        x = self.Fully_Connected_en_1(x, training=training)
        x = self.Fully_Connected_en_2(x, training=training)
        x = self.Fully_Connected_en_3(x, training=training)
        x = self.drop_1(x, training=training)

        x = self.Fully_Connected_de_2(x, training=training)
        x = self.drop_2(x, training=training)

        x = self.RNN_LSTM_1(x, training=training)

        # Multi-head attention block
        att_out = self.MultiHeadAtt(query=x, key=x, value=x, training=training)
        att_out = self.drop_att(att_out, training=training)
        att_out = self.LayerNorm(x + att_out)  # residual connection

        # Final LSTM: compress to [batch, features]
        x = self.RNN_LSTM_2(att_out, training=training)

        x = self.Fully_Connected_end1(x, training=training)
        x = self.Fully_Connected_end(x, training=training)
        x = self.DualQuaternionNormalization(x)

        return x
# --------------------------------------------------------------------------------------------------
# Model: geo_s2e_att_two returns intermediate states + gradients
# --------------------------------------------------------------------------------------------------
class geo_s2e_att_two(Model):
    def __init__(self,
                 act_geo1='relu', act_geo2='relu', act_geo3='relu', act_geo4='relu',
                 dropout_rate_1=0.1, dropout_rate_2=0.1,
                 e1=32, e2=64, e3=128,
                 d3=128, d2=64,
                 act_lstm1='tanh', act_lstm2="sigmoid",
                 do_lstm_1=0.2, do_lstm_2=0.2, do_lstm_3=0.2, do_lstm_4=0.2,
                 act16='tanh',
                 **kwargs):
        super().__init__(**kwargs)

        self.BN_geo_input = layers.TimeDistributed(layers.BatchNormalization(name='BN_geo_inp'))

        self.Fully_Connected_en_1 = layers.TimeDistributed(
            layers.Dense(e1, activation=act_geo1, name='FC_geo_en_1', kernel_regularizer=l2(1e-4)))
        self.Fully_Connected_en_2 = layers.TimeDistributed(
            layers.Dense(e2, activation=act_geo2, name='FC_geo_en_2', kernel_regularizer=l2(1e-4)))
        self.Fully_Connected_en_3 = layers.TimeDistributed(
            layers.Dense(e3, activation=act_geo3, name='FC_geo_en_3'))
        self.drop_1 = layers.TimeDistributed(layers.Dropout(rate=dropout_rate_1, name='drop_geo_1'))

        self.Fully_Connected_de_2 = layers.TimeDistributed(
            layers.Dense(d3, activation=act_geo4, name='FC_geo_de_1'))
        self.drop_2 = layers.TimeDistributed(layers.Dropout(rate=dropout_rate_2, name='drop_geo_2'))

        self.RNN_LSTM_1 = layers.LSTM(d2,
                                      activation=act_lstm1,
                                      recurrent_activation=act_lstm2,
                                      return_sequences=True,
                                      unroll=False,
                                      use_bias=True,
                                      dropout=do_lstm_1,
                                      recurrent_dropout=do_lstm_2,
                                      kernel_regularizer=l2(1e-4),
                                      name='LSTM_end_1')

        self.RNN_LSTM_2 = layers.LSTM(32,
                                      activation=act_lstm1,
                                      recurrent_activation=act_lstm2,
                                      return_sequences=False,  # Output shape: [batch, features]
                                      unroll=False,
                                      use_bias=True,
                                      dropout=do_lstm_3,
                                      recurrent_dropout=do_lstm_4,
                                      kernel_regularizer=l2(1e-4),
                                      name='LSTM_end_2')

        # MultiHeadAttention (no TimeDistributed needed since output is [batch, time, feat])
        self.MultiHeadAtt = layers.MultiHeadAttention(num_heads=4, key_dim=8, name='MHA')

        self.LayerNorm = layers.LayerNormalization()
        self.drop_att = layers.Dropout(rate=0.1, name='drop_att')

        self.Fully_Connected_end1 = layers.Dense(16, activation=act16,
                                                 kernel_regularizer=l2(1e-4), name='FC_all_end0')
        self.Fully_Connected_end = layers.Dense(8, activation=None, name='FC_all_end')

        self.DualQuaternionNormalization = DualQuaternionNormalization()

    @tf.function
    def call(self, geo_lidar_input, training: bool = True):
        geo_inp, _ = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        x = self.BN_geo_input(geo_inp, training=training)
        x = self.Fully_Connected_en_1(x, training=training)
        x = self.Fully_Connected_en_2(x, training=training)
        x = self.Fully_Connected_en_3(x, training=training)
        x = self.drop_1(x, training=training)

        x = self.Fully_Connected_de_2(x, training=training)
        x = self.drop_2(x, training=training)

        x = self.RNN_LSTM_1(x, training=training)

        # Multi-head attention block
        att_out = self.MultiHeadAtt(query=x, key=x, value=x, training=training)
        att_out = self.drop_att(att_out, training=training)
        att_out = self.LayerNorm(x + att_out)  # residual connection

        # Final LSTM: compress to [batch, features]
        x = self.RNN_LSTM_2(att_out, training=training)

        x = self.Fully_Connected_end1(x, training=training)
        x = self.Fully_Connected_end(x, training=training)
        x = self.DualQuaternionNormalization(x)

         # Calculate gradients for each output element
        self.layer_grad0 = tf.gradients(x[:, 0], x)
        self.layer_grad1 = tf.gradients(x[:, 1], x)
        self.layer_grad2 = tf.gradients(x[:, 2], x)
        self.layer_grad3 = tf.gradients(x[:, 3], x)
        self.layer_grad4 = tf.gradients(x[:, 4], x)
        self.layer_grad5 = tf.gradients(x[:, 5], x)
        self.layer_grad6 = tf.gradients(x[:, 6], x)
        self.layer_grad7 = tf.gradients(x[:, 7], x)

        return x, att_out, att_out, \
            self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3, \
            self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7
    
# ______________________________________________________________________________________________________________________
# LiDAR Only
# ______________________________________________________________________________________________________________________
class LiD_s2e_nowe(Model):

    def __init__(self, 
                 # _____________________________________________________resnet
                 # num_filters=64, filters1=64, 
                 filters2=64, filters3=128, filters4=256,
                 dense_last=512, act_last='relu',
                 # _____________________________________________________Lidar
                 act_li='relu', dropout4=0.2,
                 d1_li=256, d2_li=128, d3_li=64, d4_li=32,
                 d_lstm_li1=32, 
                 # d_lstm_li2=32, d_lstm_li3=16,
                 act_lstm1='tanh', act_lstm2="sigmoid", 
                 act16='tanh',
                 do_lstm_1=0.1, do_lstm_2=0.1, do_lstm_3=0.1, do_lstm_4=0.1,
                 # _____________________________________________________
                 training=True, 
                 **kwargs):
        super().__init__(**kwargs)
                       
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(training=True, 
                                                                           num_filters=64,filters1=64, #Don't touch
                                                                           filters2=filters2, filters3=filters3, filters4=filters4, 
                                                                           dense_last=dense_last, act_last=act_last))   
        
        self.Fully_Connected_li1 = layers.TimeDistributed(layers.Dense(d1_li, activation=act_li, name='FC_lidar_1')) #1
        self.Fully_Connected_li2 = layers.TimeDistributed(layers.Dense(d2_li, activation=act_li, name='FC_lidar_2')) #2
        self.Fully_Connected_li3 = layers.TimeDistributed(layers.Dense(d3_li, activation=act_li, name='FC_lidar_3')) #3
        self.Fully_Connected_li4 = layers.TimeDistributed(layers.Dense(d4_li, activation=act_li, name='FC_lidar_4'))
        
        self.Dropout_li1 = layers.TimeDistributed(layers.Dropout(dropout4, name='dropout_lidar_1'))
        self.Dropout_li2 = layers.TimeDistributed(layers.Dropout(dropout4, name='dropout_lidar_2'))
        self.Dropout_li3 = layers.TimeDistributed(layers.Dropout(dropout4, name='dropout_lidar_3'))
        self.Dropout_li4 = layers.TimeDistributed(layers.Dropout(dropout4, name='dropout_lidar_4'))

        
        self.RNN_LSTM_1 = layers.LSTM(d_lstm_li1,
                                      activation=act_lstm1,
                                      recurrent_activation=act_lstm2,
                                      return_sequences=True,
                                      unroll=False,
                                      use_bias=True,
                                      dropout=do_lstm_1,  # Dropout rate for input units
                                      recurrent_dropout=do_lstm_2,  # Dropout rate for recurrent units
                                      # kernel_regularizer=l2(1e-4),
                                      name='LSTM_end_1')
        
        self.RNN_LSTM_2 = layers.LSTM(32,
                                      activation=act_lstm1,
                                      recurrent_activation=act_lstm2,
                                      unroll=False,
                                      use_bias=True,
                                      dropout=do_lstm_3,  # Dropout rate for input units
                                      recurrent_dropout=do_lstm_4,  # Dropout rate for recurrent units
                                      # kernel_regularizer=l2(1e-4),
                                      name='LSTM_end_2')
        
        self.Fully_Connected_end1 = layers.Dense(16,
                                                 activation=act16,
                                                 # kernel_regularizer=l2(1e-4),
                                                 name='FC_all_end0')
        self.Fully_Connected_end2 = layers.Dense(8,
                                                activation=None,
                                                name='FC_all_end')
        
        self.DualQuaternionNormalization = DualQuaternionNormalization()
        
        
        

    @tf.function
    def call(self, geo_lidar_input, training: bool = True):
        geo_inp, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        x_Resnet = self.create_res_net(lidar_inp)

        # FC1 = self.Dropout_li0(self.Fully_Connected_li0(x12))
        FC2 = self.Dropout_li1(self.Fully_Connected_li1(x_Resnet), training=training)
        FC3 = self.Dropout_li2(self.Fully_Connected_li2(FC2), training=training)
        FC4 = self.Dropout_li3(self.Fully_Connected_li3(FC3), training=training)
        FC5 = self.Dropout_li4(self.Fully_Connected_li4(FC4), training=training)


        combined = self.RNN_LSTM_1(FC5)
        combined = self.RNN_LSTM_2(combined)
        out0 = self.Fully_Connected_end1(combined, training=training)
        outp = self.Fully_Connected_end2(out0, training=training)
        outp = self.DualQuaternionNormalization(outp)

        return outp
    

class LiD_s2e_nowe_two(Model):

    def __init__(self, 
                 # _____________________________________________________resnet
                 # num_filters=64, filters1=64, 
                 filters2=128, filters3=256, filters4=512,
                 dense_last=1000, act_last='relu',
                 # _____________________________________________________Lidar
                 act_li='relu', dropout4=0.2,
                 d1_li=512, d2_li=256, d3_li=128, d4_li=64,
                 d_lstm_li1=64, 
                 # d_lstm_li2=32, d_lstm_li3=16,
                 act_lstm1='tanh', act_lstm2="sigmoid", act16='tanh',
                 do_lstm_1=0.2, do_lstm_2=0.2, do_lstm_3=0.2, do_lstm_4=0.2,
                 # _____________________________________________________
                 training=True, 
                 **kwargs):
        super().__init__(**kwargs)
                       
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(training=True, 
                                                                           num_filters=64,filters1=64, #Don't touch
                                                                           filters2=filters2, filters3=filters3, filters4=filters4, 
                                                                           dense_last=dense_last, act_last=act_last))   
        
        self.Fully_Connected_li1 = layers.TimeDistributed(layers.Dense(d1_li, activation=act_li, name='FC_lidar_1')) #1
        self.Fully_Connected_li2 = layers.TimeDistributed(layers.Dense(d2_li, activation=act_li, name='FC_lidar_2')) #2
        self.Fully_Connected_li3 = layers.TimeDistributed(layers.Dense(d3_li, activation=act_li, name='FC_lidar_3')) #3
        self.Fully_Connected_li4 = layers.TimeDistributed(layers.Dense(d4_li, activation=act_li, name='FC_lidar_4'))
        
        self.Dropout_li1 = layers.TimeDistributed(layers.Dropout(dropout4, name='dropout_lidar_1'))
        self.Dropout_li2 = layers.TimeDistributed(layers.Dropout(dropout4, name='dropout_lidar_2'))
        self.Dropout_li3 = layers.TimeDistributed(layers.Dropout(dropout4, name='dropout_lidar_3'))
        self.Dropout_li4 = layers.TimeDistributed(layers.Dropout(dropout4, name='dropout_lidar_4'))

        
        self.RNN_LSTM_1 = layers.LSTM(d_lstm_li1,
                                      activation=act_lstm1,
                                      recurrent_activation=act_lstm2,
                                      return_sequences=True,
                                      unroll=False,
                                      use_bias=True,
                                      dropout=do_lstm_1,  # Dropout rate for input units
                                      recurrent_dropout=do_lstm_2,  # Dropout rate for recurrent units
                                      # kernel_regularizer=l2(1e-4),
                                      name='LSTM_end_1')
        
        self.RNN_LSTM_2 = layers.LSTM(32,
                                      activation=act_lstm1,
                                      recurrent_activation=act_lstm2,
                                      unroll=False,
                                      use_bias=True,
                                      dropout=do_lstm_3,  # Dropout rate for input units
                                      recurrent_dropout=do_lstm_4,  # Dropout rate for recurrent units
                                      # kernel_regularizer=l2(1e-4),
                                      name='LSTM_end_2')
        
        self.Fully_Connected_end1 = layers.Dense(16,
                                                 activation=act16,
                                                 # kernel_regularizer=l2(1e-4),
                                                 name='FC_all_end0')
        self.Fully_Connected_end2 = layers.Dense(8,
                                                activation=None,
                                                name='FC_all_end')
        
        self.DualQuaternionNormalization = DualQuaternionNormalization()
        
        

        
    @tf.function
    def call(self, geo_lidar_input, training: bool = False):
        geo_inp, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        x_Resnet = self.create_res_net(lidar_inp)

        # FC1 = self.Dropout_li0(self.Fully_Connected_li0(x12))
        FC2 = self.Dropout_li1(self.Fully_Connected_li1(x_Resnet), training=training)
        FC3 = self.Dropout_li2(self.Fully_Connected_li2(FC2), training=training)
        FC4 = self.Dropout_li3(self.Fully_Connected_li3(FC3), training=training)
        FC5 = self.Dropout_li4(self.Fully_Connected_li4(FC4), training=training)


        combined = self.RNN_LSTM_1(FC5)
        combined = self.RNN_LSTM_2(combined)
        out0 = self.Fully_Connected_end1(combined, training=training)
        outp = self.Fully_Connected_end2(out0, training=training)
        outp = self.DualQuaternionNormalization(outp)

        
        self.layer_grad0 = tf.gradients(outp[:, 0], combined)
        self.layer_grad1 = tf.gradients(outp[:, 1], combined)
        self.layer_grad2 = tf.gradients(outp[:, 2], combined)
        self.layer_grad3 = tf.gradients(outp[:, 3], combined)
        self.layer_grad4 = tf.gradients(outp[:, 4], combined)
        self.layer_grad5 = tf.gradients(outp[:, 5], combined)
        self.layer_grad6 = tf.gradients(outp[:, 6], combined)
        self.layer_grad7 = tf.gradients(outp[:, 7], combined)

        return outp, combined, combined, \
            self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3, \
            self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7

# _________________________
# Lidar Only _Attention Weighitng
# _________________________    
class LiD_s2e_att(Model):
    def __init__(self,
                 act_li='relu', act_lstm='tanh',
                 d1_li=512, d2_li=256, d3_li=128, d4_li=64,
                 do_lstm_1=0.2, do_lstm_2=0.2, do_lstm_3=0.2, do_lstm_4=0.2,
                 **kwargs):
        super().__init__(**kwargs)

        # ResNet-like encoder wrapped in TimeDistributed
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(training=True))

        # Fully connected layers with dropout
        self.Fully_Connected_li1 = layers.TimeDistributed(layers.Dense(d1_li, activation=act_li))
        self.Fully_Connected_li2 = layers.TimeDistributed(layers.Dense(d2_li, activation=act_li))
        self.Fully_Connected_li3 = layers.TimeDistributed(layers.Dense(d3_li, activation=act_li))
        self.Fully_Connected_li4 = layers.TimeDistributed(layers.Dense(d4_li, activation=act_li))

        self.Dropout_li1 = layers.TimeDistributed(layers.Dropout(0.2))
        self.Dropout_li2 = layers.TimeDistributed(layers.Dropout(0.2))
        self.Dropout_li3 = layers.TimeDistributed(layers.Dropout(0.2))
        self.Dropout_li4 = layers.TimeDistributed(layers.Dropout(0.2))

        # Multi-head attention
        self.Attention_li = layers.MultiHeadAttention(num_heads=4, key_dim=64)

        # LSTM layers
        self.RNN_LSTM_1 = layers.LSTM(64, activation=act_lstm, recurrent_activation="sigmoid",
                                      return_sequences=True, dropout=do_lstm_1,
                                      recurrent_dropout=do_lstm_2, name='LSTM_end_1')
        self.RNN_LSTM_2 = layers.LSTM(32, activation=act_lstm, recurrent_activation="sigmoid",
                                      dropout=do_lstm_3, recurrent_dropout=do_lstm_4,
                                      return_sequences=False, name='LSTM_end_2')

        # Output layer
        self.Fully_Connected_end = layers.Dense(8, activation=None, name='FC_all_end')
        self.DualQuaternionNormalization = DualQuaternionNormalization()

    @tf.function
    def call(self, geo_lidar_input, training: bool = True):
        _, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        # Extract features from lidar using ResNet
        x = self.create_res_net(lidar_inp)

        # Fully connected + Dropout
        x = self.Dropout_li1(self.Fully_Connected_li1(x), training=training)
        x = self.Dropout_li2(self.Fully_Connected_li2(x), training=training)
        x = self.Dropout_li3(self.Fully_Connected_li3(x), training=training)
        x = self.Dropout_li4(self.Fully_Connected_li4(x), training=training)

        # Apply attention over time dimension
        x = self.Attention_li(query=x, value=x, key=x, training=training)

        # LSTM layers
        x = self.RNN_LSTM_1(x, training=training)
        x = self.RNN_LSTM_2(x, training=training)

        # Final dense + normalization
        x = self.Fully_Connected_end(x, training=training)
        x = self.DualQuaternionNormalization(x)

        return x
    

class LiD_s2e_att_two(Model):
    def __init__(self,
                 act_li='relu', act_lstm='tanh',
                 d1_li=512, d2_li=256, d3_li=128, d4_li=64,
                 do_lstm_1=0.2, do_lstm_2=0.2, do_lstm_3=0.2, do_lstm_4=0.2,
                 **kwargs):
        super().__init__(**kwargs)

        # ResNet-like encoder wrapped in TimeDistributed
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(training=True))

        # Fully connected layers with dropout
        self.Fully_Connected_li1 = layers.TimeDistributed(layers.Dense(d1_li, activation=act_li))
        self.Fully_Connected_li2 = layers.TimeDistributed(layers.Dense(d2_li, activation=act_li))
        self.Fully_Connected_li3 = layers.TimeDistributed(layers.Dense(d3_li, activation=act_li))
        self.Fully_Connected_li4 = layers.TimeDistributed(layers.Dense(d4_li, activation=act_li))

        self.Dropout_li1 = layers.TimeDistributed(layers.Dropout(0.2))
        self.Dropout_li2 = layers.TimeDistributed(layers.Dropout(0.2))
        self.Dropout_li3 = layers.TimeDistributed(layers.Dropout(0.2))
        self.Dropout_li4 = layers.TimeDistributed(layers.Dropout(0.2))

        # Multi-head attention
        self.Attention_li = layers.MultiHeadAttention(num_heads=4, key_dim=64)

        # LSTM layers
        self.RNN_LSTM_1 = layers.LSTM(64, activation=act_lstm, recurrent_activation="sigmoid",
                                      return_sequences=True, dropout=do_lstm_1,
                                      recurrent_dropout=do_lstm_2, name='LSTM_end_1')
        self.RNN_LSTM_2 = layers.LSTM(32, activation=act_lstm, recurrent_activation="sigmoid",
                                      dropout=do_lstm_3, recurrent_dropout=do_lstm_4,
                                      return_sequences=False, name='LSTM_end_2')

        # Output layer
        self.Fully_Connected_end = layers.Dense(8, activation=None, name='FC_all_end')
        self.DualQuaternionNormalization = DualQuaternionNormalization()

    @tf.function
    def call(self, geo_lidar_input, training: bool = True):
        _, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        # Extract features from lidar using ResNet
        x = self.create_res_net(lidar_inp)

        # Fully connected + Dropout
        x = self.Dropout_li1(self.Fully_Connected_li1(x), training=training)
        x = self.Dropout_li2(self.Fully_Connected_li2(x), training=training)
        x = self.Dropout_li3(self.Fully_Connected_li3(x), training=training)
        x = self.Dropout_li4(self.Fully_Connected_li4(x), training=training)

        # Apply attention over time dimension
        x = self.Attention_li(query=x, value=x, key=x, training=training)

        # LSTM layers
        x = self.RNN_LSTM_1(x, training=training)
        x = self.RNN_LSTM_2(x, training=training)

        outp = self.Fully_Connected_end(x, training=training)
        outp = self.DualQuaternionNormalization(outp)

        # Compute gradients per output channel
        grads = [tf.gradients(outp[:, i], outp)[0] for i in range(8)]

        return (outp, x, x, *grads)
    
# _________________________
# Lidar Only _INAF Weighitng
# _________________________ 
class LiD_s2e_INAF(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.Attention_li = layers.MultiHeadAttention(num_heads=4, key_dim=32)


        self.Fully_Connected_end1 = layers.Dense(16,
                                        activation=None,
                                        # use_bias=True,
                                        # bias_regularizer=regularizers.L2(1e-4),
                                        name='FC_all_end0')
        self.Fully_Connected_end2 = layers.Dense(8,
                                                activation=None,
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end')

    @tf.function
    def call(self, geo_lidar_input, training: bool = True):
        INAF_P, INAF_W = geo_lidar_input['Param'], geo_lidar_input['Weight']
        
        inputs_expanded = tf.expand_dims(INAF_P, axis=1)

        # Apply pre-made MultiHeadAttention
        attention_output = self.Attention_li(query=inputs_expanded, value=inputs_expanded, attention_mask=None)

        # Squeeze and reshape to remove the added dimension
        attention_output = tf.squeeze(attention_output, axis=1)

        # Apply weights to attention output
        weighted_attention_output = attention_output * INAF_W

        # Expand dimensions along the second axis
        # INAF_P_expanded = tf.expand_dims(INAF_P, axis=1)
        # INAF_W_expanded = tf.expand_dims(INAF_W, axis=1)

        # Apply self-attention (Q=INAF_P, K=INAF_P, V=INAF_W)
        # FC5_att = self.Attention_li(INAF_W_expanded, INAF_P_expanded, INAF_P_expanded)
        
        out0 = self.Fully_Connected_end1(weighted_attention_output, training=training)
        outp = self.Fully_Connected_end2(out0, training=training)

        return outp
    

class LiD_s2e_INAF_two(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.Attention_li = layers.MultiHeadAttention(num_heads=4, key_dim=32)


        self.Fully_Connected_end1 = layers.Dense(16,
                                        activation=None,
                                        # use_bias=True,
                                        # bias_regularizer=regularizers.L2(1e-4),
                                        name='FC_all_end0')
        self.Fully_Connected_end2 = layers.Dense(8,
                                                activation=None,
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end')
    @tf.function
    def call(self, geo_lidar_input, training: bool = False):
        INAF_P, INAF_W = geo_lidar_input['Param'], geo_lidar_input['Weight']
        
        inputs_expanded = tf.expand_dims(INAF_P, axis=1)

        # Apply pre-made MultiHeadAttention
        attention_output = self.Attention_li(query=inputs_expanded, value=inputs_expanded, attention_mask=None)
        attention_output = tf.squeeze(attention_output, axis=1)
        weighted_attention_output = attention_output * INAF_W
        
        out0 = self.Fully_Connected_end1(weighted_attention_output, training=training)
        outp = self.Fully_Connected_end2(out0, training=training)
        
        return outp
# ______________________________________________________________________________________________________________________
# Combined Layers
# ______________________________________________________________________________________________________________________
class Combined_s2e_nowe(Model):
    
    def __init__(self, 
                 #_____________________________________________________geo branch
                 act_geo1='relu', act_geo2='relu', act_geo3='relu', act_geo4='relu',                 
                 dropout_rate_1_geo=0.3, dropout_rate_2_geo = 0.2, 
                 e1_geo=128, e2_geo=256, e3_geo=512, 
                 d3_geo=256, d2_geo=192,
                 act_geo_lstm1='relu', act_geo_lstm2='relu', 
                 do_lstm_geo_1=0.2, do_lstm_geo_2=0.2, do_lstm_geo_3=0.2, do_lstm_geo_4=0.2,
                 #_____________________________________________________lidar branch
                 # ____resnet
                 filters2=64, filters3=128, filters4=256,
                 dense_last=512, act_last='relu',
                 # ____
                 act1_li='relu', act2_li='relu', act3_li='relu', act4_li='relu',
                 dropout_rate_li=0.2,
                 d1_li=256, d2_li=128, d3_li=64, d4_li=32,
                 d1_lstm_li=32,
                 act_li_lstm1='tanh', act_li_lstm2='tanh', act_li_lstm3='tanh', act_li_lstm4='tanh', 
                 do_lstm_li_1=0.1, do_lstm_li_2=0.1, do_lstm_li_3=0.1, do_lstm_li_4=0.1,
                 #_____________________________________________________fuse
                 d1_fuse=64, act_fuse='relu', dropout_rate_fuse=0.1,
                 d3_end=32, d2_end=16, 
                 act_end_1='tanh', act_end_2='tanh',
                 **kwargs):
        super().__init__(**kwargs)
        
        
        # ______________________________________________________________________________________________________________
        # Geo Branch
        # ______________________________________________________________________________________________________________
        self.BN_geo_input = layers.TimeDistributed(layers.BatchNormalization(name='BN_geo_inp'))
        
        self.Fully_Connected_en_1_geo = layers.TimeDistributed(layers.Dense(e1_geo, activation=act_geo1, name='FC_geo_en_1'))
        self.Fully_Connected_en_2_geo = layers.TimeDistributed(layers.Dense(e2_geo, activation=act_geo2, name='FC_geo_en_2'))        
        self.Fully_Connected_en_3_geo = layers.TimeDistributed(layers.Dense(e3_geo, activation=act_geo3, name='FC_geo_en_3'))
        self.Dropout_geo_1 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=dropout_rate_1_geo, name='drop_geo_1'))
        # ______________________________________________________________________________________________________________
        self.Fully_Connected_de_2_geo = layers.TimeDistributed(layers.Dense(d3_geo, activation=act_geo4, name='FC_geo_de_1'))
        self.Dropout_geo_2 = layers.TimeDistributed(layers.Dropout(rate=dropout_rate_2_geo, name='drop_geo_2'))
        # ______________________________________________________________________________________________________________
        self.RNN_LSTM_geo_1 = layers.LSTM(d2_geo,
                                      activation=act_geo_lstm1,
                                      recurrent_activation=act_geo_lstm2,
                                      return_sequences=True,
                                      unroll=False,
                                      use_bias=True,
                                      dropout=do_lstm_geo_1,  # Dropout rate for input units
                                      recurrent_dropout=do_lstm_geo_2,  # Dropout rate for recurrent units
                                      # kernel_regularizer=l2(1e-4),
                                      name='LSTM_geo_1')
        
        self.RNN_LSTM_geo_2 = layers.LSTM(32,
                                      activation=act_geo_lstm1,
                                      recurrent_activation=act_geo_lstm2,
                                      unroll=False,
                                      use_bias=True,
                                      dropout=do_lstm_geo_3,  # Dropout rate for input units
                                      recurrent_dropout=do_lstm_geo_4,  # Dropout rate for recurrent units
                                      # kernel_regularizer=l2(1e-4),
                                      name='LSTM_geo_2')
        
        # ______________________________________________________________________________________________________________
        # Lidar Branch
        # ______________________________________________________________________________________________________________
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(num_filters=64,filters1=64, #Don't touch
                                                                           filters2=filters2, filters3=filters3, filters4=filters4, 
                                                                           dense_last=dense_last, act_last=act_last))   

        
        self.Fully_Connected_li_1 = layers.TimeDistributed(layers.Dense(d1_li, activation=act1_li, name='FC_lidar_1')) #1
        self.Fully_Connected_li_2 = layers.TimeDistributed(layers.Dense(d2_li, activation=act2_li, name='FC_lidar_2')) #2
        self.Fully_Connected_li_3 = layers.TimeDistributed(layers.Dense(d3_li, activation=act3_li, name='FC_lidar_3')) #3
        self.Fully_Connected_li_4 = layers.TimeDistributed(layers.Dense(d4_li, activation=act4_li, name='FC_lidar_4')) #4
        
        self.Dropout_li_1 = layers.TimeDistributed(layers.Dropout(dropout_rate_li, name='dropout_lidar_1'))
        self.Dropout_li_2 = layers.TimeDistributed(layers.Dropout(dropout_rate_li, name='dropout_lidar_2'))
        self.Dropout_li_3 = layers.TimeDistributed(layers.Dropout(dropout_rate_li, name='dropout_lidar_3'))
        self.Dropout_li_4 = layers.TimeDistributed(layers.Dropout(dropout_rate_li, name='dropout_lidar_4'))

        
        self.RNN_LSTM_li_1 = layers.LSTM(d1_lstm_li,
                                         activation=act_li_lstm1,
                                         recurrent_activation=act_li_lstm2,
                                         return_sequences=True,
                                         unroll=False,
                                         use_bias=True,                                      
                                         dropout=do_lstm_li_1,  # Dropout rate for input units
                                         recurrent_dropout=do_lstm_li_2,  # Dropout rate for recurrent units
                                         # kernel_regularizer=l2(1e-4),
                                         name='LSTM_lidar_1')
        
        self.RNN_LSTM_li_2 = layers.LSTM(32,
                                         activation=act_li_lstm3,
                                         recurrent_activation=act_li_lstm4,
                                         unroll=False,
                                         use_bias=True,                                      
                                         dropout=do_lstm_li_3,  # Dropout rate for input units
                                         recurrent_dropout=do_lstm_li_4,  # Dropout rate for recurrent units
                                         # kernel_regularizer=l2(1e-4),
                                         name='LSTM_lidar_2')
        
        
        # ______________________________________________________________________________________________________________
        # Last module
        # ______________________________________________________________________________________________________________
        self.BN_Geo = layers.BatchNormalization(name='BN_geo_branch')
        self.BN_Li = layers.BatchNormalization(name='BN_Li_branch')

        self.dropout_fusion = layers.Dropout(dropout_rate_fuse, name='fusion_dropout')
        self.concatenate_end = layers.Concatenate(axis=1)
        
        # Define the layers for feature fusion
        self.dense_fusion = layers.Dense(d1_fuse, activation=act_fuse, name='fusion_dense')
        # self.dropout_fusion = layers.Dropout(dropout_rate_fuse, name='fusion_dropout')

        
        self.Fully_Connected_end_0 = layers.Dense(d3_end,
                                                activation=act_end_1,
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end0')
        self.Fully_Connected_end_1 = layers.Dense(d2_end,
                                                activation=act_end_2,
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end1')
        self.Fully_Connected_end_2 = layers.Dense(8,
                                                activation=None,
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end2')
        self.DualQuaternionNormalization = DualQuaternionNormalization()

    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = True):
        geo_inp, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        # ______________________________________________________________________________________________________________
        # Geo Branch
        # ______________________________________________________________________________________________________________        
        geo_inp1 = self.BN_geo_input(geo_inp, training=training)

        x1_geo = self.Fully_Connected_en_1_geo(geo_inp1, training=training)
        x2_geo = self.Fully_Connected_en_2_geo(x1_geo, training=training)        
        x2_geo = self.Fully_Connected_en_3_geo(x2_geo, training=training)

        x2_geo = self.Dropout_geo_1(x2_geo, training=training)
        # ______________________________________________________________________________________________________________
        x_mid = self.Fully_Connected_de_2_geo(x2_geo, training=training)
        x_mid = self.Dropout_geo_2(x_mid, training=training)
        # ______________________________________________________________________________________________________________
        geo_outp = self.RNN_LSTM_geo_1(x_mid, training=training)
        geo_branch = self.RNN_LSTM_geo_2(geo_outp, training=training)
        
        # ______________________________________________________________________________________________________________
        # Lidar Branch
        # ______________________________________________________________________________________________________________
        x_Resnet = self.create_res_net(lidar_inp, training=training)
        
        # FC1 = self.Dropout_li0(self.Fully_Connected_li0(x12))
        x1_li = self.Dropout_li_1(self.Fully_Connected_li_1(x_Resnet), training=training)
        x2_li = self.Dropout_li_2(self.Fully_Connected_li_2(x1_li), training=training)
        x3_li = self.Dropout_li_3(self.Fully_Connected_li_3(x2_li), training=training)
        x4_li = self.Dropout_li_4(self.Fully_Connected_li_4(x3_li), training=training)

        x5_li = self.RNN_LSTM_li_1(x4_li, training=training)
        lidar_branch = self.RNN_LSTM_li_2(x5_li, training=training)
        
        # ______________________________________________________________________________________________________________
        # Last module
        # ______________________________________________________________________________________________________________
        geo_branch_N = self.BN_Geo(geo_branch, training=training)
        lidar_branch_N = self.BN_Li(lidar_branch, training=training)
        
        merged_features = self.concatenate_end([geo_branch_N, lidar_branch_N])
        merged_features = self.dropout_fusion(merged_features, training=training)
        combined = self.dense_fusion(merged_features, training=training)      
        # combined = self.dropout_fusion(merged_features, training=training)
        
        out0 = self.Fully_Connected_end_0(combined, training=training)
        out1 = self.Fully_Connected_end_1(out0, training=training)
        outp = self.Fully_Connected_end_2(out1, training=training)
        outp = self.DualQuaternionNormalization(outp)

        return outp


class Combined_s2e_nowe_two(Model):

    def __init__(self, 
                 #_____________________________________________________geo branch
                 act_geo1='relu', act_geo2='relu', act_geo3='relu', act_geo4='relu',                 
                 dropout_rate_1_geo=0.3, dropout_rate_2_geo = 0.2, 
                 e1_geo=128, e2_geo=256, e3_geo=512, 
                 d3_geo=256, d2_geo=192,
                 act_geo_lstm1='relu', act_geo_lstm2='relu', 
                 do_lstm_geo_1=0.2, do_lstm_geo_2=0.2, do_lstm_geo_3=0.2, do_lstm_geo_4=0.2,
                 #_____________________________________________________lidar branch
                 # ____resnet
                 filters2=64, filters3=128, filters4=256,
                 dense_last=512, act_last='relu',
                 # ____
                 act1_li='relu', act2_li='relu', act3_li='relu', act4_li='relu',
                 dropout_rate_li=0.2,
                 d1_li=256, d2_li=128, d3_li=64, d4_li=32,
                 d1_lstm_li=32,
                 act_li_lstm1='tanh', act_li_lstm2='tanh', act_li_lstm3='tanh', act_li_lstm4='tanh', 
                 do_lstm_li_1=0.1, do_lstm_li_2=0.1, do_lstm_li_3=0.1, do_lstm_li_4=0.1,
                 #_____________________________________________________fuse
                 d1_fuse=64, act_fuse='relu', dropout_rate_fuse=0.1,
                 d3_end=32, d2_end=16, 
                 act_end_1='tanh', act_end_2='tanh',
                 **kwargs):
        super().__init__(**kwargs)
        
        
        # ______________________________________________________________________________________________________________
        # Geo Branch
        # ______________________________________________________________________________________________________________
        self.BN_geo_input = layers.TimeDistributed(layers.BatchNormalization(name='BN_geo_inp'))
        
        self.Fully_Connected_en_1_geo = layers.TimeDistributed(layers.Dense(e1_geo, activation=act_geo1, name='FC_geo_en_1', kernel_regularizer=l2(1e-4)))
        self.Fully_Connected_en_2_geo = layers.TimeDistributed(layers.Dense(e2_geo, activation=act_geo2, name='FC_geo_en_2', kernel_regularizer=l2(1e-4)))        
        self.Fully_Connected_en_3_geo = layers.TimeDistributed(layers.Dense(e3_geo, activation=act_geo3, name='FC_geo_en_3'))
        self.Dropout_geo_1 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=dropout_rate_1_geo, name='drop_geo_1'))
        # ______________________________________________________________________________________________________________
        self.Fully_Connected_de_2_geo = layers.TimeDistributed(layers.Dense(d3_geo, activation=act_geo4, name='FC_geo_de_1'))
        self.Dropout_geo_2 = layers.TimeDistributed(layers.Dropout(rate=dropout_rate_2_geo, name='drop_geo_2'))
        # ______________________________________________________________________________________________________________
        self.RNN_LSTM_geo_1 = layers.LSTM(d2_geo,
                                      activation=act_geo_lstm1,
                                      recurrent_activation=act_geo_lstm2,
                                      return_sequences=True,
                                      unroll=False,
                                      use_bias=True,
                                      dropout=do_lstm_geo_1,  # Dropout rate for input units
                                      recurrent_dropout=do_lstm_geo_2,  # Dropout rate for recurrent units
                                      kernel_regularizer=l2(1e-4),
                                      name='LSTM_geo_1')
        
        self.RNN_LSTM_geo_2 = layers.LSTM(32,
                                      activation=act_geo_lstm1,
                                      recurrent_activation=act_geo_lstm2,
                                      unroll=False,
                                      use_bias=True,
                                      dropout=do_lstm_geo_3,  # Dropout rate for input units
                                      recurrent_dropout=do_lstm_geo_4,  # Dropout rate for recurrent units
                                      kernel_regularizer=l2(1e-4),
                                      name='LSTM_geo_2')
        
        # ______________________________________________________________________________________________________________
        # Lidar Branch
        # ______________________________________________________________________________________________________________
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(num_filters=64,filters1=64, #Don't touch
                                                                           filters2=filters2, filters3=filters3, filters4=filters4, 
                                                                           dense_last=dense_last, act_last=act_last))   
        
        self.Fully_Connected_li_1 = layers.TimeDistributed(layers.Dense(d1_li, activation=act1_li, name='FC_lidar_1')) #1
        self.Fully_Connected_li_2 = layers.TimeDistributed(layers.Dense(d2_li, activation=act2_li, name='FC_lidar_2')) #2
        self.Fully_Connected_li_3 = layers.TimeDistributed(layers.Dense(d3_li, activation=act3_li, name='FC_lidar_3')) #3
        self.Fully_Connected_li_4 = layers.TimeDistributed(layers.Dense(d4_li, activation=act4_li, name='FC_lidar_4')) #4
        
        self.Dropout_li_1 = layers.TimeDistributed(layers.Dropout(dropout_rate_li, name='dropout_lidar_1'))
        self.Dropout_li_2 = layers.TimeDistributed(layers.Dropout(dropout_rate_li, name='dropout_lidar_2'))
        self.Dropout_li_3 = layers.TimeDistributed(layers.Dropout(dropout_rate_li, name='dropout_lidar_3'))
        self.Dropout_li_4 = layers.TimeDistributed(layers.Dropout(dropout_rate_li, name='dropout_lidar_4'))

        
        self.RNN_LSTM_li_1 = layers.LSTM(d1_lstm_li,
                                         activation=act_li_lstm1,
                                         recurrent_activation=act_li_lstm2,
                                         return_sequences=True,
                                         unroll=False,
                                         use_bias=True,                                      
                                         dropout=do_lstm_li_1,  # Dropout rate for input units
                                         recurrent_dropout=do_lstm_li_2,  # Dropout rate for recurrent units
                                         kernel_regularizer=l2(1e-4),
                                         name='LSTM_lidar_1')
        
        self.RNN_LSTM_li_2 = layers.LSTM(32,
                                         activation=act_li_lstm3,
                                         recurrent_activation=act_li_lstm4,
                                         unroll=False,
                                         use_bias=True,                                      
                                         dropout=do_lstm_li_3,  # Dropout rate for input units
                                         recurrent_dropout=do_lstm_li_4,  # Dropout rate for recurrent units
                                         kernel_regularizer=l2(1e-4),
                                         name='LSTM_lidar_2')
        
        
        # ______________________________________________________________________________________________________________
        # Last module
        # ______________________________________________________________________________________________________________
        self.BN_Geo = layers.BatchNormalization(name='BN_geo_branch')
        self.BN_Li = layers.BatchNormalization(name='BN_Li_branch')

        self.dropout_fusion = layers.Dropout(dropout_rate_fuse, name='fusion_dropout')
        self.concatenate_end = layers.Concatenate(axis=1)
        
        # Define the layers for feature fusion
        self.dense_fusion = layers.Dense(d1_fuse, activation=act_fuse, name='fusion_dense')
        # self.dropout_fusion = layers.Dropout(dropout_rate_fuse, name='fusion_dropout')

        
        self.Fully_Connected_end_0 = layers.Dense(d3_end,
                                                activation=act_end_1,
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end0')
        self.Fully_Connected_end_1 = layers.Dense(d2_end,
                                                activation=act_end_2,
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end1')
        self.Fully_Connected_end_2 = layers.Dense(8,
                                                activation=None,
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end2')
        self.DualQuaternionNormalization = DualQuaternionNormalization()


    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = False):
        geo_inp, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        # ______________________________________________________________________________________________________________
        # Geo Branch
        # ______________________________________________________________________________________________________________        
        geo_inp1 = self.BN_geo_input(geo_inp, training=training)

        x1_geo = self.Fully_Connected_en_1_geo(geo_inp1, training=training)
        x2_geo = self.Fully_Connected_en_2_geo(x1_geo, training=training)        
        x2_geo = self.Fully_Connected_en_3_geo(x2_geo, training=training)

        x2_geo = self.Dropout_geo_1(x2_geo, training=training)
        # ______________________________________________________________________________________________________________
        x_mid = self.Fully_Connected_de_2_geo(x2_geo, training=training)
        x_mid = self.Dropout_geo_2(x_mid, training=training)
        # ______________________________________________________________________________________________________________
        geo_outp = self.RNN_LSTM_geo_1(x_mid, training=training)
        geo_branch = self.RNN_LSTM_geo_2(geo_outp, training=training)
        
        # ______________________________________________________________________________________________________________
        # Lidar Branch
        # ______________________________________________________________________________________________________________
        x_Resnet = self.create_res_net(lidar_inp, training=training)
        
        # FC1 = self.Dropout_li0(self.Fully_Connected_li0(x12))
        x1_li = self.Dropout_li_1(self.Fully_Connected_li_1(x_Resnet), training=training)
        x2_li = self.Dropout_li_2(self.Fully_Connected_li_2(x1_li), training=training)
        x3_li = self.Dropout_li_3(self.Fully_Connected_li_3(x2_li), training=training)
        x4_li = self.Dropout_li_4(self.Fully_Connected_li_4(x3_li), training=training)

        x5_li = self.RNN_LSTM_li_1(x4_li, training=training)
        lidar_branch = self.RNN_LSTM_li_2(x5_li, training=training)
        
        # ______________________________________________________________________________________________________________
        # Last module
        # ______________________________________________________________________________________________________________
        geo_branch_N = self.BN_Geo(geo_branch, training=training)
        lidar_branch_N = self.BN_Li(lidar_branch, training=training)
        
        merged_features = self.concatenate_end([geo_branch_N, lidar_branch_N])
        merged_features = self.dropout_fusion(merged_features, training=training)
        combined = self.dense_fusion(merged_features, training=training)      
        # combined = self.dropout_fusion(merged_features, training=training)
        
        out0 = self.Fully_Connected_end_0(combined, training=training)
        out1 = self.Fully_Connected_end_1(out0, training=training)
        outp = self.Fully_Connected_end_2(out1, training=training)
        outp = self.DualQuaternionNormalization(outp)

        self.layer_grad0 = tf.gradients(outp[:, 0], combined)
        self.layer_grad1 = tf.gradients(outp[:, 1], combined)
        self.layer_grad2 = tf.gradients(outp[:, 2], combined)
        self.layer_grad3 = tf.gradients(outp[:, 3], combined)
        self.layer_grad4 = tf.gradients(outp[:, 4], combined)
        self.layer_grad5 = tf.gradients(outp[:, 5], combined)
        self.layer_grad6 = tf.gradients(outp[:, 6], combined)
        self.layer_grad7 = tf.gradients(outp[:, 7], combined)

        return outp, geo_branch_N, lidar_branch_N, \
            self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3, \
            self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7
    
    
class Combined_s2e_nowe_simple(Model):
    def __init__(self, 
                 # GEO Branch
                 act_geo1='relu', act_geo2='relu', act_geo4='relu',
                 dropout_rate_1_geo=0.3,
                 e1_geo=128, e2_geo=256,
                 d3_geo=256,
                 act_geo_lstm1='relu', act_geo_lstm2='relu',
                 do_lstm_geo_3=0.2, do_lstm_geo_4=0.2,

                 # LIDAR Branch
                 filters2=64, filters3=128, filters4=256,
                 dense_last=512, act_last='relu',
                 act1_li='relu', act2_li='relu',
                 dropout_rate_li=0.2,
                 d1_li=256, d2_li=128,
                 act_li_lstm3='tanh', act_li_lstm4='tanh',
                 do_lstm_li_3=0.1, do_lstm_li_4=0.1,

                 # FUSION
                 d1_fuse=64, act_fuse='relu',
                 dropout_rate_fuse=0.1,
                 d3_end=32, d2_end=16,
                 act_end_1='tanh', act_end_2='tanh',
                 **kwargs):
        super().__init__(**kwargs)

        # ----------------------------- Geo Branch -----------------------------
        self.BN_geo_input = layers.TimeDistributed(layers.BatchNormalization(name='BN_geo_inp'))
        self.Fully_Connected_en_1_geo = layers.TimeDistributed(layers.Dense(e1_geo, activation=act_geo1, name='FC_geo_en_1'))
        self.Fully_Connected_en_2_geo = layers.TimeDistributed(layers.Dense(e2_geo, activation=act_geo2, name='FC_geo_en_2'))
        self.Dropout_geo_1 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=dropout_rate_1_geo, name='drop_geo_1'))
        self.Fully_Connected_de_2_geo = layers.TimeDistributed(layers.Dense(d3_geo, activation=act_geo4, name='FC_geo_de_1'))

        self.RNN_LSTM_geo_2 = layers.LSTM(32,
                                          activation=act_geo_lstm1,
                                          recurrent_activation=act_geo_lstm2,
                                          unroll=False,
                                          use_bias=True,
                                          dropout=do_lstm_geo_3,
                                          recurrent_dropout=do_lstm_geo_4,
                                          name='LSTM_geo_2')

        # ----------------------------- Lidar Branch -----------------------------
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(num_filters=64, filters1=64,
                                                                           filters2=filters2, filters3=filters3,
                                                                           filters4=filters4,
                                                                           dense_last=dense_last, act_last=act_last))

        self.Fully_Connected_li_1 = layers.TimeDistributed(layers.Dense(d1_li, activation=act1_li, name='FC_lidar_1'))
        self.Fully_Connected_li_2 = layers.TimeDistributed(layers.Dense(d2_li, activation=act2_li, name='FC_lidar_2'))
        self.Dropout_li_1 = layers.TimeDistributed(layers.Dropout(dropout_rate_li, name='dropout_lidar_1'))

        self.RNN_LSTM_li_2 = layers.LSTM(32,
                                         activation=act_li_lstm3,
                                         recurrent_activation=act_li_lstm4,
                                         unroll=False,
                                         use_bias=True,
                                         dropout=do_lstm_li_3,
                                         recurrent_dropout=do_lstm_li_4,
                                         name='LSTM_lidar_2')

        # ----------------------------- Fusion & Output -----------------------------
        self.BN_Geo = layers.BatchNormalization(name='BN_geo_branch')
        self.BN_Li = layers.BatchNormalization(name='BN_Li_branch')

        self.concatenate_end = layers.Concatenate(axis=1)
        self.dropout_fusion = layers.Dropout(dropout_rate_fuse, name='fusion_dropout')
        self.dense_fusion = layers.Dense(d1_fuse, activation=act_fuse, name='fusion_dense')

        self.Fully_Connected_end_0 = layers.Dense(d3_end, activation=act_end_1, name='FC_all_end0')
        self.Fully_Connected_end_1 = layers.Dense(d2_end, activation=act_end_2, name='FC_all_end1')
        self.Fully_Connected_end_2 = layers.Dense(8, activation=None, name='FC_all_end2')

        self.DualQuaternionNormalization = DualQuaternionNormalization()

    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = True):
        geo_inp, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        # ----------------------------- Geo Branch -----------------------------
        geo_inp1 = self.BN_geo_input(geo_inp, training=training)
        x1_geo = self.Fully_Connected_en_1_geo(geo_inp1, training=training)
        x2_geo = self.Fully_Connected_en_2_geo(x1_geo, training=training)
        x2_geo = self.Dropout_geo_1(x2_geo, training=training)
        x_mid = self.Fully_Connected_de_2_geo(x2_geo, training=training)
        geo_branch = self.RNN_LSTM_geo_2(x_mid, training=training)

        # ----------------------------- Lidar Branch -----------------------------
        x_Resnet = self.create_res_net(lidar_inp, training=training)
        x1_li = self.Dropout_li_1(self.Fully_Connected_li_1(x_Resnet), training=training)
        x2_li = self.Fully_Connected_li_2(x1_li, training=training)
        lidar_branch = self.RNN_LSTM_li_2(x2_li, training=training)

        # ----------------------------- Fusion & Output -----------------------------
        geo_branch_N = self.BN_Geo(geo_branch, training=training)
        lidar_branch_N = self.BN_Li(lidar_branch, training=training)

        merged_features = self.concatenate_end([geo_branch_N, lidar_branch_N])
        merged_features = self.dropout_fusion(merged_features, training=training)
        combined = self.dense_fusion(merged_features, training=training)

        out0 = self.Fully_Connected_end_0(combined, training=training)
        out1 = self.Fully_Connected_end_1(out0, training=training)
        outp = self.Fully_Connected_end_2(out1, training=training)
        outp = self.DualQuaternionNormalization(outp)

        return outp
class Combined_s2e_nowe_two_simple(Model):
    def __init__(self, 
                 # GEO Branch
                 act_geo1='relu', act_geo2='relu', act_geo4='relu',
                 dropout_rate_1_geo=0.3,
                 e1_geo=128, e2_geo=256,
                 d3_geo=256,
                 act_geo_lstm1='relu', act_geo_lstm2='relu',
                 do_lstm_geo_3=0.2, do_lstm_geo_4=0.2,

                 # LIDAR Branch
                 filters2=64, filters3=128, filters4=256,
                 dense_last=512, act_last='relu',
                 act1_li='relu', act2_li='relu',
                 dropout_rate_li=0.2,
                 d1_li=256, d2_li=128,
                 act_li_lstm3='tanh', act_li_lstm4='tanh',
                 do_lstm_li_3=0.1, do_lstm_li_4=0.1,

                 # FUSION
                 d1_fuse=64, act_fuse='relu',
                 dropout_rate_fuse=0.1,
                 d3_end=32, d2_end=16,
                 act_end_1='tanh', act_end_2='tanh',
                 **kwargs):
        super().__init__(**kwargs)

        # ----------------------------- Geo Branch -----------------------------
        self.BN_geo_input = layers.TimeDistributed(layers.BatchNormalization(name='BN_geo_inp'))
        self.Fully_Connected_en_1_geo = layers.TimeDistributed(layers.Dense(e1_geo, activation=act_geo1, name='FC_geo_en_1'))
        self.Fully_Connected_en_2_geo = layers.TimeDistributed(layers.Dense(e2_geo, activation=act_geo2, name='FC_geo_en_2'))
        self.Dropout_geo_1 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=dropout_rate_1_geo, name='drop_geo_1'))
        self.Fully_Connected_de_2_geo = layers.TimeDistributed(layers.Dense(d3_geo, activation=act_geo4, name='FC_geo_de_1'))

        self.RNN_LSTM_geo_2 = layers.LSTM(32,
                                          activation=act_geo_lstm1,
                                          recurrent_activation=act_geo_lstm2,
                                          unroll=False,
                                          use_bias=True,
                                          dropout=do_lstm_geo_3,
                                          recurrent_dropout=do_lstm_geo_4,
                                          name='LSTM_geo_2')

        # ----------------------------- Lidar Branch -----------------------------
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(num_filters=64, filters1=64,
                                                                           filters2=filters2, filters3=filters3,
                                                                           filters4=filters4,
                                                                           dense_last=dense_last, act_last=act_last))

        self.Fully_Connected_li_1 = layers.TimeDistributed(layers.Dense(d1_li, activation=act1_li, name='FC_lidar_1'))
        self.Fully_Connected_li_2 = layers.TimeDistributed(layers.Dense(d2_li, activation=act2_li, name='FC_lidar_2'))
        self.Dropout_li_1 = layers.TimeDistributed(layers.Dropout(dropout_rate_li, name='dropout_lidar_1'))

        self.RNN_LSTM_li_2 = layers.LSTM(32,
                                         activation=act_li_lstm3,
                                         recurrent_activation=act_li_lstm4,
                                         unroll=False,
                                         use_bias=True,
                                         dropout=do_lstm_li_3,
                                         recurrent_dropout=do_lstm_li_4,
                                         name='LSTM_lidar_2')

        # ----------------------------- Fusion & Output -----------------------------
        self.BN_Geo = layers.BatchNormalization(name='BN_geo_branch')
        self.BN_Li = layers.BatchNormalization(name='BN_Li_branch')

        self.concatenate_end = layers.Concatenate(axis=1)
        self.dropout_fusion = layers.Dropout(dropout_rate_fuse, name='fusion_dropout')
        self.dense_fusion = layers.Dense(d1_fuse, activation=act_fuse, name='fusion_dense')

        self.Fully_Connected_end_0 = layers.Dense(d3_end, activation=act_end_1, name='FC_all_end0')
        self.Fully_Connected_end_1 = layers.Dense(d2_end, activation=act_end_2, name='FC_all_end1')
        self.Fully_Connected_end_2 = layers.Dense(8, activation=None, name='FC_all_end2')

        self.DualQuaternionNormalization = DualQuaternionNormalization()
    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = False):
        geo_inp, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        # ----------------------------- Geo Branch -----------------------------
        geo_inp1 = self.BN_geo_input(geo_inp, training=training)
        x1_geo = self.Fully_Connected_en_1_geo(geo_inp1, training=training)
        x2_geo = self.Fully_Connected_en_2_geo(x1_geo, training=training)
        x2_geo = self.Dropout_geo_1(x2_geo, training=training)
        x_mid = self.Fully_Connected_de_2_geo(x2_geo, training=training)
        geo_branch = self.RNN_LSTM_geo_2(x_mid, training=training)

        # ----------------------------- Lidar Branch -----------------------------
        x_Resnet = self.create_res_net(lidar_inp, training=training)
        x1_li = self.Dropout_li_1(self.Fully_Connected_li_1(x_Resnet), training=training)
        x2_li = self.Fully_Connected_li_2(x1_li, training=training)
        lidar_branch = self.RNN_LSTM_li_2(x2_li, training=training)

        # ----------------------------- Fusion & Output -----------------------------
        geo_branch_N = self.BN_Geo(geo_branch, training=training)
        lidar_branch_N = self.BN_Li(lidar_branch, training=training)

        merged_features = self.concatenate_end([geo_branch_N, lidar_branch_N])
        merged_features = self.dropout_fusion(merged_features, training=training)
        combined = self.dense_fusion(merged_features, training=training)

        out0 = self.Fully_Connected_end_0(combined, training=training)
        out1 = self.Fully_Connected_end_1(out0, training=training)
        outp = self.Fully_Connected_end_2(out1, training=training)
        outp = self.DualQuaternionNormalization(outp)

        # Compute gradients of each output component
        grads = [tf.gradients(outp[:, i], combined) for i in range(8)]

        return (outp, geo_branch_N, lidar_branch_N, *grads)

# _________________________
# Combined _Attention Weighitng
# _________________________ 

class Combined_s2e_att(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ______________________________________________________________________________________________________________
        # Geo Branch
        # ______________________________________________________________________________________________________________
        self.BN_geo_input = layers.TimeDistributed(layers.BatchNormalization(name='BN_geo_inp'))

        self.Fully_Connected_geo_1 = layers.TimeDistributed(layers.Dense(128, activation=act_geo, name='FC_geo_1'))
        self.Fully_Connected_geo_2 = layers.TimeDistributed(layers.Dense(256, activation=act_geo, name='FC_geo_2'))
        self.drop_geo_1 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=0.1, name='drop_geo_1'))
        # ______________________________________________
        self.Hidden_geo_1 = layers.TimeDistributed(layers.Dense(512, activation=act_geo, name='HS_geo_1'))
        self.Hidden_geo_2 = layers.TimeDistributed(layers.Dense(512, activation=act_geo, name='HS_geo_2'))
        self.drop_geo_2 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=0.1, name='drop_geo_2'))
        # _______________________________________________
        self.Fully_Connected_geo_3 = layers.TimeDistributed(layers.Dense(256, activation=act_geo, name='FC_geo_3'))
        self.Fully_Connected_geo_4 = layers.TimeDistributed(layers.Dense(128, activation=act_geo, name='FC_geo_4'))
        self.drop_geo_3 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=0.1, name='drop_geo_3'))
        # _________________________________________________
        self.RNN_LSTM_geo_1 = layers.LSTM(64,
                                      activation='tanh',
                                      recurrent_activation="sigmoid",
                                      return_sequences=True,
                                      # recurrent_activation="tanh",
                                      # recurrent_dropout=0.2,
                                      unroll=False,
                                      use_bias=True,
                                      name='LSTM_geo_1')
        
        self.RNN_LSTM_geo_2 = layers.LSTM(32,
                                      activation='tanh',
                                      recurrent_activation="sigmoid",
                                      # recurrent_activation="tanh",
                                      # recurrent_dropout=0.2,
                                      unroll=False,
                                      use_bias=True,
                                      name='LSTM_geo_2')
        
        # ______________________________________________________________________________________________________________
        # Lidar Branch
        # ______________________________________________________________________________________________________________
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(training=True))   
        
        # self.Fully_Connected_li0 = layers.TimeDistributed(layers.Dense(1024, activation=act_li, name='FC_lidar_0')) #1
        self.Fully_Connected_li_1 = layers.TimeDistributed(layers.Dense(512, activation=act_li, name='FC_lidar_1')) #1
        self.Fully_Connected_li_2 = layers.TimeDistributed(layers.Dense(256, activation=act_li, name='FC_lidar_2')) #2
        self.Fully_Connected_li_3 = layers.TimeDistributed(layers.Dense(128, activation=act_li, name='FC_lidar_3')) #3
        self.Fully_Connected_li_4 = layers.TimeDistributed(layers.Dense(64, activation=act_li, name='FC_lidar_4'))
        
        # self.Dropout_li0 = layers.TimeDistributed(layers.Dropout(0.2, name='dropout_lidar_0'))
        self.Dropout_li_1 = layers.TimeDistributed(layers.Dropout(0.2, name='dropout_lidar_1'))
        self.Dropout_li_2 = layers.TimeDistributed(layers.Dropout(0.2, name='dropout_lidar_2'))
        self.Dropout_li_3 = layers.TimeDistributed(layers.Dropout(0.2, name='dropout_lidar_3'))
        self.Dropout_li_4 = layers.TimeDistributed(layers.Dropout(0.2, name='dropout_lidar_4'))

        
        self.RNN_LSTM_li_1 = layers.LSTM(64,
                                      activation='tanh',
                                      recurrent_activation="sigmoid",
                                      return_sequences=True,
                                      # recurrent_activation="tanh",
                                      # recurrent_dropout=0.2,
                                      unroll=False,
                                      use_bias=True,
                                      name='LSTM_lidar_1')
        
        self.RNN_LSTM_li_2 = layers.LSTM(32,
                                      activation='tanh',
                                      recurrent_activation="sigmoid",
                                      # recurrent_activation="tanh",
                                      # recurrent_dropout=0.2,
                                      unroll=False,
                                      use_bias=True,
                                      name='LSTM_lidar_2')
        
        # ______________________________________________________________________________________________________________
        # Last module
        # ______________________________________________________________________________________________________________
        self.BN_Geo = layers.BatchNormalization(name='BN_geo_branch')
        self.BN_Li = layers.BatchNormalization(name='BN_Li_branch')

        self.concatenate_end = layers.Concatenate(axis=1)
        
        # self.Attention_co = layers.MultiHeadAttention(num_heads=8, key_dim=64)
        self.Attention_geo = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        self.Attention_lidar = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        
        self.Fully_Connected_end_0 = layers.Dense(32,
                                                activation='tanh',
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end0')
        self.Fully_Connected_end_1 = layers.Dense(16,
                                                activation='tanh',
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end1')
        self.Fully_Connected_end_2 = layers.Dense(8,
                                                activation=None,
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end2')
        
        self.dropout = layers.Dropout(0.5)
        self.layer_norm1 =  layers.BatchNormalization(name='norm1')
        self.layer_norm2 = layers.LayerNormalization(name='norm2')
        self.concatenate_branches = layers.Concatenate(axis=1)



    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = True):
        geo_inp, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        # ______________________________________________________________________________________________________________
        # Geo Branch
        # ______________________________________________________________________________________________________________
        geo_inp = self.BN_geo_input(geo_inp, training=training)

        x1_geo = self.Fully_Connected_geo_1(geo_inp, training=training)
        x2_geo = self.Fully_Connected_geo_2(x1_geo, training=training)
        x2_geo = self.drop_geo_1(x2_geo, training=training)

        # ___________________________________________________________
        x3_geo = self.Hidden_geo_1(x2_geo, training=training)
        x4_geo = self.Hidden_geo_2(x3_geo, training=training)
        x4_geo = self.drop_geo_2(x4_geo, training=training)
        # ____________________________________________________________
        x5_geo = self.Fully_Connected_geo_3(x4_geo, training=training)
        x6_geo = self.Fully_Connected_geo_4(x5_geo, training=training)
        x6_geo = self.drop_geo_3(x6_geo, training=training)

        # ______________________________________________________________
        x7_geo = self.RNN_LSTM_geo_1(x6_geo, training=training)
        geo_branch = self.RNN_LSTM_geo_2(x7_geo, training=training)
        
        # ______________________________________________________________________________________________________________
        # Lidar Branch
        # ______________________________________________________________________________________________________________
        x_Resnet = self.create_res_net(lidar_inp)
        
        # FC1 = self.Dropout_li0(self.Fully_Connected_li0(x12))
        x1_li = self.Dropout_li_1(self.Fully_Connected_li_1(x_Resnet), training=training)
        x2_li = self.Dropout_li_2(self.Fully_Connected_li_2(x1_li), training=training)
        x3_li = self.Dropout_li_3(self.Fully_Connected_li_3(x2_li), training=training)
        x4_li = self.Dropout_li_4(self.Fully_Connected_li_4(x3_li), training=training)

        x5_li = self.RNN_LSTM_li_1(x4_li)
        lidar_branch = self.RNN_LSTM_li_2(x5_li)
        
        # ______________________________________________________________________________________________________________
        # Last module
        # ______________________________________________________________________________________________________________
        geo_branch_N = self.BN_Geo(geo_branch, training=training)
        lidar_branch_N = self.BN_Li(lidar_branch, training=training)
        
        combined = self.concatenate_end([geo_branch_N, lidar_branch_N])
        # combined = self.layer_norm1(combined)
        
        combined_res = combined
        
        # combined = layers.Reshape((combined.shape[1], 1))(combined)  
        # combined = self.Attention_co(combined, combined, combined)
        geo_branch_N = layers.Reshape((geo_branch_N.shape[1], 1))(geo_branch_N)
        lidar_branch_N = layers.Reshape((lidar_branch_N.shape[1], 1))(lidar_branch_N)
        attended_geo = self.Attention_geo(geo_branch_N, geo_branch_N, geo_branch_N)
        attended_lidar = self.Attention_lidar(lidar_branch_N, lidar_branch_N, lidar_branch_N)
        attended_geo = layers.Flatten()(attended_geo)
        attended_lidar = layers.Flatten()(attended_lidar)
        combined = self.concatenate_branches([attended_geo, attended_lidar])
      
        # combined = layers.Flatten()(combined)
        
        combined = layers.Add()([combined_res, combined])
        combined = self.layer_norm2(combined)
        
        out0 = self.Fully_Connected_end_0(combined, training=training)
        out1 = self.Fully_Connected_end_1(out0, training=training)
        outp = self.Fully_Connected_end_2(out1, training=training)
        return outp


class Combined_s2e_att_two(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ______________________________________________________________________________________________________________
        # Geo Branch
        # ______________________________________________________________________________________________________________
        self.BN_geo_input = layers.TimeDistributed(layers.BatchNormalization(name='BN_geo_inp'))

        self.Fully_Connected_geo_1 = layers.TimeDistributed(layers.Dense(128, activation=act_geo, name='FC_geo_1'))
        self.Fully_Connected_geo_2 = layers.TimeDistributed(layers.Dense(256, activation=act_geo, name='FC_geo_2'))
        self.drop_geo_1 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=0.1, name='drop_geo_1'))
        # ______________________________________________
        self.Hidden_geo_1 = layers.TimeDistributed(layers.Dense(512, activation=act_geo, name='HS_geo_1'))
        self.Hidden_geo_2 = layers.TimeDistributed(layers.Dense(512, activation=act_geo, name='HS_geo_2'))
        self.drop_geo_2 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=0.1, name='drop_geo_2'))
        # _______________________________________________
        self.Fully_Connected_geo_3 = layers.TimeDistributed(layers.Dense(256, activation=act_geo, name='FC_geo_3'))
        self.Fully_Connected_geo_4 = layers.TimeDistributed(layers.Dense(128, activation=act_geo, name='FC_geo_4'))
        self.drop_geo_3 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=0.1, name='drop_geo_3'))
        # _________________________________________________
        self.RNN_LSTM_geo_1 = layers.LSTM(64,
                                      activation='tanh',
                                      recurrent_activation="sigmoid",
                                      return_sequences=True,
                                      # recurrent_activation="tanh",
                                      # recurrent_dropout=0.2,
                                      unroll=False,
                                      use_bias=True,
                                      name='LSTM_geo_1')
        
        self.RNN_LSTM_geo_2 = layers.LSTM(32,
                                      activation='tanh',
                                      recurrent_activation="sigmoid",
                                      # recurrent_activation="tanh",
                                      # recurrent_dropout=0.2,
                                      unroll=False,
                                      use_bias=True,
                                      name='LSTM_geo_2')
        
        # ______________________________________________________________________________________________________________
        # Lidar Branch
        # ______________________________________________________________________________________________________________
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(training=True))   
        
        # self.Fully_Connected_li0 = layers.TimeDistributed(layers.Dense(1024, activation=act_li, name='FC_lidar_0')) #1
        self.Fully_Connected_li_1 = layers.TimeDistributed(layers.Dense(512, activation=act_li, name='FC_lidar_1')) #1
        self.Fully_Connected_li_2 = layers.TimeDistributed(layers.Dense(256, activation=act_li, name='FC_lidar_2')) #2
        self.Fully_Connected_li_3 = layers.TimeDistributed(layers.Dense(128, activation=act_li, name='FC_lidar_3')) #3
        self.Fully_Connected_li_4 = layers.TimeDistributed(layers.Dense(64, activation=act_li, name='FC_lidar_4'))
        
        # self.Dropout_li0 = layers.TimeDistributed(layers.Dropout(0.2, name='dropout_lidar_0'))
        self.Dropout_li_1 = layers.TimeDistributed(layers.Dropout(0.2, name='dropout_lidar_1'))
        self.Dropout_li_2 = layers.TimeDistributed(layers.Dropout(0.2, name='dropout_lidar_2'))
        self.Dropout_li_3 = layers.TimeDistributed(layers.Dropout(0.2, name='dropout_lidar_3'))
        self.Dropout_li_4 = layers.TimeDistributed(layers.Dropout(0.2, name='dropout_lidar_4'))

        
        self.RNN_LSTM_li_1 = layers.LSTM(64,
                                      activation='tanh',
                                      recurrent_activation="sigmoid",
                                      return_sequences=True,
                                      # recurrent_activation="tanh",
                                      # recurrent_dropout=0.2,
                                      unroll=False,
                                      use_bias=True,
                                      name='LSTM_lidar_1')
        
        self.RNN_LSTM_li_2 = layers.LSTM(32,
                                      activation='tanh',
                                      recurrent_activation="sigmoid",
                                      # recurrent_activation="tanh",
                                      # recurrent_dropout=0.2,
                                      unroll=False,
                                      use_bias=True,
                                      name='LSTM_lidar_2')
        
        # ______________________________________________________________________________________________________________
        # Last module
        # ______________________________________________________________________________________________________________
        self.BN_Geo = layers.BatchNormalization(name='BN_geo_branch')
        self.BN_Li = layers.BatchNormalization(name='BN_Li_branch')

        self.concatenate_end = layers.Concatenate(axis=1)
        
        # self.Attention_co = layers.MultiHeadAttention(num_heads=8, key_dim=64)
        self.Attention_geo = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        self.Attention_lidar = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        
        self.Fully_Connected_end_0 = layers.Dense(32,
                                                activation='tanh',
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end0')
        self.Fully_Connected_end_1 = layers.Dense(16,
                                                activation='tanh',
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end1')
        self.Fully_Connected_end_2 = layers.Dense(8,
                                                activation=None,
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end2')
        
        self.dropout = layers.Dropout(0.5)
        self.layer_norm1 =  layers.BatchNormalization(name='norm1')
        self.layer_norm2 = layers.LayerNormalization(name='norm2')
        self.concatenate_branches = layers.Concatenate(axis=1)




    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = False):
        geo_inp, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        # ______________________________________________________________________________________________________________
        # Geo Branch
        # ______________________________________________________________________________________________________________
        geo_inp = self.BN_geo_input(geo_inp, training=training)

        x1_geo = self.Fully_Connected_geo_1(geo_inp, training=training)
        x2_geo = self.Fully_Connected_geo_2(x1_geo, training=training)
        x2_geo = self.drop_geo_1(x2_geo, training=training)

        # ___________________________________________________________
        x3_geo = self.Hidden_geo_1(x2_geo, training=training)
        x4_geo = self.Hidden_geo_2(x3_geo, training=training)
        x4_geo = self.drop_geo_2(x4_geo, training=training)
        # ____________________________________________________________
        x5_geo = self.Fully_Connected_geo_3(x4_geo, training=training)
        x6_geo = self.Fully_Connected_geo_4(x5_geo, training=training)
        x6_geo = self.drop_geo_3(x6_geo, training=training)

        # ______________________________________________________________
        x7_geo = self.RNN_LSTM_geo_1(x6_geo, training=training)
        geo_branch = self.RNN_LSTM_geo_2(x7_geo, training=training)
        
        # ______________________________________________________________________________________________________________
        # Lidar Branch
        # ______________________________________________________________________________________________________________
        x_Resnet = self.create_res_net(lidar_inp)
        
        # FC1 = self.Dropout_li0(self.Fully_Connected_li0(x12))
        x1_li = self.Dropout_li_1(self.Fully_Connected_li_1(x_Resnet), training=training)
        x2_li = self.Dropout_li_2(self.Fully_Connected_li_2(x1_li), training=training)
        x3_li = self.Dropout_li_3(self.Fully_Connected_li_3(x2_li), training=training)
        x4_li = self.Dropout_li_4(self.Fully_Connected_li_4(x3_li), training=training)

        x5_li = self.RNN_LSTM_li_1(x4_li)
        lidar_branch = self.RNN_LSTM_li_2(x5_li)
        
        # ______________________________________________________________________________________________________________
        # Last module
        # ______________________________________________________________________________________________________________
        geo_branch_N = self.BN_Geo(geo_branch, training=training)
        lidar_branch_N = self.BN_Li(lidar_branch, training=training)
        
        combined = self.concatenate_end([geo_branch_N, lidar_branch_N])
        # combined = self.layer_norm1(combined)
        
        combined_res = combined
        
        # combined = layers.Reshape((combined.shape[1], 1))(combined)  
        # combined = self.Attention_co(combined, combined, combined)
        geo_branch_N = layers.Reshape((geo_branch_N.shape[1], 1))(geo_branch_N)
        lidar_branch_N = layers.Reshape((lidar_branch_N.shape[1], 1))(lidar_branch_N)
        attended_geo = self.Attention_geo(geo_branch_N, geo_branch_N, geo_branch_N)
        attended_lidar = self.Attention_lidar(lidar_branch_N, lidar_branch_N, lidar_branch_N)
        attended_geo = layers.Flatten()(attended_geo)
        attended_lidar = layers.Flatten()(attended_lidar)
        combined = self.concatenate_branches([attended_geo, attended_lidar])
      
        # combined = layers.Flatten()(combined)
        
        combined = layers.Add()([combined_res, combined])
        combined = self.layer_norm2(combined)
        
        out0 = self.Fully_Connected_end_0(combined, training=training)
        out1 = self.Fully_Connected_end_1(out0, training=training)
        outp = self.Fully_Connected_end_2(out1, training=training)

        self.layer_grad0 = tf.gradients(outp[:, 0], combined)
        self.layer_grad1 = tf.gradients(outp[:, 1], combined)
        self.layer_grad2 = tf.gradients(outp[:, 2], combined)
        self.layer_grad3 = tf.gradients(outp[:, 3], combined)
        self.layer_grad4 = tf.gradients(outp[:, 4], combined)
        self.layer_grad5 = tf.gradients(outp[:, 5], combined)
        self.layer_grad6 = tf.gradients(outp[:, 6], combined)
        self.layer_grad7 = tf.gradients(outp[:, 7], combined)

        return outp, geo_branch, lidar_branch, \
            self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3, \
            self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7

class Combined_s2e_att_simple(Model):
    def __init__(self, 
                 # ----- Geo branch parameters -----
                 act_geo1='relu', act_geo2='relu', act_geo4='relu',                 
                 dropout_rate_1_geo=0.3, 
                 e1_geo=128, e2_geo=256, d3_geo=256,
                 act_geo_lstm1='relu', act_geo_lstm2='relu', 
                 do_lstm_geo_3=0.2, do_lstm_geo_4=0.2,
                 
                 # ----- Lidar branch (ResNet + Dense + LSTM) -----
                 filters2=64, filters3=128, filters4=256,
                 dense_last=512, act_last='relu',
                 act1_li='relu', act2_li='relu',
                 dropout_rate_li=0.2,
                 d1_li=256, d2_li=128,
                 act_li_lstm3='tanh', act_li_lstm4='tanh', 
                 do_lstm_li_3=0.1, do_lstm_li_4=0.1,

                 # ----- Fusion and output layer parameters -----
                 d1_fuse=64, act_fuse='relu', 
                 dropout_rate_fuse=0.1,
                 d3_end=32, d2_end=16, 
                 act_end_1='tanh', act_end_2='tanh',
                 output_dim=8,
                 **kwargs):
        super().__init__(**kwargs)

        # ----- Geo branch: dense + LSTM pipeline -----
        self.BN_geo_input = layers.TimeDistributed(layers.BatchNormalization(name='BN_geo_inp'))
        self.Fully_Connected_en_1_geo = layers.TimeDistributed(layers.Dense(e1_geo, activation=act_geo1))
        self.Fully_Connected_en_2_geo = layers.TimeDistributed(layers.Dense(e2_geo, activation=act_geo2))        
        self.Dropout_geo_1 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=dropout_rate_1_geo))
        self.Fully_Connected_de_2_geo = layers.TimeDistributed(layers.Dense(d3_geo, activation=act_geo4))
        self.RNN_LSTM_geo_2 = layers.LSTM(32, activation=act_geo_lstm1, recurrent_activation=act_geo_lstm2,
                                          unroll=False, use_bias=True, dropout=do_lstm_geo_3, recurrent_dropout=do_lstm_geo_4,
                                          name='LSTM_geo_2')

        # ----- Lidar branch: ResNet feature extraction + Dense + LSTM -----
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(num_filters=64, filters1=64,
                                                                           filters2=filters2, filters3=filters3,
                                                                           filters4=filters4, 
                                                                           dense_last=dense_last, act_last=act_last))   
        self.Fully_Connected_li_1 = layers.TimeDistributed(layers.Dense(d1_li, activation=act1_li)) 
        self.Fully_Connected_li_2 = layers.TimeDistributed(layers.Dense(d2_li, activation=act2_li)) 
        self.Dropout_li_1 = layers.TimeDistributed(layers.Dropout(dropout_rate_li))
        self.RNN_LSTM_li_2 = layers.LSTM(32, activation=act_li_lstm3, recurrent_activation=act_li_lstm4,
                                         unroll=False, use_bias=True, dropout=do_lstm_li_3, recurrent_dropout=do_lstm_li_4,
                                         name='LSTM_lidar_2')

        # ----- Fusion: normalization + attention + concatenation -----
        self.BN_Geo = layers.BatchNormalization()
        self.BN_Li = layers.BatchNormalization()
        self.Attention_geo = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        self.Attention_lidar = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        self.concatenate_end = layers.Concatenate(axis=1)
        self.concatenate_branches = layers.Concatenate(axis=1)
        self.layer_norm2 = layers.LayerNormalization()

        # ----- Final output dense layers -----
        self.Fully_Connected_end_0 = layers.Dense(d3_end, activation=act_end_1)
        self.Fully_Connected_end_1 = layers.Dense(d2_end, activation=act_end_2)
        self.Fully_Connected_end_2 = layers.Dense(output_dim, activation=None)

        # ----- Output normalization -----
        self.DualQuaternionNormalization = DualQuaternionNormalization()

    
    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = False):
        geo_inp, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        # ----- Geo forward pass -----
        geo_inp1 = self.BN_geo_input(geo_inp, training=training)
        x1_geo = self.Fully_Connected_en_1_geo(geo_inp1, training=training)
        x2_geo = self.Fully_Connected_en_2_geo(x1_geo, training=training)
        x2_geo = self.Dropout_geo_1(x2_geo, training=training)
        x_mid = self.Fully_Connected_de_2_geo(x2_geo, training=training)
        geo_branch = self.RNN_LSTM_geo_2(x_mid, training=training)

        # ----- Lidar forward pass -----
        x_Resnet = self.create_res_net(lidar_inp, training=training)
        x1_li = self.Dropout_li_1(self.Fully_Connected_li_1(x_Resnet), training=training)
        x2_li = self.Fully_Connected_li_2(x1_li, training=training)
        lidar_branch = self.RNN_LSTM_li_2(x2_li, training=training)

        # ----- Batch normalization for both branches -----
        geo_branch_N = self.BN_Geo(geo_branch, training=training)
        lidar_branch_N = self.BN_Li(lidar_branch, training=training)

        # ----- Save pre-attention combined features -----
        combined_res = self.concatenate_end([geo_branch_N, lidar_branch_N])

        # ----- Multi-head attention over each branch -----
        geo_att_in = tf.expand_dims(geo_branch_N, axis=1)  # Shape: (B, 1, D)
        lidar_att_in = tf.expand_dims(lidar_branch_N, axis=1)
        attended_geo = self.Attention_geo(geo_att_in, geo_att_in, geo_att_in)
        attended_lidar = self.Attention_lidar(lidar_att_in, lidar_att_in, lidar_att_in)

        # ----- Concatenate attended outputs -----
        attended_geo = layers.Flatten()(attended_geo)
        attended_lidar = layers.Flatten()(attended_lidar)
        combined = self.concatenate_branches([attended_geo, attended_lidar])

        # ----- Residual + layer normalization -----
        combined = layers.Add()([combined_res, combined])
        combined = self.layer_norm2(combined)

        # ----- Final dense layers for regression/classification -----
        out0 = self.Fully_Connected_end_0(combined, training=training)
        out1 = self.Fully_Connected_end_1(out0, training=training)
        outp = self.Fully_Connected_end_2(out1, training=training)
        outp = self.DualQuaternionNormalization(outp)

        return outp

class Combined_s2e_att_two_simple(Model):
    def __init__(self, 
                 # ----- Geo branch parameters -----
                 act_geo1='relu', act_geo2='relu', act_geo4='relu',                 
                 dropout_rate_1_geo=0.3, 
                 e1_geo=128, e2_geo=256, d3_geo=256,
                 act_geo_lstm1='relu', act_geo_lstm2='relu', 
                 do_lstm_geo_3=0.2, do_lstm_geo_4=0.2,
                 
                 # ----- Lidar branch (ResNet + Dense + LSTM) -----
                 filters2=64, filters3=128, filters4=256,
                 dense_last=512, act_last='relu',
                 act1_li='relu', act2_li='relu',
                 dropout_rate_li=0.2,
                 d1_li=256, d2_li=128,
                 act_li_lstm3='tanh', act_li_lstm4='tanh', 
                 do_lstm_li_3=0.1, do_lstm_li_4=0.1,

                 # ----- Fusion and output layer parameters -----
                 d1_fuse=64, act_fuse='relu', 
                 dropout_rate_fuse=0.1,
                 d3_end=32, d2_end=16, 
                 act_end_1='tanh', act_end_2='tanh',
                 output_dim=8,
                 **kwargs):
        super().__init__(**kwargs)

        # ----- Geo branch: dense + LSTM pipeline -----
        self.BN_geo_input = layers.TimeDistributed(layers.BatchNormalization(name='BN_geo_inp'))
        self.Fully_Connected_en_1_geo = layers.TimeDistributed(layers.Dense(e1_geo, activation=act_geo1))
        self.Fully_Connected_en_2_geo = layers.TimeDistributed(layers.Dense(e2_geo, activation=act_geo2))        
        self.Dropout_geo_1 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=dropout_rate_1_geo))
        self.Fully_Connected_de_2_geo = layers.TimeDistributed(layers.Dense(d3_geo, activation=act_geo4))
        self.RNN_LSTM_geo_2 = layers.LSTM(32, activation=act_geo_lstm1, recurrent_activation=act_geo_lstm2,
                                          unroll=False, use_bias=True, dropout=do_lstm_geo_3, recurrent_dropout=do_lstm_geo_4,
                                          name='LSTM_geo_2')

        # ----- Lidar branch: ResNet feature extraction + Dense + LSTM -----
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(num_filters=64, filters1=64,
                                                                           filters2=filters2, filters3=filters3,
                                                                           filters4=filters4, 
                                                                           dense_last=dense_last, act_last=act_last))   
        self.Fully_Connected_li_1 = layers.TimeDistributed(layers.Dense(d1_li, activation=act1_li)) 
        self.Fully_Connected_li_2 = layers.TimeDistributed(layers.Dense(d2_li, activation=act2_li)) 
        self.Dropout_li_1 = layers.TimeDistributed(layers.Dropout(dropout_rate_li))
        self.RNN_LSTM_li_2 = layers.LSTM(32, activation=act_li_lstm3, recurrent_activation=act_li_lstm4,
                                         unroll=False, use_bias=True, dropout=do_lstm_li_3, recurrent_dropout=do_lstm_li_4,
                                         name='LSTM_lidar_2')

        # ----- Fusion: normalization + attention + concatenation -----
        self.BN_Geo = layers.BatchNormalization()
        self.BN_Li = layers.BatchNormalization()
        self.Attention_geo = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        self.Attention_lidar = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        self.concatenate_end = layers.Concatenate(axis=1)
        self.concatenate_branches = layers.Concatenate(axis=1)
        self.layer_norm2 = layers.LayerNormalization()

        # ----- Final output dense layers -----
        self.Fully_Connected_end_0 = layers.Dense(d3_end, activation=act_end_1)
        self.Fully_Connected_end_1 = layers.Dense(d2_end, activation=act_end_2)
        self.Fully_Connected_end_2 = layers.Dense(output_dim, activation=None)

        # ----- Output normalization -----
        self.DualQuaternionNormalization = DualQuaternionNormalization()

    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = False):
        geo_inp, lidar_inp = geo_lidar_input['geo_input'], geo_lidar_input['AI_input']

        # ----- Geo forward pass -----
        geo_inp1 = self.BN_geo_input(geo_inp, training=training)
        x1_geo = self.Fully_Connected_en_1_geo(geo_inp1, training=training)
        x2_geo = self.Fully_Connected_en_2_geo(x1_geo, training=training)
        x2_geo = self.Dropout_geo_1(x2_geo, training=training)
        x_mid = self.Fully_Connected_de_2_geo(x2_geo, training=training)
        geo_branch = self.RNN_LSTM_geo_2(x_mid, training=training)

        # ----- Lidar forward pass -----
        x_Resnet = self.create_res_net(lidar_inp, training=training)
        x1_li = self.Dropout_li_1(self.Fully_Connected_li_1(x_Resnet), training=training)
        x2_li = self.Fully_Connected_li_2(x1_li, training=training)
        lidar_branch = self.RNN_LSTM_li_2(x2_li, training=training)

        # ----- Batch normalization for both branches -----
        geo_branch_N = self.BN_Geo(geo_branch, training=training)
        lidar_branch_N = self.BN_Li(lidar_branch, training=training)

        # ----- Save pre-attention combined features -----
        combined_res = self.concatenate_end([geo_branch_N, lidar_branch_N])

        # ----- Multi-head attention over each branch -----
        geo_att_in = tf.expand_dims(geo_branch_N, axis=1)  # Shape: (B, 1, D)
        lidar_att_in = tf.expand_dims(lidar_branch_N, axis=1)
        attended_geo = self.Attention_geo(geo_att_in, geo_att_in, geo_att_in)
        attended_lidar = self.Attention_lidar(lidar_att_in, lidar_att_in, lidar_att_in)

        # ----- Concatenate attended outputs -----
        attended_geo = layers.Flatten()(attended_geo)
        attended_lidar = layers.Flatten()(attended_lidar)
        combined = self.concatenate_branches([attended_geo, attended_lidar])

        # ----- Residual + layer normalization -----
        combined = layers.Add()([combined_res, combined])
        combined = self.layer_norm2(combined)

        # ----- Final dense layers for regression/classification -----
        out0 = self.Fully_Connected_end_0(combined, training=training)
        out1 = self.Fully_Connected_end_1(out0, training=training)
        outp = self.Fully_Connected_end_2(out1, training=training)
        outp = self.DualQuaternionNormalization(outp)
        
        self.layer_grad0 = tf.gradients(outp[:, 0], combined)
        self.layer_grad1 = tf.gradients(outp[:, 1], combined)
        self.layer_grad2 = tf.gradients(outp[:, 2], combined)
        self.layer_grad3 = tf.gradients(outp[:, 3], combined)
        self.layer_grad4 = tf.gradients(outp[:, 4], combined)
        self.layer_grad5 = tf.gradients(outp[:, 5], combined)
        self.layer_grad6 = tf.gradients(outp[:, 6], combined)
        self.layer_grad7 = tf.gradients(outp[:, 7], combined)

        return outp, geo_branch_N, lidar_branch_N, \
            self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3, \
            self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7

    
# _________________________
# Combined _INAF Weighitng
# _________________________ 
class Combined_s2e_INAF(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.Attention_Li = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        self.Attention_geo = layers.MultiHeadAttention(num_heads=4, key_dim=32)


        self.Fully_Connected_end1 = layers.Dense(16,
                                        activation=None,
                                        # use_bias=True,
                                        # bias_regularizer=regularizers.L2(1e-4),
                                        name='FC_all_end0')
        self.Fully_Connected_end2 = layers.Dense(8,
                                                activation=None,
                                                # use_bias=True,
                                                # bias_regularizer=regularizers.L2(1e-4),
                                                name='FC_all_end')

    @tf.function
    def call(self, geo_lidar_input, training: bool = True):
        INAF_P_geo, INAF_P_Li, INAF_W = geo_lidar_input['Param_geo'], geo_lidar_input['Param_Li'], geo_lidar_input['Weight']
        
        
        inputs_geo_expanded = tf.expand_dims(INAF_P_geo, axis=1)
        inputs_Li_expanded = tf.expand_dims(INAF_P_Li, axis=1)

        # Apply pre-made MultiHeadAttention
        attention_output_geo = self.Attention_geo(query=inputs_geo_expanded, value=inputs_geo_expanded, attention_mask=None)
        attention_output_Li = self.Attention_Li(query=inputs_Li_expanded, value=inputs_Li_expanded, attention_mask=None)

        # Squeeze and reshape to remove the added dimension
        attention_output_geo = tf.squeeze(attention_output_geo, axis=1)
        attention_output_Li = tf.squeeze(attention_output_Li, axis=1)

        # Apply weights to attention output
        attention_output = self.concatenate_branches([attention_output_geo, attention_output_Li]) 
        weighted_attention_output = attention_output * INAF_W

        
        combined = self.concatenate_branches([attended_geo, attended_lidar])      
        combined = layers.Add()([combined_res, combined])
        combined = self.layer_norm2(combined)
        
        out0 = self.Fully_Connected_end_0(combined, training=training)
        out1 = self.Fully_Connected_end_1(out0, training=training)
        outp = self.Fully_Connected_end_2(out1, training=training)
        
        
        # Expand dimensions along the second axis
        # INAF_P_expanded = tf.expand_dims(INAF_P, axis=1)
        # INAF_W_expanded = tf.expand_dims(INAF_W, axis=1)

        # Apply self-attention (Q=INAF_P, K=INAF_P, V=INAF_W)
        # FC5_att = self.Attention_li(INAF_W_expanded, INAF_P_expanded, INAF_P_expanded)
        
        out0 = self.Fully_Connected_end1(weighted_attention_output, training=training)
        outp = self.Fully_Connected_end2(out0, training=training)

        return outp
# ______________________________________________________________________________________________________________________
# Geo Branch
# ______________________________________________________________________________________________________________________
class GeoLayer(layers.Layer):
    """``CustomGeoLayer``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_grad = None
        self.Fully_Connected1 = layers.TimeDistributed(layers.Dense(128, activation='tanh', name='FC_geo_1'))
        self.Fully_Connected2 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_geo_2'))
        self.Fully_Connected3 = layers.LSTM(8,
                                            activation='tanh',
                                            recurrent_activation="sigmoid",
                                            # recurrent_activation="tanh",
                                            # recurrent_dropout=0.2,
                                            unroll=False,
                                            use_bias=True,
                                            name='LSTM_geo'
                                            )

    @tf.function
    def call(self, input_tensor: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        x1 = self.Fully_Connected1(input_tensor)
        x2 = self.Fully_Connected2(x1)
        x3 = self.Fully_Connected3(x2, training=training)
        return x3


class GeoLayer_two(layers.Layer):
    """``CustomGeoLayer``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3 = None, None, None, None
        self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7 = None, None, None, None

        self.layer_grad = None

        self.Fully_Connected1 = layers.TimeDistributed(layers.Dense(128, activation='tanh', name='FC_geo_1'))
        self.Fully_Connected2 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_geo_2'))
        self.Fully_Connected3 = layers.LSTM(8,
                                            activation='tanh',
                                            recurrent_activation="sigmoid",
                                            # recurrent_activation="tanh",
                                            # recurrent_dropout=0.2,
                                            unroll=False,
                                            use_bias=True,
                                            name='LSTM_geo'
                                            )

    @tf.function
    def call(self, input_tensor: tf.Tensor, training: bool = True):
        x1 = self.Fully_Connected1(input_tensor)
        x2 = self.Fully_Connected2(x1)
        x3 = self.Fully_Connected3(x2)
        self.layer_grad0 = tf.gradients(x3[:, 0], x2)
        self.layer_grad1 = tf.gradients(x3[:, 2], x2)
        self.layer_grad2 = tf.gradients(x3[:, 2], x2)
        self.layer_grad3 = tf.gradients(x3[:, 3], x2)
        self.layer_grad4 = tf.gradients(x3[:, 4], x2)
        self.layer_grad5 = tf.gradients(x3[:, 5], x2)
        self.layer_grad6 = tf.gradients(x3[:, 6], x2)
        self.layer_grad7 = tf.gradients(x3[:, 7], x2)
        return x3, self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3, self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7


class GeoLayer_intermed(layers.Layer):
    """``CustomGeoLayer``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_grad = None
        self.Fully_Connected1 = layers.TimeDistributed(layers.Dense(128, activation='tanh', name='FC_geo_1'))
        self.Fully_Connected2 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_geo_2'))

    @tf.function
    def call(self, input_tensor: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        x1 = self.Fully_Connected1(input_tensor)
        x2 = self.Fully_Connected2(x1)

        return x2


# For Geo Only
class GeoLayer_end(layers.Layer):
    """``CustomGeoLayer``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bn = layers.TimeDistributed(layers.BatchNormalization())
        # self.activaction_tanh = activations.tanh
        self.activaction_tanh = layers.TimeDistributed(layers.Activation(activation='tanh'))
        self.RNN_LSTM = layers.LSTM(8,
                                    activation='tanh',
                                    recurrent_activation="sigmoid",
                                    # recurrent_activation="tanh",
                                    # recurrent_dropout=0.2,
                                    unroll=False,
                                    use_bias=True,
                                    name='LSTM_end'
                                    )

    @tf.function
    def call(self, intermed_layer, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        bn = self.bn(intermed_layer[1])
        w1 = self.activaction_tanh(bn)
        wx = tf.math.multiply(intermed_layer[0], w1)
        outp = self.RNN_LSTM(wx, training=training)
        # outp = self.RNN_LSTM(intermed_layer[0], training=training)
        return outp


# ______________________________________________________________________________________________________________________
# LiDAR Branch
# ______________________________________________________________________________________________________________________
class Lidar_layer(Model):
    """``Custom Lidar Layer``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.Fully_Connected1 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='Lidar_model_1'))
        self.Fully_Connected2 = layers.LSTM(8,
                                            activation='tanh',
                                            recurrent_activation="sigmoid",
                                            # recurrent_activation="tanh",
                                            # recurrent_dropout=0.2,
                                            unroll=False,
                                            use_bias=True,
                                            name='LSTM_Lidar_1'
                                            )

        self.create_res_net = create_res_net()

    @tf.function
    def call(self, input_tensor: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        Layer_Resnet = self.create_res_net(input_tensor)
        lidar_branch = self.Fully_Connected1(Layer_Resnet)
        x3 = self.Fully_Connected2(lidar_branch, training=training)

        return x3


class Lidar_layer_two(Model):
    """``Custom Lidar Layer``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3 = None, None, None, None
        self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7 = None, None, None, None

        self.Fully_Connected1 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='Lidar_model_1'))
        self.Fully_Connected2 = layers.LSTM(8,
                                            activation='tanh',
                                            recurrent_activation="sigmoid",
                                            # recurrent_activation="tanh",
                                            # recurrent_dropout=0.2,
                                            unroll=False,
                                            use_bias=True,
                                            name='LSTM_Lidar_1'
                                            )

        self.create_res_net = create_res_net()

    @tf.function
    def call(self, input_tensor: tf.Tensor, training: bool = True):
        Layer_Resnet = self.create_res_net(input_tensor)
        lidar_branch = self.Fully_Connected1(Layer_Resnet)
        x3 = self.Fully_Connected2(lidar_branch, training=training)
        self.layer_grad0 = tf.gradients(x3[:, 0], lidar_branch)
        self.layer_grad1 = tf.gradients(x3[:, 2], lidar_branch)
        self.layer_grad2 = tf.gradients(x3[:, 2], lidar_branch)
        self.layer_grad3 = tf.gradients(x3[:, 3], lidar_branch)
        self.layer_grad4 = tf.gradients(x3[:, 4], lidar_branch)
        self.layer_grad5 = tf.gradients(x3[:, 5], lidar_branch)
        self.layer_grad6 = tf.gradients(x3[:, 6], lidar_branch)
        self.layer_grad7 = tf.gradients(x3[:, 7], lidar_branch)
        return x3, self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3, self.layer_grad4, \
            self.layer_grad5, self.layer_grad6, self.layer_grad7


class Lidar_layer_intermed(Model):
    """``Custom Lidar Layer``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.Fully_Connected1 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='Lidar_model_1'))
        self.create_res_net = create_res_net()

    @tf.function
    def call(self, input_tensor: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        Layer_Resnet = self.create_res_net(input_tensor)
        lidar_branch = self.Fully_Connected1(Layer_Resnet)

        return lidar_branch


# FOr Lidar Only
class Lidar_layer_end(Model):
    """``Custom Lidar Layer``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.bn = layers.TimeDistributed(layers.BatchNormalization())
        self.activaction_tanh = layers.TimeDistributed(layers.Activation(activation='tanh'))
        # self.activaction_tanh = activations.tanh
        self.RNN_LSTM = layers.LSTM(8,
                                    activation='tanh',
                                    recurrent_activation="sigmoid",
                                    # recurrent_activation="tanh",
                                    # recurrent_dropout=0.2,
                                    unroll=False,
                                    use_bias=True,
                                    name='LSTM_end'
                                    )

    @tf.function
    def call(self, intermed_layer: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        bn = self.bn(intermed_layer[1])
        w1 = self.activaction_tanh(bn)
        wx = tf.math.multiply(intermed_layer[0], w1)
        outp = self.RNN_LSTM(wx, training=training)
        return outp



# _________________________
# Soft
# _________________________

class Combined_s2e_soft_nowe(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Fully_Connected1 = layers.TimeDistributed(layers.Dense(128, activation='tanh', name='FC_geo_1'))
        self.Fully_Connected2 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_geo_2'))
        self.BN_geo = layers.TimeDistributed(layers.BatchNormalization())

        self.create_res_net = create_res_net()
        self.Fully_Connected_li1 = layers.TimeDistributed(layers.Dense(512, activation='tanh', name='FC_lidar_1'))
        self.Fully_Connected_li2 = layers.TimeDistributed(layers.Dense(512, activation='tanh', name='FC_lidar_2'))
        self.Fully_Connected_li3 = layers.TimeDistributed(layers.Dense(256, activation='tanh', name='FC_lidar_3'))
        self.Fully_Connected_li4 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_lidar_4'))
        self.BN_lidar = layers.TimeDistributed(layers.BatchNormalization())

        self.concatenate = layers.Concatenate(axis=2)
        self.dense_soft_12 = layers.TimeDistributed(layers.Dense(128, activation='tanh', name='dense_soft_12'))

        self.dense_soft_1 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='dense_soft_1'))
        self.dense_soft_2 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='dense_soft_2'))
        # self.act_soft_1 = tf.keras.activations.sigmoid
        # self.act_soft_2 = tf.keras.activations.sigmoid
        # self.concat_soft_1 = layers.Concatenate(axis=2)

        self.Fully_Connected_all_1 = layers.TimeDistributed(layers.Dense(128, activation='tanh', name='FC_all_1'))
        self.Fully_Connected_all_2 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_all_2'))
        self.RNN_LSTM = layers.LSTM(8,
                                    activation='tanh',
                                    recurrent_activation="sigmoid",
                                    # recurrent_activation="tanh",
                                    # recurrent_dropout=0.2,
                                    unroll=False,
                                    use_bias=True,
                                    name='LSTM_end'
                                    )
        self.MHA = layers.Attention()

    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = True):
        geo_inp, lidar_inp = geo_lidar_input[0], geo_lidar_input[1]

        x1_geo = self.Fully_Connected1(geo_inp)
        geo_branch = self.Fully_Connected2(x1_geo)
        # geo_branch = self.BN_geo(geo_branch)

        Layer_Resnet = self.create_res_net(lidar_inp)
        lidar_branch_1 = self.Fully_Connected_li1(Layer_Resnet)
        lidar_branch_2 = self.Fully_Connected_li2(lidar_branch_1)
        lidar_branch_3 = self.Fully_Connected_li3(lidar_branch_2)
        lidar_branch = self.Fully_Connected_li4(lidar_branch_3)
        # lidar_branch = self.BN_lidar(lidar_branch)

        # ___________
        # Soft Fusion
        # ___________
        features_concat = self.concatenate([geo_branch, lidar_branch])
        mask_dense = self.dense_soft_12(features_concat)
        mask = tf.keras.activations.sigmoid(mask_dense)
        soft_masked = tf.math.multiply(features_concat, mask)

        # Fully_Connected = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc2)

        intint1 = self.Fully_Connected_all_1(soft_masked)
        intint2 = self.Fully_Connected_all_2(intint1)
        outp = self.RNN_LSTM(intint2, training=training)

        return outp


class Combined_s2e_soft_nowe_two(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3 = None, None, None, None
        self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7 = None, None, None, None

        self.Fully_Connected1 = layers.TimeDistributed(layers.Dense(128, activation='tanh', name='FC_geo_1'))
        self.Fully_Connected2 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_geo_2'))
        self.BN_geo = layers.TimeDistributed(layers.BatchNormalization())

        self.create_res_net = create_res_net()
        self.Fully_Connected_li1 = layers.TimeDistributed(layers.Dense(512, activation='tanh', name='FC_lidar_1'))
        self.Fully_Connected_li2 = layers.TimeDistributed(layers.Dense(512, activation='tanh', name='FC_lidar_2'))
        self.Fully_Connected_li3 = layers.TimeDistributed(layers.Dense(256, activation='tanh', name='FC_lidar_3'))
        self.Fully_Connected_li4 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_lidar_4'))
        self.BN_lidar = layers.TimeDistributed(layers.BatchNormalization())

        self.concatenate = layers.Concatenate(axis=2)
        self.dense_soft_1 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='dense_soft_1'))
        self.dense_soft_2 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='dense_soft_2'))
        # self.act_soft_1 = tf.keras.activations.sigmoid()
        # self.act_soft_2 = tf.keras.activations.sigmoid()
        self.concat_soft_1 = layers.Concatenate(axis=2)

        self.Fully_Connected_all_1 = layers.TimeDistributed(layers.Dense(128, activation='tanh', name='FC_all_1'))
        self.Fully_Connected_all_2 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_all_2'))
        self.RNN_LSTM = layers.LSTM(8,
                                    activation='tanh',
                                    recurrent_activation="sigmoid",
                                    # recurrent_activation="tanh",
                                    # recurrent_dropout=0.2,
                                    unroll=False,
                                    use_bias=True,
                                    name='LSTM_end'
                                    )
        self.MHA = layers.Attention()

    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = True):
        geo_inp, lidar_inp = geo_lidar_input[0], geo_lidar_input[1]

        x1_geo = self.Fully_Connected1(geo_inp)
        geo_branch = self.Fully_Connected2(x1_geo)
        # geo_branch = self.BN_geo(geo_branch)

        Layer_Resnet = self.create_res_net(lidar_inp)
        lidar_branch_1 = self.Fully_Connected_li1(Layer_Resnet)
        lidar_branch_2 = self.Fully_Connected_li2(lidar_branch_1)
        lidar_branch_3 = self.Fully_Connected_li3(lidar_branch_2)
        lidar_branch = self.Fully_Connected_li4(lidar_branch_3)
        # lidar_branch = self.BN_lidar(lidar_branch)

        intint = self.concatenate([geo_branch, lidar_branch])

        SS1 = self.dense_soft_1(intint)
        S1 = tf.keras.activations.sigmoid(SS1)
        SS2 = self.dense_soft_2(intint)
        S2 = tf.keras.activations.sigmoid(SS2)
        AS1 = tf.math.multiply(geo_branch, S1)
        AS2 = tf.math.multiply(lidar_branch, S2)
        conc2 = self.concat_soft_1([AS1, AS2])
        # Fully_Connected = layers.TimeDistributed(layers.Dense(64, activation='tanh'))(conc2)

        intint1 = self.Fully_Connected_all_1(conc2)
        intint2 = self.Fully_Connected_all_2(intint1)
        outp = self.RNN_LSTM(intint2, training=training)

        self.layer_grad0 = tf.gradients(outp[:, 0], intint)
        self.layer_grad1 = tf.gradients(outp[:, 2], intint)
        self.layer_grad2 = tf.gradients(outp[:, 2], intint)
        self.layer_grad3 = tf.gradients(outp[:, 3], intint)
        self.layer_grad4 = tf.gradients(outp[:, 4], intint)
        self.layer_grad5 = tf.gradients(outp[:, 5], intint)
        self.layer_grad6 = tf.gradients(outp[:, 6], intint)
        self.layer_grad7 = tf.gradients(outp[:, 7], intint)

        return outp, geo_branch, lidar_branch, \
            self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3, \
            self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7





# class LiD_s2e_nowe_two(Model):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#         self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3 = None, None, None, None
#         self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7 = None, None, None, None

#         # self.create_res_net = create_res_net(training=False)
#         self.create_res_net = layers.TimeDistributed(create_res_net_NT(training=True))
#         # self.create_res_net = create_res_net_modified()
#         self.Fully_Connected_li1 = layers.TimeDistributed(layers.Dense(512, activation=act_li, name='FC_lidar_1'))
#         self.Fully_Connected_li2 = layers.TimeDistributed(layers.Dense(256, activation=act_li, name='FC_lidar_2'))
#         self.Fully_Connected_li3 = layers.TimeDistributed(layers.Dense(128, activation=act_li, name='FC_lidar_3'))
#         # self.Fully_Connected_li4 = layers.TimeDistributed(layers.Dense(64, activation=act_li, name='FC_lidar_4'))

#         # self.Hidden_state = layers.TimeDistributed(layers.Dense(32, activation=act_li, name='HS'))
#         # self.Hidden_state_2 = layers.TimeDistributed(layers.Dense(32, activation=act_li, name='HS2'))

#         self.Fully_Connected_all_1 = layers.TimeDistributed(layers.Dense(64, activation=act_li, name='FC_all_1'))
#         # self.Fully_Connected_all_2 = layers.TimeDistributed(layers.Dense(128, activation='tanh', name='FC_all_2'))
        
#         # self.RNN_LSTM_1 = layers.Bidirectional(layers.LSTM(64,
#         #                                                    activation=act_lstm,
#         #                                                    recurrent_activation="sigmoid",
#         #                                                    return_sequences=True,
#         #                                                    # recurrent_activation="tanh",
#         #                                                    # recurrent_dropout=0.2,
#         #                                                    unroll=False,
#         #                                                    use_bias=True,
#         #                                                    name='LSTM_end_1'
#         #                                                    ))
#         # self.RNN_LSTM_2 = layers.Bidirectional(layers.LSTM(32,
#         #                                                    activation=act_lstm,
#         #                                                    recurrent_activation="sigmoid",
#         #                                                    # recurrent_activation="tanh",
#         #                                                    # recurrent_dropout=0.2,
#         #                                                    unroll=False,
#         #                                                    use_bias=True,
#         #                                                    name='LSTM_end_2'
#         #                                                    ))
#         # self.Fully_Connected_end = layers.Dense(8,
#         #                                         activation=None,
#         #                                         # use_bias=True,
#         #                                         # bias_regularizer=regularizers.L2(1e-4),
#         #                                         name='FC_all_end')
#         self.RNN_LSTM_0 = layers.LSTM(8,
#                                       activation=None,
#                                       recurrent_activation="sigmoid",
#                                       # recurrent_activation="tanh",
#                                       # recurrent_dropout=0.2,
#                                       unroll=False,
#                                       use_bias=True,
#                                       name='LSTM_end_0'
#                                       )
#     @tf.function
#     def call(self, geo_lidar_input: tf.Tensor, training: bool = False):
#         geo_inp, lidar_inp = geo_lidar_input[0], geo_lidar_input[1]
#         x0_Resnet = self.create_res_net(lidar_inp)
#         x1_Li = self.Fully_Connected_li1(x0_Resnet)
#         x2_Li = self.Fully_Connected_li2(x1_Li)
#         x3_Li = self.Fully_Connected_li3(x2_Li)
#         # x4_Li = self.Fully_Connected_li4(x0_Resnet)

#         # lidar_branch = self.Hidden_state(x4_Li)
#         # lidar_branch = self.Hidden_state_2(lidar_branch)

#         combined = self.Fully_Connected_all_1(x3_Li)
#         # combined = self.Fully_Connected_all_2(combined_1)
        
#         # combined = self.RNN_LSTM_1(x4_Li)
#         # combined = self.RNN_LSTM_2(combined)
#         # outp = self.Fully_Connected_end(combined, training=training)
        
#         outp = self.RNN_LSTM_0(combined)

#         self.layer_grad0 = tf.gradients(outp[:, 0], combined)
#         self.layer_grad1 = tf.gradients(outp[:, 2], combined)
#         self.layer_grad2 = tf.gradients(outp[:, 2], combined)
#         self.layer_grad3 = tf.gradients(outp[:, 3], combined)
#         self.layer_grad4 = tf.gradients(outp[:, 4], combined)
#         self.layer_grad5 = tf.gradients(outp[:, 5], combined)
#         self.layer_grad6 = tf.gradients(outp[:, 6], combined)
#         self.layer_grad7 = tf.gradients(outp[:, 7], combined)

#         return outp, combined, combined, \
#             self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3, \
#             self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7


# ______________________________________________________________________________________________________________________
# Combined Layers weighted
# ______________________________________________________________________________________________________________________

class Combined_weighted(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.der_sig_activation = der_sig_activation()
        self.sig_activation = layers.TimeDistributed(layers.Activation(activation='sigmoid'))
        self.der_tanh_activation = der_tanh_activation()
        self.tanh_activation = layers.TimeDistributed(layers.Activation(activation='tanh'))
        self.bn = layers.TimeDistributed(layers.BatchNormalization())
        self.activaction_tanh = activations.tanh
        self.concatenate = tf.keras.layers.Concatenate(axis=2)
        self.RNN_LSTM = layers.LSTM(8,
                                    activation='tanh',
                                    recurrent_activation="sigmoid",
                                    # recurrent_activation="tanh",
                                    # recurrent_dropout=0.2,
                                    unroll=False,
                                    use_bias=True,
                                    name='LSTM_end'
                                    )
        self.Fully_Connected1 = layers.TimeDistributed(layers.Dense(128, activation='sigmoid', name='FC_geo_2'))

        self.Fully_Connected_w1 = layers.TimeDistributed(
            layers.Dense(64, activation='tanh', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), name='FC_w1'))
        self.Fully_Connected_w2 = layers.TimeDistributed(
            layers.Dense(64, activation='tanh', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), name='FC_w2'))

    @tf.function
    def call(self, intermed_layer: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        w1, int1, w2, int2 = intermed_layer[0], intermed_layer[1], intermed_layer[2], intermed_layer[3]

        ww = self.concatenate([w1, w2])
        ww1 = self.Fully_Connected_w1(ww)
        ww2 = self.Fully_Connected_w2(ww)

        intint = self.concatenate([int1, int2])
        ww12 = self.concatenate([ww1, ww2])
        wx = tf.math.multiply(intint, ww12)

        wx_MLP = self.Fully_Connected1(wx)
        outp = self.RNN_LSTM(wx_MLP, training=training)

        return outp


class Combined_weighted_soft(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.der_sig_activation = der_sig_activation()
        self.sig_activation = layers.TimeDistributed(layers.Activation(activation='sigmoid'))
        self.der_tanh_activation = der_tanh_activation()
        self.tanh_activation = layers.TimeDistributed(layers.Activation(activation='tanh'))
        self.bn = layers.TimeDistributed(layers.BatchNormalization())
        self.activaction_tanh = activations.tanh
        self.concatenate = tf.keras.layers.Concatenate(axis=2)
        self.RNN_LSTM = layers.LSTM(8,
                                    activation='tanh',
                                    recurrent_activation="sigmoid",
                                    # recurrent_activation="tanh",
                                    # recurrent_dropout=0.2,
                                    unroll=False,
                                    use_bias=True,
                                    name='LSTM_end'
                                    )
        self.Fully_Connected1 = layers.TimeDistributed(layers.Dense(128, activation='sigmoid', name='FC_geo_2'))

        self.Fully_Connected_64_1 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_64_1'))
        self.Fully_Connected_64_2 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_64_2'))
        self.Fully_Connected_64_3 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_64_3'))

    @tf.function
    def call(self, intermed_layer: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        w1, int1, w2, int2 = intermed_layer[0], intermed_layer[1], intermed_layer[2], intermed_layer[3]

        intint = self.concatenate([int1, int2])

        SS1 = self.Fully_Connected_64_1(intint)
        S1 = self.sig_activation(SS1)

        SS2 = self.Fully_Connected_64_2(intint)
        S2 = self.sig_activation(SS2)

        AS1 = tf.math.multiply(int1, S1)
        AS2 = tf.math.multiply(int2, S2)

        conc2 = self.concatenate(axis=2)([AS1, AS2])
        Fully_Connected = self.Fully_Connected_64_3(conc2)

        outp = self.RNN_LSTM(Fully_Connected)

        return outp


class Combined_weighted_simple(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.der_sig_activation = der_sig_activation()
        self.sig_activation = layers.TimeDistributed(layers.Activation(activation='sigmoid'))
        self.der_tanh_activation = der_tanh_activation()
        self.tanh_activation = layers.TimeDistributed(layers.Activation(activation='tanh'))
        self.bn = layers.TimeDistributed(layers.BatchNormalization())
        self.activaction_tanh = activations.tanh
        self.concatenate = tf.keras.layers.Concatenate(axis=2)
        self.RNN_LSTM = layers.LSTM(8,
                                    activation='tanh',
                                    recurrent_activation="sigmoid",
                                    # recurrent_activation="tanh",
                                    # recurrent_dropout=0.2,
                                    unroll=False,
                                    use_bias=True,
                                    name='LSTM_end'
                                    )
        self.Fully_Connected1 = layers.TimeDistributed(layers.Dense(128, activation='sigmoid', name='FC_geo_2'))

        self.Fully_Connected_w1 = layers.TimeDistributed(
            layers.Dense(64, activation='tanh', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), name='FC_w1'))
        self.Fully_Connected_w2 = layers.TimeDistributed(
            layers.Dense(64, activation='tanh', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), name='FC_w2'))

    @tf.function
    def call(self, intermed_layer: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        w1, int1, w2, int2 = intermed_layer[0], intermed_layer[1], intermed_layer[2], intermed_layer[3]

        intint = self.concatenate([int1, int2])

        wx_MLP = self.Fully_Connected1(intint)

        outp = self.RNN_LSTM(wx_MLP, training=training)

        return outp


class Combined_weighted_INAF_v1(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.concatenate = tf.keras.layers.Concatenate(axis=2)
        self.RNN_LSTM = layers.LSTM(8,
                                    activation='tanh',
                                    recurrent_activation="sigmoid",
                                    # recurrent_activation="tanh",
                                    # recurrent_dropout=0.2,
                                    unroll=False,
                                    use_bias=True,
                                    name='LSTM_end'
                                    )
        self.Fully_Connected1 = layers.TimeDistributed(layers.Dense(128, activation='sigmoid', name='FC_geo_2'))

        self.Fully_Connected_64_1 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_64_1'))
        self.Fully_Connected_64_2 = layers.TimeDistributed(layers.Dense(128, activation='tanh', name='FC_64_2'))
        self.Fully_Connected_64_3 = layers.TimeDistributed(layers.Dense(64, activation='tanh', name='FC_64_3'))

        #  _______________________________________________heads,
        self.MHA = layers.TimeDistributed(MultiHeadAttention(8, 128, 128, 128))

    @tf.function
    def call(self, intermed_layer: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        w1, int1, w2, int2 = intermed_layer[0], intermed_layer[1], intermed_layer[2], intermed_layer[3]

        intint = self.concatenate([int1, int2])
        w1w2 = self.concatenate([w1, w2])

        MHA_out = self.MHA(intint, w1w2, intint)  # queries, keys, values

        Fully_Connected1 = self.Fully_Connected_64_1(MHA_out)
        Fully_Connected2 = self.Fully_Connected_64_2(Fully_Connected1)
        Fully_Connected = self.Fully_Connected_64_3(Fully_Connected2)

        outp = self.RNN_LSTM(Fully_Connected)

        return outp


# _____________________________________
# Blocks
# _____________________________________
class der_sig_activation(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.TD_activ = layers.TimeDistributed(layers.Activation(activation='sigmoid'))
        self.TD_mul = layers.TimeDistributed(layers.Multiply())
        self.TD_subt = layers.TimeDistributed(layers.Subtract())

    @tf.function
    def call(self, inputs):
        activ_fun = self.TD_activ(inputs)
        # subtracted = self.TD_subt([tf.ones((4,128), dtype='float32'), sigmoid])
        # kones = tf.ones(sigmoid.shape)
        # subtracted = tf.math.subtract(kones, sigmoid)
        subtracted = layers.Lambda(lambda x: x * (1. - x))(activ_fun)
        # activ = self.TD_mul([sigmoid, subtracted])
        activ = layers.Multiply()([activ_fun, subtracted])
        return activ


class der_tanh_activation(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.TD_activ = layers.TimeDistributed(layers.Activation(activation='tanh'))
        self.TD_mul = layers.TimeDistributed(layers.Multiply())
        self.TD_subt = layers.TimeDistributed(layers.Subtract())

    @tf.function
    def call(self, inputs):
        activ_fun = self.TD_activ(inputs)
        # subtracted = self.TD_subt([tf.ones((4,128), dtype='float32'), sigmoid])
        # kones = tf.ones(sigmoid.shape)
        # subtracted = tf.math.subtract(kones, sigmoid)
        subtracted = layers.Lambda(lambda x: 1. - x ** 2)(activ_fun)
        # activ = self.TD_mul([sigmoid, subtracted])
        activ = layers.Multiply()([activ_fun, subtracted])
        return activ


# _______________________________Resnet
class residual_block(layers.Layer):

    def __init__(self, downsample, filters, kernel_size, name, is_training=True, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.TD_conv2d_1 = layers.TimeDistributed(layers.Conv2D(kernel_size=kernel_size,
                                                                strides=(1 if not downsample else 2),
                                                                # strides=strides,
                                                                filters=filters,
                                                                padding='same',
                                                                name=name + 'res_block_conv_1'))
        self.TD_conv2d_2 = layers.TimeDistributed(layers.Conv2D(kernel_size=kernel_size,
                                                                strides=1,
                                                                filters=filters,
                                                                padding='same',
                                                                name=name + 'res_block_conv_2'))
        self.TD_conv2d_ds = layers.TimeDistributed(layers.Conv2D(kernel_size=1,
                                                                 strides=2,
                                                                 # strides=strides,
                                                                 filters=filters,
                                                                 padding='same',
                                                                 name=name + 'res_block_conv_3'))
        self.downsample = downsample

        self.act_BN_1 = Relu_BN(name=name + 'res_block_relu_1', training=is_training)
        # self.act_1 = layers.TimeDistributed(layers.ReLU(name=name + 'res_block_relu_1'))

        # self.TD_BN_2 = layers.TimeDistributed(layers.BatchNormalization(name='BN_2', momentum=momentum_p))
        # self.act_BN_2 = Relu_BN(name=name + 'res_block_relu_2')
        # self.TD_BN_2 = layers.TimeDistributed(layers.BatchNormalization(name='BN_2'))

        # self.TD_BN_ds = layers.TimeDistributed(layers.BatchNormalization(name='BN_ds'))
        # self.act_BN_ds = Relu_BN(name=name + 'res_block_relu_ds')

        # self.act_end = layers.TimeDistributed(layers.ReLU(name='ReLU_end'))
        self.act_BN_end = Relu_BN(name=name + 'res_block_relu_end', training=is_training)

        # self.act_BN_2 = Tanh_BN1(name=name + 'res_block_relu_2')

    @tf.function
    def call(self, in_x):

        y = self.TD_conv2d_1(in_x)
        # y = self.act_1(y)
        y = self.act_BN_1(y)

        y = self.TD_conv2d_2(y)
        # y = self.act_BN_2(y)
        # y = self.TD_BN_2(y)

        if self.downsample:
            in_x_ds = self.TD_conv2d_ds(in_x)
            # in_x_ds = self.act_BN_ds(in_x_ds)
            # in_x_ds = self.TD_BN_ds(in_x_ds)
            out = layers.Add()([in_x_ds, y])
        else:
            out = layers.Add()([in_x, y])

        out = self.act_BN_end(out)   
        # out = self.act_end(out)  # doesn't converge

        return out


class create_res_net(layers.Layer):
    def __init__(self, num_filters=64, training=True, **kwargs):
        super().__init__(**kwargs)
        self.mode_res = training
        self.num_filters = num_filters

        # self.TD_BN = layers.TimeDistributed(layers.BatchNormalization(name='res_beg', momentum=momentum_p, trainable=self.mode_res))
        self.TD_BN = layers.BatchNormalization(name='res_beg', momentum=momentum_p)
        self.drop_1 = layers.TimeDistributed(tf.keras.layers.Dropout(rate=0.2, name='drop_values'))

        self.TD_con2d_f = layers.TimeDistributed(layers.Conv2D(kernel_size=3,
                                                               strides=(1, 2),
                                                               filters=self.num_filters,
                                                               padding='valid',
                                                               name='resnet_conv_1'))
        
        # self.TD_BN_1 = layers.TimeDistributed(layers.BatchNormalization(name='Batch_N_1', trainable=self.mode_res))
        # self.TD_BN_1 = layers.BatchNormalization(name='Batch_N_1')
        # self.TD_ReLU_1 = layers.TimeDistributed(layers.ReLU(name='resnet_ReLU_1'))

        self.TD_maxp = layers.TimeDistributed(layers.MaxPool2D(pool_size=3,
                                                               strides=(1, 2),
                                                               padding='valid',
                                                               name='resnet_maxp_1'))
        # self.act_BN = Tanh_BN1(name='resnet_relu')

        self.act_BN = Relu_BN(name='resnet_act_beg')
        
        self.residual_block_1 = residual_block(downsample=False, filters=64, kernel_size=3, name='res1', is_training=self.mode_res)
        self.residual_block_2 = residual_block(downsample=False, filters=64, kernel_size=3, name='res2', is_training=self.mode_res)

        self.residual_block_3 = residual_block(downsample=True, filters=128, kernel_size=3, name='res3', strides=(1, 2), is_training=self.mode_res)
        self.residual_block_4 = residual_block(downsample=False, filters=128, kernel_size=3, name='res4', is_training=self.mode_res)

        self.residual_block_5 = residual_block(downsample=True, filters=256, kernel_size=3, name='res5', strides=(1, 2), is_training=self.mode_res)
        self.residual_block_6 = residual_block(downsample=False, filters=256, kernel_size=3, name='res6', is_training=self.mode_res)
        # self.drop_2 = layers.TimeDistributed(tf.keras.layers.SpatialDropout2D(rate=0.2, name='drop_channel'))

        self.residual_block_7 = residual_block(downsample=True, filters=512, kernel_size=3, name='res7', strides=(2, 2), is_training=self.mode_res)
        self.residual_block_8 = residual_block(downsample=False, filters=512, kernel_size=3, name='res8', is_training=self.mode_res)

        self.TD_APool = layers.TimeDistributed(layers.AveragePooling2D(strides=(8, 23), pool_size=(8, 23), name='resnet_AP'))
        # self.act_BN = Relu_BN(name='resnet_relu')
        # self.TD_APool = layers.TimeDistributed(layers.GlobalAveragePooling2D(name='resnet_AP'))

        self.TD_F = layers.TimeDistributed(layers.Flatten())

        self.TD_Dense = layers.TimeDistributed(layers.Dense(1000, activation='softmax', name='resnet_dense'))

    @tf.function
    def call(self, in_x):
        t = self.TD_BN(in_x)
        # print(tf.shape(t))
        # if self.mode_res == 'training':
        #     t = self.drop_1(t)

        t = tf.pad(t, tf.constant([[0, 0], [0, 0], [0, 0], [1, 1], [0, 0]]), mode='REFLECT')
        t = tf.pad(t, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]), mode='CONSTANT')
        t = self.TD_con2d_f(t)
        
        # t = self.TD_BN_1(t)
        # t = self.TD_ReLU_1(t)

        t = tf.pad(t, tf.constant([[0, 0], [0, 0], [0, 0], [1, 1], [0, 0]]), mode='REFLECT')
        t = tf.pad(t, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]), mode='CONSTANT')
        t = self.TD_maxp(t)
        
        t = self.act_BN(t)
        # _____________________________________________________________________________________________________8 Res Net

        t1 = self.residual_block_1(t)
        t1 = self.residual_block_2(t1)

        t2 = self.residual_block_3(t1)
        t2 = self.residual_block_4(t2)

        t3 = self.residual_block_5(t2)
        t3 = self.residual_block_6(t3)
        # if self.mode_res == 'training':
        #     t3 = self.drop_2(t3)

        t4 = self.residual_block_7(t3)
        t4 = self.residual_block_8(t4)

        # t = self.act_BN(t)
        t = self.TD_APool(t4)
        t = self.TD_F(t)


        outputs = self.TD_Dense(t)


        return outputs


class Relu_BN(layers.Layer):
    def __init__(self, name, training=True, **kwargs):
        super().__init__(**kwargs)

        # self.TD_relu = layers.TimeDistributed(layers.LeakyReLU(alpha=0.3, name=name + 'L_relu'))
        
        # self.TD_bn = layers.TimeDistributed(layers.BatchNormalization(name=name + 'BN', trainable=training))
        self.TD_bn = layers.BatchNormalization(name=name + 'BN')
        self.TD_relu = layers.TimeDistributed(layers.ReLU(name=name + 'relu'))
        
        # self.TD_bn = layers.TimeDistributed(layers.BatchNormalization(name=name + 'BN', momentum=momentum_p))

    @tf.function
    def call(self, inputs):
        bn = self.TD_bn(inputs)
        bn_relu = self.TD_relu(bn)
        return bn_relu
    
#______________________Resnet No time distributed
class residual_block_NT(layers.Layer):

    def __init__(self, downsample, filters, name, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.TD_conv2d_1 = layers.Conv2D(kernel_size=3,
                                         strides=(1 if not downsample else 2),
                                         # strides=strides,
                                         filters=filters,
                                         padding='same',
                                         name=name + 'res_block_conv_1')
        self.TD_conv2d_2 = layers.Conv2D(kernel_size=3,
                                         # strides=1,
                                         filters=filters,
                                         padding='same',
                                         name=name + 'res_block_conv_2')
        self.TD_conv2d_ds = layers.Conv2D(kernel_size=1,
                                          strides=2,
                                          filters=filters,
                                          padding='same',
                                          name=name + 'res_block_conv_3')
        self.downsample = downsample

        self.act_BN_1 = Relu_BN_NT(name=name + 'res_block_Relu_BN_1')
        self.TD_BN_2 = layers.TimeDistributed(layers.BatchNormalization(name='BN_2'))
        self.act_end = layers.TimeDistributed(layers.ReLU(name='ReLU_end'))


    @tf.function
    def call(self, in_x, training=False):

        y = self.TD_conv2d_1(in_x)
        y = self.act_BN_1(y, training=training)

        y = self.TD_conv2d_2(y)
        y = self.TD_BN_2(y, training=training)

        if self.downsample:
            in_x_ds = self.TD_conv2d_ds(in_x)
            out = layers.Add()([in_x_ds, y])
        else:
            out = layers.Add()([in_x, y])
 
        out = self.act_end(out, training=training) 

        return out


class create_res_net_NT(layers.Layer):
    def __init__(self, num_filters=64, training=True, **kwargs):
        super().__init__(**kwargs)
        self.mode_res = training
        self.num_filters = num_filters

        self.BN_0 = layers.BatchNormalization(name='resnet_BN_0', momentum=momentum_p)
        # self.drop_1 =tf.keras.layers.Dropout(rate=0.2, name='drop_values')

        self.TD_con2d_0 = layers.Conv2D(kernel_size=3,
                                        strides=(1, 2),
                                        filters=self.num_filters,
                                        padding='valid',
                                        name='resnet_conv_1')
        
        # self.TD_BN_1 = layers.TimeDistributed(layers.BatchNormalization(name='Batch_N_1', trainable=self.mode_res))
        # self.BN_1 = layers.BatchNormalization(name='Batch_N_1')
        # self.ReLU_1 = layers.ReLU(name='resnet_ReLU_1')

        self.maxp_0 = layers.MaxPool2D(pool_size=3,
                                        strides=(1, 2),
                                        padding='valid',
                                        name='resnet_maxp_1')
        
        # self.act_BN = Tanh_BN1(name='resnet_relu')
        self.act_BN_beg = Relu_BN(name='resnet_Relu_BN_beg')
        
        self.residual_block_1 = residual_block_NT(downsample=False, filters=64, name='res1', is_training=self.mode_res)
        self.residual_block_2 = residual_block_NT(downsample=False, filters=64, name='res2', is_training=self.mode_res)

        self.residual_block_3 = residual_block_NT(downsample=True, filters=128, name='res3', strides=(1, 2), is_training=self.mode_res)
        self.residual_block_4 = residual_block_NT(downsample=False, filters=128, name='res4', is_training=self.mode_res)

        self.residual_block_5 = residual_block_NT(downsample=True, filters=256, name='res5', strides=(1, 2), is_training=self.mode_res)
        self.residual_block_6 = residual_block_NT(downsample=False, filters=256, name='res6', is_training=self.mode_res)
        # self.drop_2 = layers.TimeDistributed(tf.keras.layers.SpatialDropout2D(rate=0.2, name='drop_channel'))

        self.residual_block_7 = residual_block_NT(downsample=True, filters=512, name='res7', strides=(2, 2), is_training=self.mode_res)
        self.residual_block_8 = residual_block_NT(downsample=False, filters=512, name='res8', is_training=self.mode_res)

        self.APool = layers.AveragePooling2D(strides=(8, 23), pool_size=(8, 23), name='resnet_AP')
        # self.act_BN = Relu_BN(name='resnet_relu_BN')
        # self.TD_APool = layers.TimeDistributed(layers.GlobalAveragePooling2D(name='resnet_AP'))

        self.end_F = layers.Flatten()
        # self.TD_Dense = layers.Dense(1000, activation='softmax', name='resnet_dense')
        self.TD_Dense = layers.Dense(1000, activation='relu', name='resnet_dense')
        
                      

    @tf.function
    def call(self, in_x):
        t = self.BN_0(in_x)
        # print(tf.shape(t))
        # if self.mode_res == 'training':
        #     t = self.drop_1(t)

        t = tf.pad(t, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]), mode='REFLECT')
        t = tf.pad(t, tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]]), mode='CONSTANT')
        t = self.TD_con2d_0(t)
        
        # t = self.BN_1(t)
        # t = self.ReLU_1(t)

        t = tf.pad(t, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]), mode='REFLECT')
        t = tf.pad(t, tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]]), mode='CONSTANT')
        t = self.maxp_0(t)

        t = self.act_BN_beg(t)

        # _____________________________________________________________________________________________________8 Res Net

        t1 = self.residual_block_1(t)
        t1 = self.residual_block_2(t1)

        t2 = self.residual_block_3(t1)
        t2 = self.residual_block_4(t2)

        t3 = self.residual_block_5(t2)
        t3 = self.residual_block_6(t3)
        # if self.mode_res == 'training':
        #     t3 = self.drop_2(t3)

        t4 = self.residual_block_7(t3)
        t4 = self.residual_block_8(t4)

        # t = self.act_BN(t)
        t = self.APool(t4)
        t = self.end_F(t)


        outputs = self.TD_Dense(t)


        return outputs
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1000)


class Relu_BN_NT(layers.Layer):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)

        # self.TD_relu = layers.TimeDistributed(layers.LeakyReLU(alpha=0.3, name=name + 'L_relu'))
        
        # self.TD_bn = layers.TimeDistributed(layers.BatchNormalization(name=name + 'BN', trainable=training))
        self.TD_bn = layers.BatchNormalization(name=name + 'BN')
        self.TD_relu = layers.ReLU(name=name + 'relu')
        
        # self.TD_bn = layers.TimeDistributed(layers.BatchNormalization(name=name + 'BN', momentum=momentum_p))

    @tf.function
    def call(self, inputs, training=False):
        bn = self.TD_bn(inputs, training=training)
        bn_relu = self.TD_relu(bn)
        return bn_relu

    
class create_res_net_NT_one(layers.Layer):
    def __init__(self, num_filters=64, training=True, 
                 filters1=64, 
                 filters2=128, filters3=256, filters4=512, 
                 dense_last=1000, act_last='relu', **kwargs):
        super().__init__(**kwargs)
        
        self.dense_last = dense_last

        self.BN_0 = layers.BatchNormalization(name='resnet_BN_0', momentum=momentum_p)

        self.TD_con2d_0 = layers.Conv2D(kernel_size=3,
                                        strides=(1, 2),
                                        filters=num_filters,
                                        padding='valid',
                                        name='resnet_conv_1')
        

        self.maxp_0 = layers.MaxPool2D(pool_size=3,
                                        strides=(1, 2),
                                        padding='valid',
                                        name='resnet_maxp_1')
        
        self.act_BN_beg = Relu_BN(name='resnet_Relu_BN_beg')
        
        self.residual_block_1 = residual_block_NT(downsample=False, filters=filters1, name='res1')
        self.residual_block_2 = residual_block_NT(downsample=False, filters=filters1, name='res2')

        self.residual_block_3 = residual_block_NT(downsample=True, filters=filters2, name='res3', strides=(1, 2))
        self.residual_block_4 = residual_block_NT(downsample=False, filters=filters2, name='res4')

        self.residual_block_5 = residual_block_NT(downsample=True, filters=filters3, name='res5', strides=(1, 2))
        self.residual_block_6 = residual_block_NT(downsample=False, filters=filters3, name='res6')

        self.residual_block_7 = residual_block_NT(downsample=True, filters=filters4, name='res7', strides=(2, 2))
        self.residual_block_8 = residual_block_NT(downsample=False, filters=filters4, name='res8')

        self.APool = layers.AveragePooling2D(strides=(8, 23), pool_size=(8, 23), name='resnet_AP')
        
        self.end_F = layers.Flatten()
        self.TD_Dense = layers.Dense(dense_last, activation=act_last, name='resnet_dense')
  
                      

    @tf.function
    def call(self, in_x, training=False):
        t = self.BN_0(in_x, training=training)

        t = tf.pad(t, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]), mode='REFLECT')
        t = tf.pad(t, tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]]), mode='CONSTANT')
        t = self.TD_con2d_0(t)

        t = tf.pad(t, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]), mode='REFLECT')
        t = tf.pad(t, tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]]), mode='CONSTANT')
        t = self.maxp_0(t)

        t = self.act_BN_beg(t, training=training)

        # _____________________________________________________________________________________________________8 Res Net

        t1 = self.residual_block_1(t, training=training)
        t1 = self.residual_block_2(t1, training=training)

        t2 = self.residual_block_3(t1, training=training)
        t2 = self.residual_block_4(t2, training=training)

        t3 = self.residual_block_5(t2, training=training)
        t3 = self.residual_block_6(t3, training=training)

        t4 = self.residual_block_7(t3, training=training)
        t4 = self.residual_block_8(t4, training=training)

        t = self.APool(t4)
        t = self.end_F(t)

        outputs = self.TD_Dense(t, training=training)

        return outputs
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dense_last)
#_______________________________________________________________________________________________________________________


class Tanh_BN1(layers.Layer):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.TD_tanh = layers.TimeDistributed(layers.Activation(activations.tanh, name=name))
        self.TD_bn = layers.TimeDistributed(layers.BatchNormalization())

    @tf.function
    def call(self, inputs):
        relu = self.TD_tanh(inputs)
        bn = self.TD_bn(relu)
        return bn




class tanh_BN(layers.Layer):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        # self.TD_tanh = layers.TimeDistributed(layers.Activation(activations.tanh, name=name))
        # self.TD_bn = layers.TimeDistributed(layers.BatchNormalization())
        self.TD_tanh = layers.Activation(activations.tanh, name=name)
        self.TD_bn = layers.BatchNormalization()

    @tf.function
    def call(self, inputs):
        actv = self.TD_tanh(inputs)
        bn = self.TD_bn(actv)
        return bn

    
#______________________________________________________
#NewGP
#_______________________________________________________

class ResnetBlock_GP(layers.Layer):
    def __init__(self, filters, stride=1):
        super(ResnetBlock_GP, self).__init__()
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.residual = layers.Conv2D(filters, 1, strides=stride)
        else:
            self.residual = lambda x: x

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        res = self.residual(inputs)
        x = layers.add([x, res])
        return self.relu(x)

class CustomResnet_GP(layers.Layer):
    def __init__(self):
        super(CustomResnet_GP, self).__init__()
        self.conv1 = layers.Conv2D(64, 7, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.maxpool = layers.MaxPooling2D(3, strides=2, padding='same')
        
        self.block1 = ResnetBlock_GP(64)
        self.block2 = ResnetBlock_GP(128, stride=2)
        self.block3 = ResnetBlock_GP(256, stride=2)
        self.block4 = ResnetBlock_GP(512, stride=2)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(1000)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        return self.fc(x)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1000)

    
    
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the ResNet model for feature extraction
        self.resnet = tf.keras.applications.ResNet50(
            include_top=False, weights=None, input_shape=input_shape)
        # Define the LSTM layer for temporal modeling
        self.lstm = layers.LSTM(64, return_sequences=True)
        # Define the dense layers for regression
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(output_size)

    def call(self, inputs):
        # Take two images as inputs
        input1, input2 = inputs
        # Extract features using ResNet for each image separately
        features1 = self.resnet(input1)
        features2 = self.resnet(input2)
        # Flatten the features
        features1 = layers.Flatten()(features1)
        features2 = layers.Flatten()(features2)
        # Concatenate the features along the last dimension
        features = layers.Concatenate(axis=-1)([features1, features2])
        # Add a temporal dimension
        features = tf.expand_dims(features, axis=1)
        # Apply LSTM
        outputs = self.lstm(features)
        # Apply dense layers
        outputs = self.dense1(outputs)
        outputs = self.dense2(outputs)
        # Return the outputs
        return outputs
    
    
#___________________________________________________________________________________________________________________
#Attention 
#___________________________________________________________________________________________________________________

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention", **kwargs):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = layers.Dense(d_model)
        self.key_dense = layers.Dense(d_model)
        self.value_dense = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value = inputs
        batch_size = tf.shape(query)[0]

        # Linear projections
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value)

        # Concatenate heads and pass through final dense layer
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)

        return output, attention_weights

    

class AttentionLayer_Simplified(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer_Simplified, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttentionLayer_Simplified, self).build(input_shape)

    def call(self, inputs, training=None):
        query, key = inputs

        # Scaled Dot-Product Attention
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = tf.math.divide(attention_scores, tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32)))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attended_values = tf.matmul(attention_weights, key)

        return attended_values

    def compute_output_shape(self, input_shape):
        return input_shape[0]