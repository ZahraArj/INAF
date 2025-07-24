
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, backend, losses

def relu_bn(inputs, name, training=True):
    x = layers.BatchNormalization(name=name + '_BN')(inputs)
    x = layers.ReLU(name=name + '_RELU')(x)
    return x

def create_res_net(inputs, name, num_filters=64, training=True):
    
    momentum_p = 0.9

    x = layers.BatchNormalization(name=name+'res_net_BN1', momentum=momentum_p)(inputs)

    x = tf.pad(x, tf.constant([[0, 0], [0, 0], [0, 0], [1, 1], [0, 0]]), mode='REFLECT')
    x = tf.pad(x, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]), mode='CONSTANT')
    x = layers.TimeDistributed(layers.Conv2D(kernel_size=3, 
                                             strides=(1, 2), 
                                             filters=num_filters, 
                                             padding='valid', 
                                             name=name+'res_net_conv1'))(x)

    x = tf.pad(x, tf.constant([[0, 0], [0, 0], [0, 0], [1, 1], [0, 0]]), mode='REFLECT')
    x = tf.pad(x, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]), mode='CONSTANT')
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=3, 
                                                strides=(1, 2), 
                                                padding='valid', 
                                                name=name+'res_net_maxp1'))(x)

    x = relu_bn(x, name=name+'res_net_RB', training=training)

    # 8 Res Net
    x1 = residual_block(x, downsample=False, filters=64, name=name+'res1', is_training=training)
    x1 = residual_block(x1, downsample=False, filters=64, name=name+'res2', is_training=training)

    x2 = residual_block(x1, downsample=True, filters=128, name=name+'res3', strides=(1, 2), is_training=training)
    x2 = residual_block(x2, downsample=False, filters=128, name=name+'res4', is_training=training)

    x3 = residual_block(x2, downsample=True, filters=256, name=name+'res5', strides=(1, 2), is_training=training)
    x3 = residual_block(x3, downsample=False, filters=256, name=name+'res6', is_training=training)

    x4 = residual_block(x3, downsample=True, filters=512, name=name+'res7', strides=(2, 2), is_training=training)
    x4 = residual_block(x4, downsample=False, filters=512, name=name+'res8', is_training=training)

    x = layers.TimeDistributed(layers.AveragePooling2D(strides=(8, 23), pool_size=(8, 23), name=name+'res_net_avgP'))(x4)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed(layers.Dense(1000, 
                                            activation='relu', 
                                            name=name+'res_net_dense1'))(x)

    return x

def residual_block(inputs, downsample, filters, name, is_training=True, strides=1):
    x =layers.TimeDistributed(layers.Conv2D(kernel_size=3, 
                                            strides=(1 if not downsample else 2), 
                                            filters=filters,
                                            padding='same',
                                            name=name + '_conv1'))(inputs)
    
    x = relu_bn(x, name=name + '_RB_beg', training=is_training)

    x = layers.TimeDistributed(layers.Conv2D(kernel_size=3,
                                             filters=filters,
                                             padding='same',
                                             name=name + '_conv2'))(x)
    x = layers.BatchNormalization(name=name + '_BN2')(x)

    if downsample:
        in_x_ds = layers.TimeDistributed(layers.Conv2D(kernel_size=1, 
                                strides=2, 
                                filters=filters,
                                padding='same', 
                                name=name + '_conv3'))(inputs)
        out = layers.Add()([in_x_ds, x])
    else:
        out = layers.Add()([inputs, x])

    out = layers.ReLU(name=name + '_ReLU_end')(out)

    return out

def model_LiDar_v1(inputs_dict, training=True):
    geo_inp, lidar_inp = inputs_dict['geo_input'], inputs_dict['AI_input']

    inputs1 = create_res_net(lidar_inp[..., :7], name='RESNET1')
    inputs2 = create_res_net(lidar_inp[..., 7:], name='RESNET2')
    # inputs1 = layers.TimeDistributed(layers.Lambda(create_res_net_NT_one))(lidar_inp[..., :7])
    # inputs1 = layers.TimeDistributed(layers.Lambda(create_res_net_NT_one))(lidar_inp[..., 7:])

    x1_Resnet = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(inputs1)
    x2_Resnet = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(inputs2)

    x12 = layers.Concatenate(name='concat_layer0')([x1_Resnet, x2_Resnet])

    x12 = layers.Dense(1024, activation='relu', name='dense1')(x12)
    x12 = layers.Dense(512, activation='relu', name='dense2')(x12)
    outp = layers.Dense(8, name='dense3')(x12)

    return outp



def model_LiDar_grad_v1(inputs_dict, training=True):
    with tf.GradientTape() as tape:
        geo_inp, lidar_inp = inputs_dict['geo_input'], inputs_dict['AI_input']

        inputs1 = create_res_net(lidar_inp[..., :7], name='RESNET1')
        inputs2 = create_res_net(lidar_inp[..., 7:], name='RESNET2')
        # inputs1 = layers.TimeDistributed(layers.Lambda(create_res_net_NT_one))(lidar_inp[..., :7])
        # inputs1 = layers.TimeDistributed(layers.Lambda(create_res_net_NT_one))(lidar_inp[..., 7:])

        x1_Resnet = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(inputs1)
        x2_Resnet = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(inputs2)


        x12 = layers.Concatenate(name='concat_layer0')([x1_Resnet, x2_Resnet])
        
        x12 = layers.Dense(1024, activation='relu', name='dense1')(x12)
        with tf.GradientTape() as t:
            t.watch(x12)
            x12 = layers.Dense(512, activation='relu', name='dense2')(x12)
            outp = layers.Dense(8, name='dense3')(x12)   
        
        grad = t.gradient(outp, x12)
        
        result = output
        # gradients = t.gradient(output, x_tensor)
    
    return outp, x12


