from tensorflow.keras import layers
import tensorflow.keras.backend as K

# from: https://github.com/kobiso/CBAM-keras
def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    print('vs', input_feature.shape, input_feature._shape_val)
    print('channel_input', input_feature.shape)
    channel = input_feature.shape[-1]

    shared_layer_one = layers.TimeDistributed(layers.Dense(channel // ratio,
                                                           activation='relu',
                                                           kernel_initializer='he_normal',
                                                           use_bias=True,
                                                           bias_initializer='zeros'))
    shared_layer_two = layers.TimeDistributed(layers.Dense(channel,
                                                           kernel_initializer='he_normal',
                                                           use_bias=True,
                                                           bias_initializer='zeros'))

    avg_pool = layers.TimeDistributed(layers.GlobalAveragePooling2D())(input_feature)
    avg_pool = layers.Reshape((4, 1, 1, channel))(avg_pool)

    print("avg_pool", avg_pool.shape)
    assert avg_pool._shape_val[2:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._shape_val[2:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._shape_val[2:] == (1, 1, channel)

    max_pool = layers.TimeDistributed(layers.GlobalMaxPooling2D())(input_feature)
    max_pool = layers.Reshape((4, 1, 1, channel))(max_pool)

    print("max_pool", max_pool.shape)
    assert max_pool._shape_val[2:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._shape_val[2:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._shape_val[2:] == (1, 1, channel)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.TimeDistributed(layers.Activation('sigmoid'))(cbam_feature)

    return layers.Multiply()([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7
    # channel = input_feature._keras_shape[-1]
    cbam_feature = input_feature

    avg_pool = layers.TimeDistributed(layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True)))(cbam_feature)
    assert avg_pool._shape_val[-1] == 1
    max_pool = layers.TimeDistributed(layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True)))(cbam_feature)
    assert max_pool._shape_val[-1] == 1
    concat = layers.Concatenate(axis=4)([avg_pool, max_pool])
    print('concat', concat.shape)
    assert concat._shape_val[-1] == 2

    cbam_feature = layers.TimeDistributed(layers.Conv2D(filters=1,
                                                        kernel_size=kernel_size,
                                                        strides=1,
                                                        padding='same',
                                                        activation='sigmoid',
                                                        kernel_initializer='he_normal',
                                                        use_bias=False))(concat)
    assert cbam_feature._shape_val[-1] == 1

    print('mp2', input_feature.shape, cbam_feature.shape)

    return layers.multiply([input_feature, cbam_feature])
