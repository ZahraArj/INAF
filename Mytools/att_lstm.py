import tensorflow as tf
from tensorflow.keras import layers


def attention(inputs):
    # Trainable parameters
    hidden_size = inputs.shape.as_list()[-1]
    # u_omega = tf.Variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    u_omega = tf.Variable(initializer(shape=(hidden_size, 1)), name='u_omega')

    with tf.name_scope('v'):
        v = tf.tanh(inputs)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    print('v', v.shape)
    print('u', u_omega.shape)
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    print(vu.shape)
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
    print(alphas.shape)

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    # output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    output = tf.reduce_sum(inputs * alphas, 1)

    # Final output with tanh
    output = tf.tanh(output)

    output = layers.Dense(8, activation='softmax')(output)

    return output
