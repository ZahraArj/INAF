import tensorflow as tf


def distance_matrix(array1, array2):
    """
    arguments:
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    num_point, num_features = array1.shape
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
        tf.tile(tf.expand_dims(array2, 1),
                (1, num_point, 1)),
        (-1, num_features))
    distances = tf.norm(expanded_array1 - expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances


def av_dist(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix(array1, array2)
    distances = tf.reduce_min(distances, axis=1)
    distances = tf.reduce_mean(distances)
    return distances


def av_dist_sum(arrays):
    """
    arguments:
        arrays: array1, array2
    returns:
        sum of av_dist(array1, array2) and av_dist(array2, array1)
    """
    array1, array2 = arrays
    av_dist1 = av_dist(array1, array2)
    av_dist2 = av_dist(array2, array1)
    return av_dist1 + av_dist2


def chamfer_distance_tf(array1, array2):
    batch_size, num_point, num_features = array1.shape
    array1 = tf.cast(array1, tf.float64)
    array2 = tf.cast(array2, tf.float64)

    dist = tf.reduce_mean(
        tf.map_fn(av_dist_sum, elems=(array1, array2), dtype=tf.float64)
    )
    return dist
