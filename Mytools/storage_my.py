'''

@tf.custom_gradient
    def li_loss6(self, inputs, output):  # output: x y z i j k w
        pc1_raw, pc2_raw = tf.split(inputs, num_or_size_splits=2, axis=4)
        # B x T x W x H x Channels
        s0, s1, s2, s3, s4 = pc1_raw.shape[0], pc1_raw.shape[1], pc1_raw.shape[2], pc1_raw.shape[3], pc1_raw.shape[4]

        pc1 = tf.reshape(pc1_raw[:, -1, :, :, 0:3], shape=[-1, s2 * s3, 3])
        pc2 = tf.reshape(pc2_raw[:, -1, :, :, 0:3], shape=[-1, s2 * s3, 3])

        Rq, Tr3 = tfg.dual_quaternion.to_rotation_translation(output)
        R33 = tfg.rotation_matrix_3d.from_quaternion(Rq)
        RT = tf.concat([R33, tf.expand_dims(Tr3, axis=2)], -1)
        RT = tf.pad(RT, [[0, 0], [0, 1], [0, 0]], constant_values=[0.0, 0.0, 0.0, 1.0])

        pc1 = tf.pad(pc1, [[0, 0], [0, 0], [0, 1]], constant_values=1)
        pc1 = tf.transpose(pc1, perm=[0, 2, 1])
        pc1_tr = tf.linalg.matmul(RT, pc1)
        pc1_tr = pc1_tr[:, 0:3]
        pc1_tr = tf.transpose(pc1_tr, perm=[0, 2, 1])  # B x WH x 3

        cham_dist = chamfer_distance_tf(pc2, pc1_tr)
        print(cham_dist)
        dist_p2p, ind_all = KD_Tree(pc2, pc1_tr, s1, 500, s4, s3)


        pc2_g = tf.gather(pc2, ind_all, batch_dims=1)

        # dist_all = tf.add(dist_p2p, dist_p2pl)
        dist_all = dist_p2p

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

                dist_all_c = d_p2p

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



def KD_Tree(pc2, pc1_tr, s1, B, H, W):
    dist_p2p = np.zeros(B, dtype=float)
    ind_all = np.zeros([s1, H * W])
    for i in range(B):
        pc2i = pc2[i]
        print(pc2i.shape)
        tree2 = cKDTree(pc2i, leafsize=500, balanced_tree=False)
        dist_in, ind = tree2.query(pc1_tr[i], k=1)
        ind_all[i, :] = ind
        nonempty = np.count_nonzero(dist_in)
        dist_in = np.sum(np.abs(dist_in))
        if nonempty != 0:
            dist_in = np.divide(dist_in, nonempty)
        dist_p2p[i] = dist_in

    return dist_p2p, ind_all
















# class Grad_CallBack(tf.keras.callbacks.Callback):
#     def __init__(self):
#         super().__init__()
#         self.outputs = None
#
#     def on_predict_batch_end(self, batch, logs=None):
#         Base = BaseNet
#         model = Base.geo_model
#         self.outputs = get_gradient_func(model)
#         print('inside', self.outputs)
#
#
# def get_gradient_func(model):
#     layer_g = model.layer_grad
#     return layer_g














# #  _____________________________________________________________________________________________________________
# # Make Model
# #  _____________________________________________________________________________________________________________
# model = my_custom_model(self.Geo_model, self.Lidar_model, self.fusion)
#
# #  _____________________________________________________________________________________________________________
# # Compile Model
# #  _____________________________________________________________________________________________________________
# model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam', loss_weights=100)
#
# #  _____________________________________________________________________________________________________________
# # Save Model
# #  _____________________________________________________________________________________________________________
# # input_shape = [(None, 4, 8), (None, 4, 64, 720, 14)]
# # model.build(input_shape)
# # model.summary()
# # filename = os.path.join(self.mother_folder, 'results', 'model1.png')
# # tf.keras.utils.plot_model(model, show_shapes=True, to_file=filename)
#
# # ______________________________________________________________________________________________________________
# # Data loading
# # ______________________________________________________________________________________________________________
# G_create = Geometry_data_prepare(self.mother_folder)
# loader2 = Lidar_data_prepare(self.mother_folder)
#
# # ______________________________________________________________________________________________________________
# # Go over sequences
# # ______________________________________________________________________________________________________________
# train_loss = []
# val_loss = []
# n_epochs_best = []
#
# for seq in self.sequences:
#     seq_int = int(seq)
#     seq_scans = np.int(self.scans[seq_int])
#
#     print("_______________________________________________________________________________________________"
#           "_______________________________________________________________________________")
#     print('Sequence started: ', seq)
#     print("_______________________________________________________________________________________________"
#           "_______________________________________________________________________________")
#
#     G_data, G_gt = G_create.load_saved_data_all(seq)
#
#     mu = self.divided_train
#     seq_scans = seq_scans
#     inside_loop = np.int(np.ceil(np.divide(seq_scans, mu)))
#
#     for counter in range(inside_loop):
#         temp_start = counter * mu
#         # if counter == 0: temp_start = 5
#         temp_end = temp_start + mu
#         if seq_scans - 5 < temp_end: temp_end = seq_scans - 5
#
#         print('Sequence', seq, 'part', counter, ': ', temp_start, 'to', temp_end, 'of', seq_scans + 5,
#               '_____________________________________________________')
#
#         AI_data_temp = loader2.load_saved_data_h5(seq, temp_start, temp_end)
#         G_data_temp = G_data[temp_start: temp_end]
#         G_gt_temp = G_gt[temp_start: temp_end]
#
#         x = {'geo_input': G_data_temp, 'AI_input': AI_data_temp}
#         y = G_gt_temp
#         # print('here', x['geo_input'].shape, x['AI_input'].shape, y.shape)
#         train_history = model.fit(x=x,
#                                   y=y,
#                                   epochs=self.Epochs,
#                                   batch_size=self.Batch_size,
#                                   validation_split=0.1,
#                                   # callbacks=[callback],
#                                   verbose=1)
#
#         n_epochs_best_temp = np.argmax(train_history.history['val_loss'])
#         print(n_epochs_best_temp)
#         n_epochs_best.append(n_epochs_best_temp)
#         train_loss.append(train_history.history['loss'])
#         val_loss.append(train_history.history['val_loss'])
#         gc.collect()


'''