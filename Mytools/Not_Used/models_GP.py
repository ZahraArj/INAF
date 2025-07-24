import tensorflow as tf
import tensorflow_graphics as tfg
from tensorflow.keras import layers, models

# Define the input shape and the number of output parameters
input_shape = (None, None, 7) # Assuming the images have 7 channels
output_size = 8 # Assuming the output is 8 dual quaternion parameters

# Define the custom loss function for dual quaternions
def dual_quaternion_loss(y_true, y_pred):
  # Normalize the predicted dual quaternions
  y_pred = tfg.geometry.transformation.dual_quaternion.normalize(y_pred)
  # Compute the distance between the true and predicted dual quaternions
  distance = tfg.geometry.transformation.dual_quaternion.distance(y_true, y_pred)
  # Return the mean squared distance
  return tf.reduce_mean(tf.square(distance))



# Define the model as a subclass of the Model class
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



# ______________________________________________________________________________________________________________________
# LiDAR Only
# ______________________________________________________________________________________________________________________
class LiD_s2e_nowe(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(training=True))

        self.LSTM21 = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)  # Adjust the number of LSTM units as necessary
        self.LSTM22 = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)
        
        self.conc12 = layers.Concatenate()
        
        # Add additional layers as necessary
        self.new3 = layers.Dense(1024, activation='relu')
        self.new4 = layers.Dense(512, activation='relu')
        self.new5 = layers.Dense(8)
        
    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = True):
        geo_inp, lidar_inp = geo_lidar_input[0], geo_lidar_input[1]

        inputs1 = lidar_inp[..., :7]
        inputs2 = lidar_inp[..., 7:]
        
        x1_Resnet = self.create_res_net(inputs1)
        x2_Resnet = self.create_res_net(inputs2)

        # x1 = self.FL(x1_Resnet)
        # x2 = self.FL(x2_Resnet)
        
        x1 = self.LSTM21(x1_Resnet)
        x2 = self.LSTM22(x2_Resnet)
        
        x12 = self.conc12([x1,x2])
        
        x12 = self.new3(x12)
        x12 = self.new4(x12)
        outp = self.new5(x12)
        

        return outp

class LiD_s2e_nowe_two(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3 = None, None, None, None
        self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7 = None, None, None, None

        self.create_res_net = layers.TimeDistributed(create_res_net_NT_one(training=False))
        # self.create_res_net = layers.TimeDistributed(CustomResnet_GP())

        self.LSTM21 = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)  # Adjust the number of LSTM units as necessary
        self.LSTM22 = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)
        
        self.conc12 = layers.Concatenate()
        
        # Add additional layers as necessary
        self.new3 = layers.Dense(1024, activation='relu')
        self.new4 = layers.Dense(512, activation='relu')
        self.new5 = layers.Dense(8)
        
    @tf.function
    def call(self, geo_lidar_input: tf.Tensor, training: bool = False):
        geo_inp, lidar_inp = geo_lidar_input[0], geo_lidar_input[1]
        
        inputs1 = lidar_inp[..., :7]
        inputs2 = lidar_inp[..., 7:]
        
        x1_Resnet = self.create_res_net(inputs1)
        x2_Resnet = self.create_res_net(inputs2)

        # x1 = self.FL(x1_Resnet)
        # x2 = self.FL(x2_Resnet)
        
        x1 = self.LSTM21(x1_Resnet)
        x2 = self.LSTM22(x2_Resnet)
        
        x12 = self.conc12([x1,x2])
        
        x12 = self.new3(x12)
        x12 = self.new4(x12)
        outp = self.new5(x12)
        
        self.layer_grad0 = tf.gradients(outp[:, 0], x1)
        self.layer_grad1 = tf.gradients(outp[:, 2], x1)
        self.layer_grad2 = tf.gradients(outp[:, 2], x1)
        self.layer_grad3 = tf.gradients(outp[:, 3], x1)
        self.layer_grad4 = tf.gradients(outp[:, 4], x1)
        self.layer_grad5 = tf.gradients(outp[:, 5], x1)
        self.layer_grad6 = tf.gradients(outp[:, 6], x1)
        self.layer_grad7 = tf.gradients(outp[:, 7], x1)

        return outp, x1, x1, \
            self.layer_grad0, self.layer_grad1, self.layer_grad2, self.layer_grad3, \
            self.layer_grad4, self.layer_grad5, self.layer_grad6, self.layer_grad7
