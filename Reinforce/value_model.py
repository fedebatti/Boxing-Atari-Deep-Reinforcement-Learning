import tensorflow as tf
from tensorflow.keras import layers

class value_model(tf.keras.Model):
  # def __init__(self):
  #   super().__init__()
  #   self.conv_1 = layers.Conv2D(filters=32, kernel_size=(8,8), strides=4, activation='relu') #padding='SAME')
  #   self.conv_2 = layers.Conv2D(filters=64, kernel_size=(4,4), strides=2, activation='relu') #padding='SAME')
  #   self.conv_3 = layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu')
  #   self.flatten = layers.Flatten()
  #   self.dense_1 = layers.Dense(64)
  #   self.dense_2 = layers.Dense(1, activation=None)

  def __init__(self):
    super().__init__()
    self.conv_1 = layers.Conv2D(filters=16, kernel_size=(8,8), strides=4, activation='relu') #padding='SAME')
    self.conv_2 = layers.Conv2D(filters=32, kernel_size=(4,4), strides=2, activation='relu') #padding='SAME')
    self.flatten = layers.Flatten()
    self.dense_1 = layers.Dense(128)
    self.dense_2 = layers.Dense(1, activation=None)


  def call(self, input_data):
    x = self.conv_1(input_data)
    x = self.conv_2(x)
    # x = self.conv_3(x)
    x = self.flatten(x)
    x = self.dense_1(x)
    v = self.dense_2(x)
    return v
    