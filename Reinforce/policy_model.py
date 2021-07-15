import tensorflow as tf
from tensorflow.keras import layers


class policy_model(tf.keras.Model):
  # def __init__(self):
    # super().__init__()
    # self.conv_1 = layers.Conv2D(filters=32, kernel_size=(8,8), strides=4, activation='relu') #padding='SAME')  
    # self.conv_2 = layers.Conv2D(filters=64, kernel_size=(4,4), strides=2, activation='relu') #padding='SAME')
    # self.conv_3 = layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu')
    # self.flatten = layers.Flatten()
    # self.dense_1 = layers.Dense(128)
    # self.dense_2 = layers.Dense(18, activation='softmax')

  def __init__(self):
    super().__init__()
    self.conv_1 = layers.Conv2D(filters=16, kernel_size=(8,8), strides=4, activation='relu') #padding='SAME')  
    self.conv_2 = layers.Conv2D(filters=32, kernel_size=(4,4), strides=2, activation='relu') #padding='SAME')
    self.flatten = layers.Flatten()
    self.dense_1 = layers.Dense(256)
    self.dense_2 = layers.Dense(18, activation='softmax')

  def call(self, input_data):
    x = self.conv_1(input_data)
    x = self.conv_2(x)
    # x = self.conv_3(x)
    x = self.flatten(x)
    x = self.dense_1(x)
    v = self.dense_2(x)
    return v

  def create_prediction_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=16, kernel_size=(8,8), strides=4, activation='relu')) #padding='SAME'))
    model.add(layers.Conv2D(filters=32, kernel_size=(4,4), strides=2, activation='relu')) #padding='SAME'))
    # model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.Dense(18, activation='softmax'))
    model.compile()
    return model