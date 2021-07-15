import numpy as np

#Convert image from rgb to grayscale
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

#Reshape the observation image to reduce the input dimension of the network
def observation_preprocessing(observation):
    img = observation[1:176:2, ::2]
    img = (img - 128) / 128-1
    img = rgb2gray(img)
    return np.expand_dims(img.reshape(88,80), axis=2)
