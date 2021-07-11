"""Extract features with InceptionV3"""
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
# preprocess_input will scale the pixels appropriately
from tensorflow.keras.applications.inception_v3 import preprocess_input
# KMeans is for unsupervised clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Array processing
import numpy as np
# Glob module finds the pathnames that match the description
import glob
# Module designed to solve computer vision algorithms
import cv2
# Plotting module for python
import matplotlib.pyplot as plt

# InceptionV3 is loaded with the pretrained weights from imagenet
# False indicates that the final dense layers (also known as fully connected layers) are not included
model = InceptionV3(weights='imagenet', include_top=False)
def feature_extraction(path):
    model_feature_list=[]
    # Getting the images from the given path
    for im in glob.glob(path):
        print("Path to the image:", im)
        # Reading the image and converting to an array
        im = cv2.imread(im)
        # Resize the image
        im = cv2.resize(im,(299,299))
        # expand_dims is used to expand the array of pixels from 3D to 4D..
        # [samples, rows, columns, channels ]
        # preprocess_input scales the pixels. subtracts the mean RGB Channels of the imagenet dataset.
        img = preprocess_input(np.expand_dims(im.copy(), axis=0))
        # the model outputs a numpy array for the features in the image
        model_feature = model.predict(img)
        # Converting the feature vector to an array format
        model_feature_np = np.array(model_feature)
        # Takes all the channels and turns it into one
        model_feature_list.append(model_feature_np.flatten())

    return np.array(model_feature_list)

# Features from the last max pooling layer
InceptionV3_features = feature_extraction("/Users/narimanemadzadeh/Desktop/ibc_images/*.tif")
print("Shape of the InceptionV3 features: ", np.shape(InceptionV3_features))

InceptionV3_features_clustering = KMeans(n_clusters=3, random_state=0).fit(InceptionV3_features)
print("Kmeans Lables: ", InceptionV3_features_clustering.labels_)
