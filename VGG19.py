"""Extract Features with VGG19"""
"""Importing the necessary packages"""
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
# preprocess_input will scale the pixels appropriately
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Sequential
# KMeans is for unsupervised clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA as RandomizedPCA
from scipy.spatial.distance import cdist
# Array processing
import numpy as np
# Glob module finds the pathnames that match the description
import glob
# Module designed to solve computer vision algorithms
import cv2
# Plotting module for python
import matplotlib.pyplot as plt
import matplotlib.cm as cm


new_model = Sequential()
new_model.add(VGG19(include_top=False, pooling='avg', weights = 'imagenet'))
new_model.summary()

# Not training the first layer of the model
new_model.layers[0].trainable = False

def feature_extraction(path):
    model_feature_list=[]
    # Getting the images from the given path
    for im in glob.glob(path):
        print("Path to the image:", im)
        # Reading the image and converting to an array
        im = cv2.imread(im)
        # Resize the image
        im = cv2.resize(im,(224,224))
        # expand_dims is used to expand the array of pixels from 3D to 4D..
        # [samples, rows, columns, channels ]
        # preprocess_input scales the pixels. subtracts the mean RGB Channels of the imagenet dataset.
        img = preprocess_input(np.expand_dims(im.copy(), axis=0))
        # the model outputs a numpy array for the features in the image
        model_feature = new_model.predict(img)
        # Converting the feature vector to an array format
        model_feature_np = np.array(model_feature)
        # Takes all the channels and turns it into one
        model_feature_list.append(model_feature_np.flatten())

    return np.array(model_feature_list)

# Features from the last max pooling layer
VGG19_features = feature_extraction("path to your images")
print("Shape of the VGG19 features: ", np.shape(VGG19_features))

# Compute a PCA - how many linear features are required to describe the data
model_pca = RandomizedPCA(18).fit(VGG19_features)
plt.plot(np.cumsum(model_pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.title('Principal Component Analysis')
plt.show()

model_isomap = Isomap(n_components=2)
proj = model_isomap.fit_transform(VGG19_features)
print("Shape of the projection" , np.shape(proj))
print("Isomap Projection", proj)
plt.title("Isomap on the Extracted Features")
plt.scatter(proj[:,0], proj[:,1])
plt.show()

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1,10)

for k in K:
    #Building and fitting the model
    kmeanModel = KMeans(n_clusters = k).fit(proj)
    kmeanModel.fit(proj)

    distortions.append(sum(np.min(cdist(proj, kmeanModel.cluster_centers_,'euclidean'),axis=1))/proj.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k]= sum(np.min(cdist(proj, kmeanModel.cluster_centers_,'euclidean'),axis=1))/ proj.shape[0]
    mapping2[k] = kmeanModel.inertia_

# Tabulating and Visualizing the results
# Different values of distortion
for key,val in mapping1.items():
    print(str(key)+' :  '+str(val))

# Different values of inertia
for key, val in mapping2.items():
    print(str(key)+' : '+ str(val))

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

VGG19_features_clustering = KMeans(n_clusters=3, random_state=0).fit(proj)
print("Kmeans Lables: ", VGG19_features_clustering.labels_)
