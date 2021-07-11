from tensorflow.keras.applications.vgg16 import  VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

model = VGG16()
# redefine model to output right after the following hidden layer
ixs = [2,5,9,13,17]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs = model.inputs, outputs = outputs)
model.summary()
# Load the image with the required shape
img = load_img('/Users/narimanemadzadeh/Desktop/ibc_images/11-S 11-15764-28  5PLEX_[41347,17079]_image_with_all_seg.tif', target_size=(224, 224))
# Convert the image to an array
img = img_to_array(img)
# Expand the dimmensions so that it represents a single sample
img = expand_dims(img, axis=0)
# prepare the image by scaling pixel values for the model
img = preprocess_input(img)
# Get the feature map for the first hidden layer
feature_maps = model.predict(img)

# plot the output from each block
square = 8
for fmap in feature_maps:
    ix = 1
    # plot all 64 maps in an 8*8 square
    for _ in range(square):
        for _ in range(square):
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
        # plot filter channel in grayscale
            pyplot.imshow(fmap[0,:,:,ix-1], cmap=None)
            ix +=1

# Show the figure
    pyplot.show()
