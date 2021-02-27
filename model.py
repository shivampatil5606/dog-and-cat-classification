# import the necessary packages
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import cv2
import os

# grab the list of images that we'll be describing
print("Checking Images...")
# Enter foldername with train images
# ex: My 1st train image path :- "kaggle_dogs_vs_cats/train/cat.0.jpg"
imagePaths = list(paths.list_images('kaggle_dogs_vs_cats'))
# print(len(imagePaths))
# print(imagePaths[2300])

# initialize the data matrix and labels list
data = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # print(imagePath)
    # print(i)
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    # print(label)

    # construct a feature vector raw pixel intensities, then update
    # the data matrix and labels list
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    features = cv2.resize(image, (32, 32)).flatten()
    data.append(features)
    labels.append(label)

    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("Processed {}/{}".format(i, len(imagePaths)))

print(len(data))
# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)
print(labels)

# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`
data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("Contructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Saving the variables to a file using pickle
with open('train.pickle', 'wb') as f:
    pickle.dump([trainData, testData, trainLabels, testLabels], f)

# Using saved variables
# with open('train.pickle', 'rb') as f:
#     trainData, testData, trainLabels, testLabels = pickle.load(f)

# print("No of trainData: ", len(trainData))
# print("No of trainLabels: ", len(trainLabels))
# print("No of testDate: ", len(testData))
# print("No of testLabels: ", len(testLabels))


# define the architecture of the network
model = Sequential()
model.add(Dense(768, input_dim=3072, kernel_initializer="uniform", activation="relu"))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))

# train the model using SGD
print("Compiling Model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(trainData, trainLabels, epochs=50, batch_size=128, verbose=1)

# show the accuracy on the testing set
print("Evaluating on testing data set...")
(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
print("Result : loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

# dump the network architecture and weights to file
print("Dumping model to file...")
model.save('output/finalModel.hdf5')

# Loading the saved model
# print("Reading saved model..")
# model = load_model('output/finalModel.hdf5')
