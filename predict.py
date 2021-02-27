# import the necessary packages
from __future__ import print_function
from keras.models import load_model
import numpy as np
import cv2


# initialize the class labels for the Kaggle dogs vs cats dataset
CLASSES = ["cat", "dog"]

# load the network
print("Loading network architecture and weights...")
model = load_model('output/finalModel.hdf5')
print("Model loaded..")

# Path of image to test
imagePath = "test_images/4.jpg"
image = cv2.imread(imagePath)
features = cv2.resize(image, (32, 32)).flatten() / 255.0
features = np.array([features])

# classify the image using our extracted features and pre-trained
# neural network
probs = model.predict(features)[0]
prediction = probs.argmax(axis=0)

# draw the class and probability on the test image and display it
# to our screen
label = "{}: {:.2f}%".format(CLASSES[prediction],
                             probs[prediction] * 100)
cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 3)
cv2.imshow("Image", image)
cv2.waitKey(0)
