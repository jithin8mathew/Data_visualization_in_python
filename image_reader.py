from __future__ import print_function
import keras
import tensorflow as tf
from tensorflow import keras

import matplotlib.pylab as plt
import os
import numpy as np
from keras.preprocessing import image
import cv2
from sklearn.externals import joblib
from skimage.feature import hog

# import os
# from docx import Document
# from docx.shared import Inches

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

model=tf.keras.models.load_model(
    os.getcwd()+"/weights/full_model.h5",
    custom_objects=None,
    compile=True
)
#model = load_model(os.getcwd()+"\\weights\\Model_weights.h5")
class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

def clasf(test_image):
    
    t=test_image

    import cv2
    try:

        test_image = image.img_to_array(test_image)
        #test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        #test_image = cv2.GaussianBlur(test_image, (5, 5), 0)
        test_image= test_image.reshape(28,28,1)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        plt.imshow(t)
        plt.show()
        #out_string+=(class_mapping[np.argmax(result)])
        with open('output.txt', 'a+') as w:
            w.write(class_mapping[np.argmax(result)])
    except Exception: pass
    
    #print(result)
    


# load the document image

im = cv2.imread("index.png")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
#from keras_cnn_test import class_ify
for rect in rects:
    
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    aval=im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3
    
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    try:
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
    except Exception:pass

    clasf(roi)
