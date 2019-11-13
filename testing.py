import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import numpy as np
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

model=tf.keras.models.load_model(
    os.getcwd()+"/weights/full_model.h5",
    custom_objects=None,
    compile=True
)

train_data_path = "F:/Machine_learning/Data_sets/emnist/emnist-balanced-train.csv"
#test_data_path = "F:/Machine_learning/Data_sets/emnist/emnist-balanced-test.csv"

train_data = pd.read_csv(train_data_path, header=None)

import matplotlib.pyplot as plt
num_classes = len(train_data[0].unique())
row_num = 8

# plt.imshow(train_data.values[row_num, 1:].reshape([28, 28]), cmap='Greys_r')
# plt.show()
img_flip = (train_data.values[row_num, 1:].reshape([1,28, 28,1]))

result = model.predict(img_flip)
class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

print(class_mapping[np.argmax(result)])