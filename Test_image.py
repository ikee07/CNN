#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:02:30 2023

@author: Ik
"""

import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Define the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the image
def load_image(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img / 255.0
    return img

# Load the trained CIFAR10 model
model = load_model('improved_modelV2.h5')

# URLs of the images to be fetched and predicted
URLs = [
    "https://www.zdnet.com/a/img/resize/071727877ee9884b60edd728253d2baadcb3985f/2021/02/23/19631992-64df-4af9-a288-a0cb4112e682/bombardier-globaleye-jet.jpg?width=1200&height=900&fit=crop&auto=webp",
    "https://images.all-free-download.com/images/graphiclarge/classic_jaguar_210354.jpg",
    "https://ichef.bbci.co.uk/news/976/cpsprodpb/67CF/production/_108857562_mediaitem108857561.jpg",
    "https://wagznwhiskerz.com/wp-content/uploads/2017/10/home-cat.jpg",
   
]

# Fetch and predict for each URL
for URL in URLs:
    picture_path = tf.keras.utils.get_file(origin=URL)
    img = load_image(picture_path)
    result = model.predict(img)

    image = plt.imread(picture_path)
    plt.imshow(image)
    plt.show()  # Display the image
    
    print('\nPrediction: This image most likely belongs to ' + class_names[int(result.argmax(axis=-1))])
