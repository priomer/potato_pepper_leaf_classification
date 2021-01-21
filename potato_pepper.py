# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:18:59 2021

@author: ocn
"""

#prediction by pepper_potato.py

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.utils.np_utils import to_categorical

def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(200, 200))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    
    return img_tensor
    
model = load_model("model_pepper_potato_plant_disease3.h5")
img_path = 'E:/AVRN_Report/PlantVillage/pepper_potato/test/55.JPG'
check_image = load_image(img_path)
prediction = model.predict_classes(check_image)
print(prediction)

prediction =np.argmax(to_categorical(prediction), axis=1)
if prediction==0:
    prediction="Pepper_bell_bacterial"
elif prediction==1:
    prediction="Pepper_bell_healthy"
elif prediction==2:
    prediction="potato_early_blight"
elif prediction==3:
    prediction="potato_healthy"
else:
    prediction="potato_late_blight"


print(prediction)    
    
    
