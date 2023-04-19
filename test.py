from random import random
import numpy as np
from numpy import load 
from numpy import zeros 
from numpy import asarray
from numpy.random import randint 
from os import listdir
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot 
import numpy as np
from helper import select_sample,show_plot
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import load_model
from sklearn.utils import resample
from input_pipeline import load_images

path = '.\\Final Dataset\\'

dataA_all = load_images(path + 'trainA\\')
print('Loaded dataA: ', dataA_all.shape)
dataB_all = load_images(path + 'trainB\\')
print('Loaded dataB: ', dataB_all.shape)
# load dataset
A_data = resample(dataA_all, 
                 replace=False,     
                 n_samples=50,    
                 random_state=42) # reproducible results

B_data = resample(dataB_all, 
                 replace=False,     
                 n_samples=50,    
                 random_state=42) # reproducible results

A_data = (A_data - 127.5) / 127.5
B_data = (B_data - 127.5) / 127.5


cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('', cust)
model_BtoA = load_model('', cust)

# plot A->B->A (Monet to photo to Monet)
A_real = select_sample(A_data, 1)
B_generated  = model_AtoB.predict(A_real)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real, B_generated, A_reconstructed)
# plot B->A->B (Photo to Monet to Photo)
B_real = select_sample(B_data, 1)
A_generated  = model_BtoA.predict(B_real)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real, A_generated, B_reconstructed)

##########################
#Load a single custom image
test_image = load_img('test_image.jpg')
test_image = img_to_array(test_image)
test_image_input = np.array([test_image])  # Convert single image to a batch.
test_image_input = (test_image_input - 127.5) / 127.5

# plot B->A->B (Photo to Monet to Photo)
monet_generated  = model_BtoA.predict(test_image_input)
photo_reconstructed = model_AtoB.predict(monet_generated)
show_plot(test_image_input, monet_generated, photo_reconstructed)

