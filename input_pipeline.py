import numpy as np
from numpy import load 
from numpy import asarray
from numpy.random import randint 
from os import listdir
from keras.utils import img_to_array
from keras.utils import load_img
import numpy as np


def load_real_samples(filename):
    #     load and prepare the training images
    data = load(filename)
    X1,X2 = data['arr_0'],data['arr_1']
#     scale from 0,255 to -1,1
    X1 = (X1-127.5)/127.5
    X2 = (X2-127.5)/127.5
    return [X1,X2]


# load all images in a directory into memory
def load_images(path, size=(512,512)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)

def generate_real_samples(dataset,n_samples,patch_shape):
    ix = randint(0,dataset.shape[0],n_samples)
    X=dataset[ix]
    y = np.ones([n_samples,patch_shape,patch_shape,1])
    return X,y

def generate_fake_samples(g_model,dataset,patch_shape):
    X= g_model.predict(dataset)
    y = np.zeros([len(X),patch_shape,patch_shape,1])
    return X,y

def preprocess_data(data):
    	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]


