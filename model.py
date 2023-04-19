from random import random
import numpy as np
from numpy import load 
from numpy import zeros 
from numpy import asarray
from numpy.random import randint 
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input 
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import Activation 
from tensorflow.python.keras.layers import Concatenate
# downloaded the instancenormalization from https://www.github.com/keras-team/keras-contrib.git as instructed in the paper U.C.B unpaired .....

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import matplotlib.pyplot as plt

def define_discriminator(image_shape):
    #     weight initialization 
# Functional API
    init = RandomNormal(stddev = 0.02)
#     source image shape 
    in_image = Input(shape = image_shape)
#     C64 kernel - 4x4 stride - 2x2
    d = Conv2D(64 , (4,4) ,strides =(2,2),padding ='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha = 0.2)(d)
#     C128 - 4x4 kernel stride - 2x2 
    d = Conv2D(128 , (4,4) ,strides =(2,2),padding ='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha = 0.2)(d)
#     C256 - 4x4 kernel ,stride - 2x2
    d = Conv2D(256 , (4,4) ,strides =(2,2),padding ='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha =0.2)(d)
#     C512
    d = Conv2D(512 , (4,4) ,strides =(2,2),padding ='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha =0.2)(d)
#     second last output layer 4x4 kernel but stride 1x1
    d = Conv2D(256 , (4,4) ,strides =(1,1),padding ='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha =0.2)(d)
#     patch output 
    patch_output  = Conv2D(1,(4,4),padding ='same',kernel_initializer = init)(d)
#     defining the model
    model = Model(in_image, patch_output)
    
    model.compile(loss='mae',optimizer=Adam(lr=0.0002,beta_1= 0.5),loss_weights=[0.5])
    return model

def resnet_block(n_filters, input_layer):
    init = RandomNormal(stddev = 0.02)
#     first Conv layer 
    g = Conv2D(n_filters,(3,3),padding ='same',kernel_initializer = init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g =Activation('relu')(g)
#     second conv layer
    g = Conv2D(n_filters,(3,3),padding ='same',kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    
#     merge channel wise with the input layer
    g = Concatenate()([g,input_layer])
    return g

def define_generator(image_shape,n_resnet=9):
    init = RandomNormal(stddev=0.02)
#     image input 
    in_image = Input(shape=image_shape)
#     c7s1-64
    g = Conv2D(64,(7,7),padding ='same',kernel_initializer = init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
#     d128
    g = Conv2D(128,(3,3),strides = (2,2),padding ='same',kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
#     d256
    g = Conv2D(256,(3,3),strides = (2,2),padding ='same',kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
#     R256
    for _ in range(n_resnet):
        g = resnet_block(256,g)
#     u128
    g = Conv2DTranspose(128,(3,3),strides = (2,2),padding ='same',kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
#     u64
    g = Conv2DTranspose(64,(3,3),strides = (2,2),padding ='same',kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
#      c7s1-3
    g = Conv2D(3,(7,7),padding='same',kernel_initializer = init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
#     define model
    model =Model(in_image,out_image)
    return model

def define_composite_model(g_model_1,d_model,g_model_2,image_shape):
    g_model_1.trainable = True
    d_model.trainable =False
    g_model_2.trainable =False
#     adversarial loss 
    input_gen = Input(shape=  image_shape)
    gen1_out = g_model_1(input_gen)
    output_d  = d_model(gen1_out)
#     identity loss
    input_id = Input(shape =image_shape)
    output_id  = g_model_1(input_id)
#     cycle loss forward 
    output_f = g_model_2(input_id)
#     cycle loss back
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)
#     define the model
    model =Model([input_gen,input_id],[output_d,output_id,output_f,output_b])
    
    opt=Adam(lr = 0.0002,beta_1=0.5)
    model.compile(loss=['mse','mae','mae','mae'],loss_weights = [1,5,10,10],optimizer =opt)
    return model

def load_real_samples(filename):
    #     load and prepare the training images
    data = load(filename)
    X1,X2 = data['arr_0'],data['arr_1']
#     scale from 0,255 to -1,1
    X1 = (X1-127.5)/127.5
    X2 = (X2-127.5)/127.5
    return [X1,X2]


def generate_real_samples(dataset,n_samples,patch_shape):
    ix = randint(0,dataset.shape[0],n_samples)
    X=dataset[ix]
    y = np.ones([n_samples,patch_shape,patch_shape,1])
    return X,y

def generate_fake_samples(g_model,dataset,patch_shape):
    X= g_model.predict(dataset)
    y = np.zeros([len(X),patch_shape,patch_shape,1])
    return X,y

def save_models(step,g_model_AtoB,g_model_BtoA):
    filename1 = 'g_model_AtoB{}.h5'.format(step+1)
    g_model_AtoB.save(filename1)
    
    filename2 = 'g_model_BtoA{}.h5'.format(step+1)
    g_model_BtoA.save(filename2)
    print("Saved-> {} and {}".format(filename1,filename2))
    
def summarize_performance(step,g_model,train_X,name,n_samples=5):
    X_in,_ = generate_real_samples(train_X,n_samples,0)
    X_out,_ = generate_fake_samples(g_model,X_in,0)
#     scale all pixel values from -1,1 to 0,1
    X_in =(X_in+1)/2.0
    X_out= (X_out+1)/2.0
#     plot real samples
    for i in range(n_samples):
        plt.subplot(2,n_samples,1+n_samples+i)
        plt.axis('off')
        plt.imshow(X_in[i])
#         plot translated samples
    for i in range(n_samples):
        plt.subplot(2,n_samples,1+n_samples+i)
        plt.axis('off')
        plt.imshow(X_out[i])
#         save plot to a file 
    filename1 = "{}epc-generated.png".format(step+1)
    plt.savefig(filename1)
    plt.close()

def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=1):
    # define properties of the training run
    n_epochs, n_batch, = epochs, 1  #batch size fixed to 1 as suggested in the paper
    # determine the output square shape of the discriminator
    n_patch = d_model_A.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # prepare image pool for fake images
    poolA, poolB = list(), list()
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples from each domain (A and B)
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
        # generate a batch of fake samples using both B to A and A to B generators.
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
        # update fake images in the pool. Remember that the paper suggstes a buffer of 50 images
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)

        # update generator B->A via the composite model
        g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        # update discriminator for A -> [real/fake]
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

        # update generator A->B via the composite model
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
        # update discriminator for B -> [real/fake]
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        # summarize performance
        #Since our batch size =1, the number of iterations would be same as the size of our dataset.
        #In one epoch you'd have iterations equal to the number of images.
        #If you have 100 images then 1 epoch would be 100 iterations
        print('Iteration>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
        # evaluate the model performance periodically
        #If batch size (total images)=100, performance will be summarized after every 75th iteration.
        if (i+1) % (bat_per_epo * 1) == 0:
            # plot A->B translation
            summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            # plot B->A translation
            summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
        if (i+1) % (bat_per_epo * 5) == 0:
            # save the models
            # #If batch size (total images)=100, model will be saved after 
            #every 75th iteration x 5 = 375 iterations.
            save_models(i, g_model_AtoB, g_model_BtoA)

