from matplotlib import pyplot as plt
from datetime import datetime 
from model import define_discriminator,define_composite_model,define_generator,train
from input_pipeline import load_images , preprocess_data
from sklearn.utils import resample

path = ".\\Dataset1\\"
dataA_all = load_images(path + 'Normal\\')
print('Loaded dataA: ', dataA_all.shape)

dataB_all = load_images(path + 'Covid-19\\')
print('Loaded dataB: ', dataB_all.shape)


dataA = resample(dataA_all, 
                 replace=False,     
                 n_samples=500,    
                 random_state=42) 

#We could have just read the list of files and only load a subset, better memory management. 
dataB = resample(dataB_all, 
                 replace=False,     
                 n_samples=500,    
                 random_state=42) 

# plot source images
n_samples = 3
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(dataA[i].astype('uint8'))
plt.title("Some Normal Chest X Ray Images",loc='right')

# plot target image
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(dataB[i].astype('uint8'))
plt.title("Some COVID 19 Chest X Ray Images",loc='right')

plt.show()

data = [dataA, dataB]
print('Loaded', data[0].shape, data[1].shape)



dataset = preprocess_data(data)

image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)


start1 = datetime.now() 
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=5)

stop1 = datetime.now()
#Execution time of the model 
execution_time = stop1-start1
print("Execution time is: ", execution_time)