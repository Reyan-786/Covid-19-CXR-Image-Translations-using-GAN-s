# Covid-19-CXR-Image-Translations-using-GAN-s
This is my recent project ,in which I worked on Image 2 Image Translations using a Cycle GAN network. 

# Task
The goal of the project was to take as input an image of normal Chest X-Ray i.e *free of Covid-19* and apply image processing using **Cycle GAN** and translate it into a *Covid-19* Chest X-Ray Image.


# Model 
The Model used for doing this was trained for 30 epochs (due to lack of computational power) though a GAN should be ideally trained for a minimum of 100 epochs.

I compensated this by reducing the batch size to 1.

# Data Set 
the training set was taken from 3 sources -: 

1. https://github.com/ieee8023/covid-chestxray-dataset
2. https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
3. https://github.com/agchung

the final dataset had 500 images for both classes. (i.e 500 for Normal CXR and 500 for Covid-19 CXR)

# Model Architecture

The Cycle GAN model has 2 generator networks and 2 discriminator networks for tramslations between domain A -> domain B or vice versa.

**Generator Network**

the generator network is as shown in the figure.

<p align="center">
  <img src="./model_view.png" height ="350" width="350" title="hover text">
</p>




