# Covid-19-CXR-Image-Translations-using-GAN-s
<hr/>


# Task
The goal of the project was to take as input an image of normal Chest X-Ray i.e *free of Covid-19* and apply image processing using **Cycle GAN** and translate it into a *Covid-19* Chest X-Ray Image.

<hr/>

# Model 
The Model used for doing this was trained for 30 epochs (due to lack of computational power) though a GAN should be ideally trained for a minimum of 100 epochs.

I compensated this by reducing the batch size to 1.
<hr/>

# Data Set 

the training set was taken from 3 sources -: 

1. https://github.com/ieee8023/covid-chestxray-dataset
2. https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
3. https://github.com/agchung

the final dataset had 500 images for both classes. (i.e 500 for Normal CXR and 500 for Covid-19 CXR)

<hr/>

# Model Architecture

The Cycle GAN model has 2 generator networks and 2 discriminator networks for tramslations between domain A -> domain B or vice versa.

**Generator Network**

the generator network is as shown in the figure.

<p align="center">
  <img src="./Model Diagrams/model_view.png" height ="250" width="500" title="Generator Model">
</p>

The Generator used in this implementation involves three parts to it: `In-network Downsampling`, `Several Residual Blocks` and `In-network Upsampling`

The first layer has `kernel-size` = 7 while the other has `kernel size` = 3

Each `Residual Block` consists of two Convolution Layers. The first Convolution Layer is followed by `Instance Normalization` and `ReLu` activation. The output is then passed through a second Convolution Layer followed by `Instance Normalization`. The output obtained from this is then concatenated to the original input. 
They are used to solve the problem of `Vanishing Gradient/Exploding Gradient`.
Kernel size = (3,3).

The number of Residual Blocks depends on the size of the input image. For 128x128 images, 6 residual blocks are used and for 256x256 and higher dimensional images, 9 residual blocks are used.


**Discriminator Network**

<p align="center">
  <img src="./Model Diagrams/discriminator_model_view.png" height ="300" width="500" title="Discriminator Model">
</p>

The discriminator is a `Patch GAN` i.e it return a label for a patch rather than for the entire image. That is why we have a Conv2D layer at as last layer instead of a Dense layer. 

Discriminator network has kernel size (4,4) and stride =(2,2).

PATCH SHAPE = (16,16)

<hr/>

# Results Obtained
Following are the results, obtained by using a model trained for 30 epochs, each epoch having 576 steps. 
<p align="center">
  <img src="./generated_vs_original/Generated_vs_Original5.png" height ="300" width="500" title="Generated v/s Original Image -Result">
</p>
<p align="center">
  <img src="./generated_vs_original/Generated_vs_Original1.png" height ="300" width="500" title="Generated v/s Original Image -Result">
</p>
<p align="center">
  <img src="./generated_vs_original/Generated_vs_Original2.png" height ="300" width="500" title="Generated v/s Original Image -Result">
</p>
<p align="center">
  <img src="./generated_vs_original/Generated_vs_Original3.png" height ="300" width="500" title="Generated v/s Original Image -Result">
</p>
<p align="center">
  <img src="./generated_vs_original/Generated_vs_Original4.png" height ="300" width="500" title="Generated v/s Original Image -Result">
</p>

#### Comment on the image generated : 
1. The generated image on the left appears to have more noise or artifacts compared to the "Normal" image on the left.
2. In the generated "COVID-19" image, there may be regions that are intended to represent pathological features associated with COVID-19, such as ground-glass opacities or consolidation. These features are characterized by areas of increased opacity in the lung fields.

<hr/>

# References and Links

1. Hasib Zunair and A. Ben Hamza , “Synthesis of COVID-19 Chest X-rays using Unpaired Image-to-Image Translation”, https://link.springer.com/journal/13278, Springer, 2021.

2. “Enhancing Automated COVID-19 Chest X-ray Diagnosis by Image-to-Image GAN Translation”, 2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM) 978-1-7281-6215-7/20/$31.00 ©2020 IEEE DOI: 10.1109/BIBM49941.2020.9313466,Enhancing Automated COVID-19 Chest X-ray Diagnosis by Image-to-Image GAN Translation | IEEE Conference Publication | IEEE Xplore

3. “Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks”,Jun-Yan Zhu∗
Taesung Park,Phillip Isola,Alexei A. Efros,arXiv:1703.10593v7 [cs.CV] 24 Aug 2020,[1703.10593] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (arxiv.org)

4. A Gentle Introduction to CycleGAN for Image Translation - MachineLearningMastery.com 
5. The Beauty of CycleGAN. The intuition and math behind… | by Sebastian Theiler | Analytics Vidhya | Medium 
6. https://github.com/ieee8023/covid-chestxray-dataset 
7. https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
8. https://github.com/agchung
9. [Deep Learning with PyTorch: Zero to GANs | Jovian](https://jovian.com/learn/deep-learning-with-pytorch-zero-to-gans)
10. [Generative Adversarial Network (GAN) - GeeksforGeeks](https://www.geeksforgeeks.org/generative-adversarial-network-gan/)


<hr/>

# Future Updates

1. To add a similarity check between the generated image and the input.
2. Add more functionalities like present covid-19 cases/ active cases.
3. The API's for fetching the covid-19 statistics is deprectaed, as well as the visual map for the same is deprecated, a future update would be to fix them by integrating new API's


<hr/>

# Usage
To use this, first clone this repo into your desktop, install all the dependencies ,then run the app.py using flask run 
**Note** To generate the images you need to have a trained model for translating normal domain image into the covid-19 domain image. 
<hr>

# Collaborations 
This project is open for collaborations, you are free to reach out to me regarding any collaborations, or in case you need the models that I have trained for testing/research purposes. 

