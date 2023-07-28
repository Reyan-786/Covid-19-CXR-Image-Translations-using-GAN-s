# Covid-19-CXR-Image-Translations-using-GAN-s
<hr/>
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
  <img src="./model_view.png" height ="250" width="500" title="Generator Model">
</p>

The Generator used in this implementation involves three parts to it: `In-network Downsampling`, `Several Residual Blocks` and `In-network Upsampling`

The first layer has `kernel-size` = 7 while the other has `kernel size` = 3

Each `Residual Block` consists of two Convolution Layers. The first Convolution Layer is followed by `Instance Normalization` and `ReLu` activation. The output is then passed through a second Convolution Layer followed by `Instance Normalization`. The output obtained from this is then concatenated to the original input. 
They are used to solve the problem of `Vanishing Gradient/Exploding Gradient`.
Kernel size = (3,3).

The number of Residual Blocks depends on the size of the input image. For 128x128 images, 6 residual blocks are used and for 256x256 and higher dimensional images, 9 residual blocks are used.


**Discriminator Network**

<p align="center">
  <img src="./discriminator_model_view.png" height ="300" width="500" title="Discriminator Model">
</p>

The discriminator is a `Patch GAN` i.e it return a label for a patch rather than for the entire image. That is why we have a Conv2D layer at as last layer instead of a Dense layer. 

Discriminator network has kernel size (4,4) and stride =(2,2).

PATCH SHAPE = (16,16)

<hr/>

# Results Obtained

<p align="center">
  <img src="./Generated_vs_Original5.png" height ="300" width="500" title="Generated v/s Original Image -Result">
</p>

#### This is one of the result obtained after training the GAN model, where you can see the `Genarated` image has some cloudy patches in the lung region as compared to the `Original` Image.

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





