# MyLibEVA
* A deep nueral network library 
* Built using pytorch
* This library has ready  to use functions for training and testing the CNN models for CV applications
* This project is currently under development

### Contents

* main.py
  * This file has training and test functions 

* models folder
  * has all CNN models that are  ready to be trained
  * resnet.py
  * custom_resnet.py
  
* utils .py
  *  Files has required utility functions.
   
* Remaining files in this repository have required functions for calculating classwise accuracy, creatingcustom dataset classes ,performing  data augmentation ,
  identifying misclassifed images,plotting graphs for train and test accuracy and lossses.
  * accuracy.py
  * custom_dataset.Cifar10.py
  * dataAugmentation.py
  * misclassification.py
  * trainingtesting.py
  
 * gradcam folder
  * Files in this folder have required functions for gradcam calculation and visualization
  * gradcam.py
  * visualization.py

