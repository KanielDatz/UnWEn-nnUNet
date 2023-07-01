# Welcome to UnnUNet!

**This repisatory is a part of our final project in [EE046211 Deep Learning course](https://github.com/taldatech/ee046211-deep-learning) in the [Technion - Israel Institute of Technology](https://en.wikipedia.org/wiki/Technion_%E2%80%93_Israel_Institute_of_Technology)
By Daniel Katz and Natalie Mendelson, spring 2023**
- [  Welcome to UnnUNet!](# Welcome to UnnUNet!)
  * [Overview](#Overview)
  * [ Approach](#Approach)

# Overview
**Motivation**
Detecting segmentation model errors is **crucial** in medical imaging due to the consequences of mistakes in this field.
Segmentation model performance is usually measured using metrics as Intersection-Over-Union or Dice Coefficient. but these are relevant only when ground truth is present.
**During inference, we don't have ground truth.**
**Our goal is to introduces a method to estimate models' prediction uncertainty for medical image segmentation **

**Model** 
We implemented our method on [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
nnU-Net is a semantic segmentation method that automatically adapts to a given dataset. It will analyze the provided training cases and automatically configure a matching U-Net-based segmentation pipeline. 
nnU-Net is widely recognized for its exceptional performance in image segmentation tasks. However, one limitation of nnU-Net is the lack of a measure to indicate the possibility of failure or uncertainty, particularly in large-scale image segmentation applications with heterogeneous data. This is the issue we adress in our project.
If you are not familiar with nnUNet we advice you to take a look at [nnUNet git](https://github.com/MIC-DKFZ/nnUNet) and [paper.](https://www.nature.com/articles/s41592-020-01008-z)

**Dataset**
We used a [publicly availible](https://cardiacmr.hms.harvard.edu/downloads-0) dataset from Harvard [Cardiac MR Center Dataverse](https://github.com/HMS-CardiacMR).
It contains cardiac T1 weighted images for 210 patients, 5 slices per patient and 11 T1-weighted image per slice.
Manual contours for Epi- and Endocardial contours are provided for each T1-weighted image.
Total of ~11.5K images and labels.

# Approach
The project employed several steps to estimate the uncertainty of nnU-Net predictions. 
Learning rate was modified to utilize cyclic learning rate (clr) technique.
<img width="600" alt="Cyclic lr" src="https://github.com/KanielDatz/UnnUNet/blob/master/UnnUnet_documentation/assets/cyclicLr.png?raw=true">

By changing the learning rate in a cyclic manner, the model's convergence to multiple minima was ensured. At each of these minima, multiple checkpoints were extracted from the model.

<img width="600" alt="Cyclic lr" src="https://github.com/KanielDatz/UnnUNet/blob/master/UnnUnet_documentation/assets/progress.png?raw=true">

Next, an ensemble approach was adopted by aggregating the predictions (probabilities) of each class from the extracted checkpoints. This ensemble of predictions was then used to calculate the entropy of both classes. The entropy of each image was summed, and normalization was achieved by dividing the sum by the contour of the labeled segmentation.

<img width="113" alt="image" src="https://github.com/KanielDatz/UnnUNet/blob/master/UnnUnet_documentation/assets/cyclicLr.png?raw=true">
<img width="114" alt="image" src="https://github.com/KanielDatz/UnnUNet/blob/master/UnnUnet_documentation/assets/cyclicLr.png?raw=true">

## Results 
Using this process, an uncertainty score was obtained for each image. To evaluate the relationship between uncertainty and accuracy, the correlation between the uncertainty score and the Dice coefficient was examined. The Dice coefficient is a common metric used to assess the similarity between predicted and ground truth segmentations.

<img width="287" alt="image" src="https://github.com/KanielDatz/UnWEn-nnUNet/assets/128894307/20c98ec7-af72-4a8e-919e-fefc9b4242b8">


##
#######################################################################
## What can nnU-Net do for you?
If you are a **domain scientist** (biologist, radiologist, ...) looking to analyze your own images, nnU-Net provides 
an out-of-the-box solution that is all but guaranteed to provide excellent results on your individual dataset. Simply 
convert your dataset into the nnU-Net format and enjoy the power of AI - no expertise required!

If you are an **AI researcher** developing segmentation methods, nnU-Net:
- offers a fantastic out-of-the-box applicable baseline algorithm to compete against
- can act as a method development framework to test your contribution on a large number of datasets without having to 
tune individual pipelines (for example evaluating a new loss function)
- provides a strong starting point for further dataset-specific optimizations. This is particularly used when competing 
in segmentation challenges
- provides a new perspective on the design of segmentation methods: maybe you can find better connections between 
dataset properties and best-fitting segmentation pipelines?

## What is the scope of nnU-Net?
nnU-Net is built for semantic segmentation. It can handle 2D and 3D images with arbitrary 
input modalities/channels. It can understand voxel spacings, anisotropies and is robust even when classes are highly
imbalanced.

nnU-Net relies on supervised learning, which means that you need to provide training cases for your application. The number of 
required training cases varies heavily depending on the complexity of the segmentation problem. No 
one-fits-all number can be provided here! nnU-Net does not require more training cases than other solutions - maybe 
even less due to our extensive use of data augmentation. 

nnU-Net expects to be able to process entire images at once during preprocessing and postprocessing, so it cannot 
handle enormous images. As a reference: we tested images from 40x40x40 pixels all the way up to 1500x1500x1500 in 3D 
and 40x40 up to ~30000x30000 in 2D! If your RAM allows it, larger is always possible.

## How does nnU-Net work?
Given a new dataset, nnU-Net will systematically analyze the provided training cases and create a 'dataset fingerprint'. 
nnU-Net then creates several U-Net configurations for each dataset: 
- `2d`: a 2D U-Net (for 2D and 3D datasets)
- `3d_fullres`: a 3D U-Net that operates on a high image resolution (for 3D datasets only)
- `3d_lowres` → `3d_cascade_fullres`: a 3D U-Net cascade where first a 3D U-Net operates on low resolution images and 
then a second high-resolution 3D U-Net refined the predictions of the former (for 3D datasets with large image sizes only)

**Note that not all U-Net configurations are created for all datasets. In datasets with small image sizes, the 
U-Net cascade (and with it the 3d_lowres configuration) is omitted because the patch size of the full 
resolution U-Net already covers a large part of the input images.**

nnU-Net configures its segmentation pipelines based on a three-step recipe:
- **Fixed parameters** are not adapted. During development of nnU-Net we identified a robust configuration (that is, certain architecture and training properties) that can 
simply be used all the time. This includes, for example, nnU-Net's loss function, (most of the) data augmentation strategy and learning rate.
- **Rule-based parameters** use the dataset fingerprint to adapt certain segmentation pipeline properties by following 
hard-coded heuristic rules. For example, the network topology (pooling behavior and depth of the network architecture) 
are adapted to the patch size; the patch size, network topology and batch size are optimized jointly given some GPU 
memory constraint. 
- **Empirical parameters** are essentially trial-and-error. For example the selection of the best U-net configuration 
for the given dataset (2D, 3D full resolution, 3D low resolution, 3D cascade) and the optimization of the postprocessing strategy.

## How to get started?
Read these:
- [Installation instructions](documentation/installation_instructions.md)
- [Dataset conversion](documentation/dataset_format.md)
- [Usage instructions](documentation/how_to_use_nnunet.md)

Additional information:
- [Region-based training](documentation/region_based_training.md)
- [Manual data splits](documentation/manual_data_splits.md)
- [Pretraining and finetuning](documentation/pretraining_and_finetuning.md)
- [Intensity Normalization in nnU-Net](documentation/explanation_normalization.md)
- [Manually editing nnU-Net configurations](documentation/explanation_plans_files.md)
- [Extending nnU-Net](documentation/extending_nnunet.md)
- [What is different in V2?](documentation/changelog.md)

[//]: # (- [Ignore label]&#40;documentation/ignore_label.md&#41;)

## Where does nnU-net perform well and where does it not perform?
nnU-Net excels in segmentation problems that need to be solved by training from scratch, 
for example: research applications that feature non-standard image modalities and input channels,
challenge datasets from the biomedical domain, majority of 3D segmentation problems, etc . We have yet to find a 
dataset for which nnU-Net's working principle fails!

Note: On standard segmentation 
problems, such as 2D RGB images in ADE20k and Cityscapes, fine-tuning a foundation model (that was pretrained on a large corpus of 
similar images, e.g. Imagenet 22k, JFT-300M) will provide better performance than nnU-Net! That is simply because these 
models allow much better initialization. Foundation models are not supported by nnU-Net as 
they 1) are not useful for segmentation problems that deviate from the standard setting (see above mentioned 
datasets), 2) would typically only support 2D architectures and 3) conflict with our core design principle of carefully adapting 
the network topology for each dataset (if the topology is changed one can no longer transfer pretrained weights!) 

# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
