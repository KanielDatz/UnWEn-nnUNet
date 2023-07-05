## Welcome to UnnUNet!
TL;DR - *We propose a method of measuring segmentation model performance when ground truth is not available.*

**This repository is a part of our final project in [EE046211 Deep Learning course](https://github.com/taldatech/ee046211-deep-learning) in the [Technion - Israel Institute of Technology](https://en.wikipedia.org/wiki/Technion_%E2%80%93_Israel_Institute_of_Technology)
By [Daniel Katz](https://www.linkedin.com/in/danielkatz10/https://www.linkedin.com/in/danielkatz10/) and [Natalie Mendelson](https://www.linkedin.com/in/natalie-mendelson-b59646234/https://www.linkedin.com/in/natalie-mendelson-b59646234/), spring 2023**
our presentation is available [here](https://tome.app/project-db1e/unnunet-cljid6bos1ij5pm3dapzj8tmz) and a video in Hebrew is available [here](https://youtu.be/_wcqZOEGk_0)


# Overview
**Motivation**

Detecting segmentation model errors is **crucial** in medical imaging due to the consequences of mistakes in this field.
Segmentation model performance is usually measured using metrics such as Intersection-Over-Union or Dice Coefficient. but these are relevant only when ground truth is present.
**During inference, we don't have ground truth.**
**Our goal is to introduce a method to estimate models' prediction uncertainty for medical image segmentation **

**Model** 

We implemented our method on [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
nnU-Net is a semantic segmentation method that automatically adapts to a given dataset. It will analyze the provided training cases and automatically configure a matching U-Net-based segmentation pipeline. 
nnU-Net is widely recognized for its exceptional performance in image segmentation tasks. However, one limitation of nnU-Net is the lack of a measure to indicate the possibility of failure or uncertainty, particularly in large-scale image segmentation applications with heterogeneous data. This is the issue we address in our project.
If you are not familiar with nnUNet we advise you to take a look at [nnUNet git](https://github.com/MIC-DKFZ/nnUNet) and [paper.](https://www.nature.com/articles/s41592-020-01008-z)

**Dataset**
We used a [publicly available](https://cardiacmr.hms.harvard.edu/downloads-0) dataset from Harvard [Cardiac MR Center Dataverse](https://github.com/HMS-CardiacMR).
It contains cardiac T1-weighted images for 210 patients, 5 slices per patient, and 11 T1-weighted images per slice.
Manual contours for Epi- and Endocardial contours are provided for each T1-weighted image.
Total of ~11.5K images and labels.

# Approach
The project employed several steps to estimate the uncertainty of nnU-Net predictions. 
The learning rate was modified to utilize the cyclic learning rate (clr) technique.

<img width="600" alt="Cyclic lr" src="https://github.com/KanielDatz/UnnUNet/blob/master/UnnUnet_documentation/assets/cyclicLr.png?raw=true">

By changing the learning rate in a cyclic manner, the model's convergence to multiple minima was ensured. At each of these minima, multiple checkpoints were extracted from the model.
(read about extraction types in [Train](##train)

<img width="600" alt="Cyclic lr" src="https://github.com/KanielDatz/UnnUNet/blob/master/UnnUnet_documentation/assets/progress.png?raw=true">

Next, we make predictions (probability maps) of each class from the extracted checkpoints. This prediction and the variance between them are used to asses prediction uncertainty.
Our pipeline is able to produce an uncertainty map and score in three different methods, as described in [choosing uncertainty metric](#choosing).
nnUnet trains 5-fold cross-validation by default, but less is also possible.
For each fold the prediction of the model is done using the checkpoint with the best dice metric in the training.
on further 
 inference, all 5 folds are used to make predictions, the final prediction of the model is chosen by taking the prediction that has the lowest uncertainty score.

# Uncertainty Metric
**We implemented 3 types of uncertainty metrics:**
- **student T-test between classes:**
we run pixel-wise [T test](https://en.wikipedia.org/wiki/Student%27s_t-test) between the probability maps of each class.
for n checkpoints, we will have a group of n probabilities for each class for each pixel. so we can run a statistical hypothesis test the assumption H0: the two classes come from the same distribution. 
after running the test we get the P value map - we then set  the pixels where the P value is lower than 5% to null.

- **entropy between classes:**
After obtaining the mean probability maps through ensembling (considering all saved checkpoints), we calculated the entropy value for each pixel.
<img width="600" alt="Cyclic lr" src="https://github.com/KanielDatz/UnnUNet/blob/master/UnnUnet_documentation/assets//entropyclasses.png?raw=true">

- **entropy between all predictions:**
we calculate entropy on **all** prediction and **all** classes for each pixel.
<img width="600" alt="Cyclic lr" src="https://github.com/KanielDatz/UnnUNet/blob/master/UnnUnet_documentation/assets//entropyNM.png?raw=true">


# Our files and modifications
We tended to mark our changes with commenting #$ within the code files of nnUNet.
**Modified files in nnUNet package:**
|**file**| modification |
|---------------------------------------------------|--|
| `setup.py` | changed to support UnnUNet setup
|`nnunetv2\training\nnUNetTrainer\nnUNetTrainer.py`| modified for cyclic lr support and custom checkpoint saving|
|`nnunetv2\training\lr_scheduler\polylr.py`| modified for cyclic lr support|
|`nnunetv2\run\run_training.py`|modified for cyclic lr|
| `nnunetv2\utilities\utils.py` |   added an option for temperature in softmax|

**UnnUNET new files**
|**file**| modification |
|---------------------------------------------------|--|
|`nnunetv2\unnunet\predict_from_folder.py`| prediction using multiple checkpoints|
|`nnunetv2\unnunet\run_uncertainty_on_fold.py`| uncertainty map calculation|
|`nnunetv2\unnunet\uncertainty_utils.py`|utilities for UnnUNet|


# How to use
## Install
First, go to nnUNet [installation instructions ](documentation/installation_instructions.md) and make sure you have all prerequisites. 

next run :

          git clone https://github.com/KanielDatz/UnnUNet.git
          cd UnnUNet
          pip install -e .
     
UnnUNet needs to know where you intend to save raw data, preprocessed data, and trained models.
For this, you need to set a few environment variables. Please follow the instructions [here](documentation/setting_up_paths.md).

# Train

 *- instructions are given on 2d configuration only but can be implemented on all nnUnet configurations. If you wish to dive deeper we recommend reading nnUNet ['how to use file.](documentation/how_to_use_nnunet.md)*

**Dataset** - Before you can train your model, please prepare your dataset to match the dataset format that fits the nnUNet. [follow the instruction on nnUNet documentation.](documentation/dataset_format.md)
Place your dataset in UnnUNet_raw directory you set as discribed in  [here](# documentation/setting_up_paths.md)

To run preprocessing on the data:
`nnUNetv2_plan_and_preprocess -d DATASET --verify_dataset_integrity`
write your dataset id or name instead of `DATASET`

After running preprocessing the dataset fingerprint and training plans will be available in the UnnUnet_preproccesd directory.

For training each fold, run using bash:

     CUDA_VISIBLE_DEVICES=[Index of GPU] nnUNetv2_train [DATASET] 2d [FOLD] --npz -device cuda  -num_epochs [NUM_E] -num_of_cycles [Tc] -checkpoints [RULE]

**When:**

   `Index of GPU` - choose the index of the GPU you want to run on your machine
   
   `DATASET` - dataset name or id

   `FOLD` - which fold do you wish to train
   
   `Tc`  - Set to 1 to get the regular nnUNet. (default is 1)
   
   `NUM_E` - total number of epochs. (default is 1200)
   
   `RULE` - here you choose how you want to save the checkpoints -
   
   - `sparse` - will save 6 evenly spaced checkpoints each cycle, starting at 0.7*(epochs per cycle).
   - `late` - will save checkpoints from 10 last epochs of the cycle.
    
*you can adjust as wanted. we recommend first experimenting with one fold.*	

*try `nnUNetv2_train -h` for help!*

# Inference
on inference, we first run a prediction for each checkpoint to get uncertainty maps and then combine the results to output a prediction and uncertainty map.

**To get probability maps:**

 1. Make a directory with the images you wish to predict in the [nnUNet format.](documentation/dataset_format.md)
    
 3. run:
     `UnnUnet_predict_from_folder -dataset DATASET -fold FOLD -input_folder INPATH -output_folder OUTPATH -rule [RULE]`

**when:**

`DATASET` - dataset name or id

`FOLD` - which fold do you wish to train

`INPATH` - path to the folder with the images you want to predict.

`OUTPATH` path to the output folder for the probability maps.

`RULE` - here you choose how you want to save the checkpoints -`spars` or `late`
   
*try `UnnUnet_predict_from_folder -h` for help!*

**Now in OUTPATH folder you will have a folder for each checkpoint prediction.**

**To get an uncertainty map and score:**

 3. You need to choose the uncertainty method for calculation.
    
 5. run:
  `UnnUnet_run_uncertainty_on_fold --proba_dir PATH --raw_path PATH --labels PATH --score_type TYPE --outpot_pred_path PATH`

`--proba_dir` path to the folder with the checkpoints folders (output of the previous script)

`--raw_path` path to the folder with the dataset the user wants to predict ( input of the previous script)

`--labels` path to the labels of the dataset. optional, if given- the model will add dice to the final output if given.

`--score_type` The score type to use for the uncertainty score. default is `class_entropy` - other options are `total_entropy` and `t_test`

`--outpot_pred_path` path to the folder where the predictions will be saved. default is `proba_dir + /unnunet_pred`
   
# Acknowledgements
Our teachers:
-   **Prof. Daniel Soudry and TA Tal Daniel.** Electrical and Computer Engineering Department, Technion
    
-   **Dr. Moti Freiman and Eyal H.** Computational MRI Lab, Biomedical Engineering, Technion.

The papers we relied on:
1.  [**Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.** Nature methods, 18(2), 203-211.](https://www.google.com/url?q=https://www.nature.com/articles/s41592-020-01008-z&sa=D&source=docs&ust=1677235958581755&usg=AOvVaw3dWL0SrITLhCJUBiNIHCQO)
    
2.  [**Zhao, Y., Yang, C., Schweidtmann, A., Tao, Q. (2022). Efficient Bayesian Uncertainty Estimation for nnU-Net.** In: Wang, L., Dou, Q., Fletcher, P.T., Speidel, S., Li, S. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2022. MICCAI 2022. Lecture Notes in Computer Science, vol 13438. Springer, Cham.](https://doi.org/10.1007/978-3-031-16452-1_51) 
    
3.  [Vo Nguyen Le Duy and Shogo Iwazaki and Ichiro Takeuchi (2021). **Quantifying Statistical Significance of Neural Network Representation-Driven Hypotheses by Selective Inference**](https://openreview.net/forum?id=jC9G3ns6jH)
    
4.  [**Hossam El‐Rewaidy, Maryam Nezafat, Jihye Jang, Shiro Nakamori, Ahmed S. Fahmy, and Reza Nezafat. "Nonrigid active shape model–based registration framework for motion correction of cardiac T1 mapping." Magnetic resonance in medicine (2018), doi: 10.1002/mrm.27068**](https://cardiacmr.hms.harvard.edu/downloads-0)

Special Thanks to nnUNet developers!
nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).

