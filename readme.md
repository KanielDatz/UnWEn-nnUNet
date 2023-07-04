## Welcome to UnnUNet!

**This repisatory is a part of our final project in [EE046211 Deep Learning course](https://github.com/taldatech/ee046211-deep-learning) in the [Technion - Israel Institute of Technology](https://en.wikipedia.org/wiki/Technion_%E2%80%93_Israel_Institute_of_Technology)
By Daniel Katz and Natalie Mendelson, spring 2023**
- [Welcome to UnnUNet!](##Welcome to UnnUNet!)
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
(read about extraction types in [Train](##train)

<img width="600" alt="Cyclic lr" src="https://github.com/KanielDatz/UnnUNet/blob/master/UnnUnet_documentation/assets/progress.png?raw=true">

Next we make predictions (probabilities) of each class from the extracted checkpoints. This predictions and the varience between them is used to asses prediction ancertainty.
Our pipline is able to produce uncertainty map and score in three different methods, as discribed in [choosing uncertainty metric](#choosing).
nnUnet trains 5 fold cross validation by defult, but less is also possible.
For each fold the prediction of the model is done using the checkpoint with the best dice metric in the training.
on enference, all 5 folds are used to make predictions, the final prediction of the model is chosen by taking the prediction that has lowest uncertainty score.


# Our files and modifactions
**HERE WE WRITE OUR MODIFICATIONS ON NNUNET FILES**
**HERE COMES A TABLE OF FILES AND FUNCTIONS**


# How to use
## Install
First got to nnUNet [installation instructions ](documentation/installation_instructions.md) and make sure you have all pre requesits. 
next run:
          ```bash
          git clone https://github.com/KanielDatz/UnnUNet.git
          cd UnnUNet
          pip install -e .
      ```
3) UnnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to
   set a few environment variables. Please follow the instructions [here](# documentation/setting_up_paths.md).

## Train

 *- instructions are given on 2d configaration only but can be implemented on all nnUnet configarations. If you wish to dive deeper we recommand reading nnUNet ['how to use' file.](documentation%20/how_to_use_nnunet.md)*

**Dataset** - Before you can train your model, please prepare your dataset to match the dataset format that fits the nnUNet. [follow the instruction on nnUNet documantation.](documentation/dataset_format.md)
Place your dataset in UnnUNet_raw directory you set as discribed in  [here](# documentation/setting_up_paths.md)

To run preprocessing on the data:
`nnUNetv2_plan_and_preprocess -d DATASET --verify_dataset_integrity`
write your dataset id or name instead of `DATASET`

After running preprocessing the dataset fingerprint and training plans will be availible in UnnUnet_preproccesd directory.

For training each fold , run using bash:

     CUDA_VISIBLE_DEVICES=[Index of GPU] nnUNetv2_train [DATASET] 2d [FOLD] --npz -device cuda  -num_epochs [NUM_E] -num_of_cycles [Tc] -checkpoints [RULE]

**when:**
   `Index of GPU` - choose the index of the GPU you want to run on your machine
   `DATASET` - dataset name or id
   `FOLD` - which fold you wish to train
   `Tc`  - Set to 1 to get the regular nnUNet. (defult is 1)
   `NUM_E` - total number of epochs. (defult is 1200)
   `RULE` - here you choose how you want to save the checkpoints -
	   `sparse` - will save 6 evenly spaced checkpoints each cycle, starting at 0.7*(epochs per cycle).
	   `late` - will save checkpoints from 10 last epochs of the cycle.
*you can adjust as wanted. we recomand first experimenting with one fold.*	   
*try `nnUNetv2_train -h` for help!*

# Enference
on enference we first run prediction for each checkpoint to get uncertainty maps and then combine the results to outpot a prediction and uncertainty map.
**To get probability maps:**
 1. Make a directory with the images you wish to predict in the [nnUNet format.](%28documentation/dataset_format.md%29) 
 2. run: `UnnUnet_predict_from_folder -dataset DATASET -fold FOLD -input_folder INPATH -output_folder OUTPATH -rule [RULE]`
   when:
     `DATASET` - dataset name or id
   `FOLD` - which fold you wish to train
   `INPATH` - path to the foldr with the images you want to predict.
    `OUTPATH` path to output folder for the probability maps.
   `RULE` - here you choose how you want to save the checkpoints -`spars` or `late`
*try `UnnUnet_predict_from_folder -h` for help!*
**Now in oupath folder you will have a folder for each checkpoint prediction.**

**To get uncertainty map and score:**

 3. You need to choose the uncertainty method for calculation, see [explanation](#uncertainty%20calculation).
 4. run:
  `UnnUnet_run_uncertainty_on_fold --proba_dir PATH --raw_path PATH --labels PATH --score_type TYPE --outpot_pred_path PATH`

`--proba_dir` path to the folder with the checkpoints folders (output of previous script)
`--raw_path` path to the folder with the dataset the user wants to predict ( input of previous script)
`--labels` path to the labels of the dataset. optional, if given- the model will add dice to the final output if given.
`--score_type` The score type to use for the uncertainty score. default is `class_entropy` - other options are `total_entropy` and `t_test`

`--outpot_pred_path` path to the folder where the predictions will be saved. default is `proba_dir + /unnunet_pred`
    







# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).

