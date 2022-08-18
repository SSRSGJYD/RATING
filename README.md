# RATING

[![DOI](https://zenodo.org/badge/526008863.svg)](https://zenodo.org/badge/latestdoi/526008863) 

## Introduction

This project contains source code for our manuscript "RATING: Medical-knowledge-guided rheumatoid arthritis assessment from multimodal ultrasound images via deep learning".

## Environment

**Hardware requirement**

RATING system runs on a computer with NVIDIA GPUs. At least 8 GB GPU memory is needed.

**Software requirement**

A Python 3.6+ environment is needed with the packages in the  `requirements.txt` installed. 

## File Structure

```
./
├── checkpoints/                            # store experiment checkpoints
│   ├── DOPPLER/
│   ├── GS/
│   └── GSDOPPLER/
├── dataset_files/                          # put dataset files in the folder
├── datasets/                               # all datasets
│   ├── __init__.py
│   ├── base_dataset.py                     # base dataset class
│   ├── DOPPLER_dataset.py                  # dataset for Doppler US images
│   ├── GS_dataset.py                       # dataset for Greyscale US images
│   ├── GSDOPPLER_dataset.py                # dataset for paired Greyscale US and Doppler US images
│   ├── jigsaw_puzzle.py                    # dataset for self-supervised pre-training
│   └── permutations_1000.npy               # 1000 permutations for self-supervised pre-training
├── models/                             
│   ├── networks/                       
│   │   ├── backbone/                       # neural network architectures
│   │   ├── __init__.py/                
│   │   ├── cfn_net.py/                     # Context-free Network for self-supervised learning
│   │   └── GSDopplerFeatureFusion_net.py/  # GS-Doppler Feature Fusion Network
│   ├── __init__.py                     
│   ├── base_model.py                       # base model class
│   ├── cls_model.py                        # classification model
│   └── model_option.py
├── optim/                                  # optimizers
│   └── __init__.py/                    
├── options/                                # save model checkpoints
│   ├── __init__.py
│   ├── base_options.py                     # base option class
│   ├── DOPPLER_options.py                  # options for Doppler US images
│   ├── GS_options.py                       # options for Greyscale US images
│   └── GSDOPPLER_options.py                # options for paired Greyscale US and Doppler US images
├── schedulers/                             
│   ├── __init__.py
│   └── warmup_scheduler.py                 # multi-step scheduler with warm up
├── setting/                                
│   ├── __init__.py
│   ├── DOPPLER_feature_extractor.py        # setting of training Doppler US feature extraction network
│   ├── GS_feature_extractor.py             # setting of training GSUS feature extraction network
│   ├── GS_jigsaw.py                        # setting of self-supervised pre-training
│   ├── SH_classifier.py                    # setting of training GS-Doppler feature fusion networks
│   └── vascularity_classifier.py           # setting of training Doppler US classification networks
├── util/                                   # Utility tools
├── config.py                               # directory configurations 
├── do_train.py                             
├── do_train_jigsaw.py                      
├── self_supervised_pretraining.py          # script of pre-training
├── train_DOPPLER_feature_extractor.py      # script of training Doppler US feature extraction network
├── train_GS_feature_extractor.py           # script of training GSUS feature extraction network
├── train_SH_classifier.py                  # script of training GS-Doppler feature fusion network
├── train_vascularity_classifier.py         # script of training Doppler US classification network
├── test_models.py                          # run inference
├── MULTITUDE.py                            # run MULTITUDE algorithm
├── statistic_util.py                       # statistical analysis tools
├── thresh_dict                            
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Prepare Data

To build the system, data for training and validation are needed. Since RATING system adopts multi-task multi-model ensemble, data should be split into five folds, resulting in five training datasets and five validation datasets. For each dataset, a JSON file is needed to specify its information. They should be named as:

+ `train_split1.json`, `val_split1.json`

+ `train_split2.json`, `val_split2.json`

+ `train_split3.json`, `val_split3.json`

+ `train_split4.json`, `val_split4.json`

+ `train_split5.json`, `val_split5.json`

Each JSON file can be parsed as a JSON array of dictionaries representing the samples in the dataset. Each dictionary should have the following keys:

+ GS_path: path to the GSUS image.

+ GS_roi_anno: an array (left, top, right, bottom) which is the ROI annotation of the GSUS image.

+ DOPPLER_path: path to the Doppler US image.

+ DOPPLER_roi_anno: an array (left, top, right, bottom) which is the ROI annotation of the GSUS image.

+ SH_label: an integer representing the SH label.

+ vascularity_label: an integer representing the vascularity label.

## Build RATING System

**Step 1: self-supervised pre-training using GSUS images**

```shell
python self_supervised_pretraining.py
```

**Step 2: fine-tune GSUS and Doppler US feature extraction networks**

```shell
python train_GS_feature_extractor.py
python train_DOPPLER_feature_extractor.py
```

**Step 3: train classifiers of GS-Doppler feature fusion networks**

```shell
python train_SH_classifier.py
```

**Step 4: train vascularity classification networks**

```shell
python train_vascularity_classifier.py
```

## Inference

Prepare the test dataset, specify the path to the test dataset JSON file as `test_dataset` in `config.py`, and run model inference:

```shell
python test_models.py
```

When all the models finish inference, specify the output path as `save_path` in `config.py`, and run MULTITUDE algorithm to obtain final predictions:

```shell
python MULTITUDE.py
```

By default, there should be an output JSON file `prediction.json` in the root folder, which can be parsed as a JSON dictionary. It has three attributes:

+ SH: a numpy array of SH score predictions of all test samples.

+ VASCULARITY: a numpy array of vascularity score predictions of all test samples.

+ combined: a numpy array of combined score predictions of all test samples.

In addition, accuracy and linearly weighted kappa with 95% confidence interval should be printed on the console.
