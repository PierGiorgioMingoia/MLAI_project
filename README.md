# JiGen AIMLProject  <img src="https://github.com/silvia1993/Jigen_AIMLProject/blob/main/aiml.png" align="right" width="200">

Follow these steps to run the code on Google Colab, to reproduce experiments in [Self supervision for Domain Generalization and Adaptation](https://github.com/PierGiorgioMingoia/MLAI_project/blob/main/Self%20supervision%20for%20Domain%20Generalization%20and%20Adaptation.pdf)

## Dataset

1 - Download PACS dataset from here http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017

2 - Place the dataset in the MLAI_project folder making sure that the images are organized in this way:

```
PACS/kfold/art_painting/dog/pic_001.jpg
PACS/kfold/art_painting/dog/pic_002.jpg
PACS/kfold/art_painting/dog/pic_003.jpg
...
```
3 - If you want to use the PACS folder already saved in the Google Drive set this argument when run the experiments:
```
--path_dataset /content/drive/MyDrive/MLAI_project

```


## Pretrained models

1 - Create a link in your Google Drive to this folder https://drive.google.com/drive/folders/1NdMyhr76I1kQG3kX3PpidW5ZUYmpyMZN?usp=sharing (this folder contains all pretrained models used in the code)

2 - Mount your drive in Google Colab

## Environment

To run the code you have to install all the required libraries listed in the "requirements.txt" file.

you have to execute the command:

```
!pip install -r /content/MLAI_project/requirements.txt

```

## Experiments

### Domain Generalization
```
!python /content/MLAI_project/train_DG.py --source photo cartoon sketch --target art_painting --alpha_parameter 0.6 --beta_parameter 0.6 --n_jigsaw_classes 30 --n_tiles 3

```
### Domain Adaptation
```
!python /content/MLAI_project/train_DA.py --source photo cartoon sketch --target art_painting --alpha_parameter 0.6 --beta_parameter 0.8 --jigloss_target_parameter 0.7 --n_jigsaw_classes 30 --n_tiles 3

```

## Methods 

Add these arguments to perform:

### Rotation
```
--is_rotation True --is_grayscale True

```
### AdaIN style transfer
```
--is_AdaIN True

```






