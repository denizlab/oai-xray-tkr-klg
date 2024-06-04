# Prediction of Total Knee Replacement and Diagnosis of Osteoarthritis using Deep Learning on Knee Radiographs: Data from the Osteoarthritis Initiative

## Introduction
This repo contains implementation of the deep learning-based outcome prediction model used for osteoarthritis research as described in our paper: [Prediction of Total Knee Replacement and Diagnosis of Osteoarthritis using Deep Learning on Knee Radiographs: Data from the Osteoarthritis Initiative](https://doi.org/10.1148/radiol.2020192091). By using this implementation, you can either train new models using nested cross-validation or obtain TKR outcome and KL grade predictions by using our pretrained models. 

[![license](https://img.shields.io/badge/license-AGPL--3.0-brightgreen)](https://github.com/denizlab/oai-xray-tkr-klg/blob/master/LICENSE)

## Instructions
1. Please refer to `requirements.txt` to install all dependencies for this project. 
2. Check **Extract L&R Knee Radiographs** section to extract and save left and right knee images from bilateral posterioranterior(PA) fixed-flexion knee radiographs of the patients to the `./data` folder automatically. 
3. Download pretrained torchvision ResNet-34 model from [here](https://download.pytorch.org/models/resnet34-333f7ec4.pth) and save it inside the main folder.
4. Once you have data ready, you can use `train_TL_nestedCV_strata.py` to train a model with seven-fold nested cross-validation. Trained models and inference results will be saved within the folder named 'model_weights_multiTask...'
5. Use `inference.py `file to obtain TKR outcome and KL grade predictions by using our pretrained model. Please see **Inference** section for details.

## Repo Structure
* `728_Cohort_KLG_w_Strata.csv` file includes subject IDs for each patient who are included in the study. Seperate columns identify: TKR status of patients (0: controls, 1:patients underwent TKR within 9 years from baseline), knee side (0: Left Knee, 1: Right knee), KL grade from the patient's knee and strata (from case control matching) for each patient included in the study. 
* `./data`: Folder for left or right knee radiographs of the patients that will be used for training. Once the `./ExtractKnee/preprocessing.py` file is executed this folder will be populated by radiographs defined in `./ExtractKnee/output_00m.csv` file .
* `./TestSets`: contains information about subjects who are included seven seperate groups for nested cross-validation used in this study. Filenames match with the "Test Set Numbers" defined in Table 5 of the paper.
* `./ModelWeights`: Trained model weights used in this study. They can be downloaded from [here](https://drive.google.com/file/d/1Ovf4KpZ0pjEyDstt7fA7HNQkcN7fg2Vj/view?usp=sharing).
* `./ExtractKnee`: contains the code and .csv files to extract single knee images from bilateral PA fixed-flexion knee radiographs

## Extract L&R Knee Radiographs
When you have an annotation `.csv` file, this code enables you to extract knee joint images from bilateral posteroanterior (PA) fixed-flexion knee radiographs in the OAI dataset.`output_00m.csv` file was used to extract knee joint images that we used for the paper. 
The annotation file looks like
```
file_path,pred_bbox
0.C.2/9003380/20041206/00429603/001,"[0.58988559 0.24353482 0.86547155 0.57789375 0.14393548 0.25309042 0.41856452 0.5861953 ]"
0.E.1/9004175/20050503/00787104/001,"[0.65715023 0.29639241 0.94106405 0.64110759 0.08047989 0.30884113 0.36594869 0.65544459]"
```
Column `file_path` is where radiographs are stored as a DICOM  file. The second column `pred_bbox` are the coordinates of bounding boxes for left and right knees. In the knee joint extraction code, we used regular expression to parse this string. Note that these coordinates format is different compared to the original [file](https://github.com/MIPT-Oulu/DeepKnee/blob/master/Dataset/OAI_test.csv) used in previous studies. The coordinates are saved as a float number between 0 to 1 defining the ratio of pixel locations with respect to image size instead of exact coordinates within the image. 

You can use `preprocessing.py` to generate right and left knee images of size knee joint area with size 1024×1024 in HDF5 file format from bilateral PA fixed-flexion knee radiograph DICOMs. Example use would be: 
```bash
# this code will use 'output_00m.csv' to generate single knee image dataset.
python preprocessing.py -m 00m --content ./output_00m.csv --output-dir ../data/
```
`-m` sets the sub-folder name for where you want to save HDF5 files. `--content` gives path for annotation files. `--output-dir` is the folder where you want to save images with the subfolder from `-m`.

## Training a DL model
You can use this repo to train models for predicting TKR outcome and KL grade. Default parameters are defined within `train_TL_nestedCV_strata.py` file. So, you directly run the following script to train DL models with nested cross-validation. 
```bash
python train_TL_nestedCV_strata.py
```
Please see the arguments in the .py file to make input argument changes if you need. 

## Inference
Once you download pretrained model weights from [here](https://zenodo.org/doi/10.5281/zenodo.11479343) to the `./ModelWeights` folder, you can use the following script to obtain predictions of TKR and KL grade for a specific single knee radiograph (e.g. for the left knee of patient with ID=9011918):
```bash
python inference.py --filename ./data/00m/9011918_00m_LEFT_KNEE.hdf5
```
The output of this script will provide the predictions of TKR and KL grades as shown below.
```
--- Inference Results ---
Predictions for ** 9011918_00m_LEFT_KNEE.hdf5 **
Total Knee Replacement (TKR): 0.95
KL grade 0: 0.00
KL grade 1: 0.01
KL grade 2: 0.02
KL grade 3: 0.92
KL grade 4: 0.04
```

## License
This repository is licensed under the terms of the GNU AGPLv3 license.

## Reference
If you found this code useful, please cite our paper:

*Prediction of Total Knee Replacement and Diagnosis of Osteoarthritis using Deep Learning on Knee Radiographs: Data from the Osteoarthritis Initiative*
Kevin Leung, Bofei Zhang, Jimin Tan, Yiqiu Shen, Krzysztof J. Geras, James S. Babb, Kyunghyun Cho, Gregory Chang, Cem M. Deniz
Radiology
2020
```
@article{leung2020osteoarthritis,
    title = {Prediction of Total Knee Replacement and Diagnosis of Osteoarthritis using Deep Learning on Knee Radiographs: Data from the Osteoarthritis Initiative},
    author = {Kevin Leung and Bofei Zhang and Jimin Tan and Yiqiu Shen and Krzysztof J. Geras and James S. Babb and Kyunghyun Cho and Gregory Chang and Cem M. Deniz}, 
    journal = {Radiology},
    year = {2020},
    doi = {10.1148/radiol.2020192091},
    URL = {https://doi.org/10.1148/radiol.2020192091}
}
```
