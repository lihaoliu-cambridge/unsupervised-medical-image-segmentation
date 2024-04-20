# Unsupervised Medical Image Segmentation

by Lihao Liu, Angelica I Aviles-Rivero, and Carola-Bibiane Schönlieb.  


## Introduction

In this repository, we provide the PyTorch implementation for [Contrastive Registration for Unsupervised Medical Image Segmentation](https://arxiv.org/abs/2011.08894). 

<img src="https://github.com/lihaoliu-cambridge/lihaoliu-cambridge.github.io/blob/master/pic/papers/unsupervised-segmentation-results.png">  


## Requirement

torch                       1.5.0  
torchvision                 0.4.2  
SimpleITK                   1.2.4  
opencv-python               4.2.0.32  


## Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/lihaoliu-cambridge/unsupervised-medical-image-segmentation.git
   cd unsupervised-medical-image-segmentation
   ```
   
2. Download the images and segmentation masks for LPBA40 dataset:

   LPBA40 Images: [LPBA40_rigidly_registered_pairs.tar.gz](https://www.synapse.org/#!Synapse:syn3251419)  
   LPBA40 Labels: [LPBA40_rigidly_registered_label_pairs.tar.gz](https://www.synapse.org/#!Synapse:syn3251070)  
   
3. Unzip them in folder `datasets/LPBA40`:

   `datasets/LPBA40/LPBA40_rigidly_registered_pairs`  
   `datasets/LPBA40/LPBA40_rigidly_registered_label_pairs`  
   
4. Pre-process the LPBA40 dataset:

   ```shell
   cd scripts
   python preprocessing_lpba40.py
   ```
   
   output small image results:
   
   `datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small`  
   `datasets/LPBA40/LPBA40_rigidly_registered_label_pairs_small`

   output large image results:
   
   `datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_large`  
   `datasets/LPBA40/LPBA40_rigidly_registered_label_pairs_large`
   
   This step aims to standardize the distribute of all images in a similar range:  
   <img src="https://github.com/lihaoliu-cambridge/lihaoliu-cambridge.github.io/blob/master/pic/papers/unsupervised-segmentation-histogram_standardization.png" width="360"/>  
   
   
6. Train the model:
 
   ```shell
   cd ..
   python train.py  --no_html  --dataroot ./datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small  --dataset_mode lpba40_contrastive_learning  --batchSize 8  --lr 0.003  --model registration_model_contrastive_learning  --name lpba40_contrastive_learning

   ```

7. Test the saved model:
 
   ```shell
   python test_dice.py  --no_html  --dataroot ./datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small  --dataset_mode lpba40_contrastive_learning  --batchSize 1  --model registration_model_contrastive_learning  --name lpba40_contrastive_learning


   ```

## Citation

If you use our code for your research, please cite our paper:

```
@article{liu2020contrastive,
  title={Contrastive Registration for Unsupervised Medical Image Segmentation},
  author={Liu, Lihao and Aviles-Rivero, Angelica I and Sch{\"o}nlieb, Carola-Bibiane},
  journal={arXiv preprint arXiv:2011.08894},
  year={2020}
}
```


## Question

Please open an issue or email lhliu1994@gmail.com for any questions.
