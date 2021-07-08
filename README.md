# Pytorch Medical Classification
<i>Read Chinese Introduction：<a href='https://github.com/MontaEllis/Pytorch-Medical-Classification/blob/main/README-zh.md'>Here！</a></i><br />

## Recent Updates
* 2021.7.7 The train and test codes are released.


## Requirements
* pytorch1.7
* torchio<=0.18.20
* python>=3.6

## Notice
* You can modify **hparam.py** to determine whether 2D or 3D classification and whether multicategorization is possible.
* We provide algorithms for almost all 2D and 3D classification.
* This repository is compatible with almost all medical data formats(e.g. png, nii.gz, nii, mhd, nrrd, ...), by modifying **fold_arch** in **hparam.py** of the config.
* If you want to use a **multi-category** program, please modify the corresponding codes in **data_function.py** by yourself. I cannot identify your specific categories.

## Prepare Your Dataset
### Example
if your source dataset is :
```
categpry-0
├── source_1.png
├── source_2.png
├── source_3.png
└── ...
```

```
categpry-1
├── source_1.png
├── source_2.png
├── source_3.png
└── ...
```


then your should modify **fold_arch** as **\*.png**, **source_train_0_dir** as **categpry-0** and **source_train_1_dir** as **categpry-1** in **hparam.py**



## Training
* without pretrained-model
```
set hparam.train_or_test to 'train'
python main.py
```
* with pretrained-model
```
set hparam.train_or_test to 'train'
set hparam.ckpt to True
python main.py
```
  
## Inference
* testing
```
set hparam.train_or_test to 'test'
python main.py
```


## Done
### Network
* 2D
- [x] alexnet
- [x] densenet
- [x] googlenet
- [x] mobilenet
- [x] nasnet
- [x] resnet
- [x] resnext
- [x] vggnet
* 3D
- [x] densenet3d
- [x] resnet3d
- [x] resnext3d



## TODO
- [ ] dataset
- [ ] benchmark


## By The Way
This project is not perfect and there are still many problems. If you are using this project and would like to give the author some feedbacks, you can send [Kangneng Zhou](elliszkn@163.com) an email, his **wechat** number is: ellisgege666

## Acknowledgements
This repository is an unoffical PyTorch implementation of Medical segmentation in 3D and 2D and highly based on [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100) and [torchio](https://github.com/fepegar/torchio).Thank you for the above repo. Thank you to [Cheng Chen](b20170310@xs.ustb.edu.cn) and [Weili Jiang](1379252229@qq.com) for all the help I received.
