# Pytorch Medical Segmentation
<i>英文版请戳：<a href='https://github.com/MontaEllis/Pytorch-Medical-Classification/blob/main/README.md'>这里！</a></i><br />


## 最近的更新
* 2021.7.7 训练和测试代码已经发布


## 环境要求
* pytorch1.7
* torchio<=0.18.20
* python>=3.6

## 通知
* 您可以修改**hparam.py**文件来确定是2D分类还是3D分类以及是否可以进行多分类。
* 我们几乎提供了所有的2D和3D分类的算法。
* 本项目兼容几乎所有的医学数据格式(例如 png, nii.gz, nii, mhd, nrrd, ...)，修改**hparam.py**的**fold_arch**即可。
* 如果您想进行**多分类**分割，请自行修改**data_function.py**的对应代码。我不能确定您的具体分类数。



## 准备您的数据
### 例
如果您的source文件夹如下排列 :
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

您应该修改 **fold_arch** 为 **\*/\*.png**, **source_train_0_dir** 为 **categpry-0** 并修改 **source_train_1_dir** 为 **categpry-1** in **hparam.py**



## 训练
* 不使用预训练模型
```
set hparam.train_or_test to 'train'
python main.py
```
* 使用预训练模型
```
set hparam.train_or_test to 'train'
set hparam.ckpt to True
python main.py
```
  
## Inference
* 测试
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
这个项目并不完美，还存在很多问题。如果您正在使用这个项目，并想给作者一些反馈，您可以给[Kangneng Zhou](elliszkn@163.com)发邮件，或添加他的**微信**：ellisgege666

## 致谢
这个项目是一个非官方PyTorch实现的3D和2D医学分类，高度依赖于[pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)和[torchio](https://github.com/fepegar/torchio)。感谢上述项目。感谢[Cheng Chen](b20170310@xs.ustb.edu.cn)和[Weili Jiang](1379252229@qq.com)对我的帮助。
