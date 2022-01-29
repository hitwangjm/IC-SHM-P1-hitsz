## Implementation of semantic segmentation model in pytoch
## DeepLabv3+：Encoder-Decoder with Atrous Separable Convolution 
---
### CONTENTS
1. [Performance](#Performance)
2. [Environment](#Environment)
3. [Attention: Matters needing attention](#Attention)
4. [Download:File download](#Download)
5. [How2train:Training steps](#How2train)
6. [How2predict:Prediction steps](#How2predict)
7. [mIOU&mPA:Evaluation steps](#mIOU&mPA)
8. [Reference](#Reference)

### Performance
| Train dataset | Weight filename | Test dataset | Input picture size | mIOU | 
| :-----: | :-----: | :------: | :------: | :------: | 
| VOC12+SBD | [deeplab_mobilenetv2.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_mobilenetv2.pth) | VOC-Val12 | 512x512| 72.59 | 
| VOC12+SBD | [deeplab_xception.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_xception.pth) | VOC-Val12 | 512x512| 76.95 | 

### Environment
torch==1.2.0(CUDA9/CUDA10，which is supported by NVIDIA RTX20 Series graphics card and previous versions)(Please refer to requirements_cuda10.txt for other requirements)
torch==1.7.1(CUDA11.0, which is supported by NVIDIA RTX30 Series graphics card)

### Attention   
The deeplab_mobilenetv2.pth and deeplab_xception.pth in the code are based on VOC extended dataset training. Pay attention to modifying the backbone during training and prediction.

### Download
Backbone network file required for training: deeplab_mobilenetv2.pth and deeplab_xception.pth can be downloaded from Baidu Cloud or from this link.
Baidu Cloud: https://pan.baidu.com/s/10l5_HaNXw7ZDU_4Mgfu1lA 
Extraction Code: 4p4g

Tokaido Dataset：Onedrive network disk of Tokaido Dataset is as follows (temporarily unavailable):  
UofH OneDrive: https://uofh-my.sharepoint.com/:f:/g/personal/vhoskere_cougarnet_uh_edu/EqwAVkOiGPhLrgw6bvoBWA8B4TpcCIgSYGhw8viH56RRpQ?e=ltL0Xo  

### How2train
1、本文使用VOC格式进行训练。(Because the author is Chinese, many comments on the program source code are written in Chinese, but actually it does not affect the use. Those requiring special comments have been annotated in English.)  
2、本次比赛，使用excel2txt.py将官方给的csv文件转化为算法适用的txt格式，其中需要修改pred_path(.scv文件目录)和t_type(可选labcmp、labdmg、labdmg_puretex)。
3、在train.py文件夹下面，选择自己要使用的主干模型和下采样因子。本文提供的主干模型有mobilenet和xception。下采样因子可以在8和16中选择。需要注意的是，预训练模型需要和主干模型相对应。   
4、注意修改train.py的num_classes为分类个数+1。    
5、运行train.py即可开始训练（需要先按照下面要求修改train.py中源程序）。
#### train.py中需要修改的部分
 # -------------------------------#
    #   训练自己的数据集必须要修改的
    #   自己需要的分类个数+1，如2+1
    #   labcmp:num_classes = 9
    #   labdmg:num_classes = 4
    # -------------------------------#
    num_classes = 9

    # -------------------------------------------------------------------#
    #   所使用的的主干网络：mobilenet、xception 
    #   测试的模型，分为“labcmp、labdmg、labdmg_puretex”
    #   在使用xception作为主干网络时，建议在训练参数设置部分调小学习率，如：
    #   Freeze_lr   = 3e-4
    #   Unfreeze_lr = 1e-5
    # -------------------------------------------------------------------#
    backbone = "mobilenet"
    model_path = "model_data\deeplab_mobilenetv2.pth"
    #   下采样的倍数8、16 
    #   8下采样的倍数较小、理论上效果更好，但也要求更大的显存
    # ---------------------------------------------------------#
    downsample_factor = 8
    # ------------------------------#
    #   VOCdevkit_path是数据集路径,datafolder是路径下的训练对象文件夹路径(labcmp或labdmg或者labdmg_puretex)
    # ------------------------------#
    VOCdevkit_path = 'SHMdata'
    datafolder = 'labcmp'




### How2predict
#### a、使用预训练权重
1、下载完库后解压，如果想用backbone为mobilenet的进行预测，直接运行predict.py就可以了；如果想要利用backbone为xception的进行预测，在百度网盘下载deeplab_xception.pth，放入model_data，修改deeplab.py的backbone和model_path之后再运行predict.py，输入。  
```python
img/street.jpg
```
可完成预测。    
2、在predict.py里面进行设置可以进行fps测试、整个文件夹的测试和video视频检测。       

#### b、使用自己训练的权重
1、按照训练步骤训练。    
2、在deeplab.py文件里面，在如下部分修改model_path、num_classes、backbone使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，num_classes代表要预测的类的数量加1，backbone是所使用的主干特征提取网络**。    
```python
_defaults = {
    #----------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #----------------------------------------#
    "model_path"        : 'model_data/deeplab_mobilenetv2.pth',
    #----------------------------------------#
    #   所需要区分的类的个数+1
    #----------------------------------------#
    "num_classes"       : 21,
    #----------------------------------------#
    #   所使用的的主干网络
    #----------------------------------------#
    "backbone"          : "mobilenet",
    #----------------------------------------#
    #   输入图片的大小
    #----------------------------------------#
    "input_shape"       : [512, 512],
    #----------------------------------------#
    #   下采样的倍数，一般可选的为8和16
    #   与训练时设置的一样即可
    #----------------------------------------#
    "downsample_factor" : 16,
    #--------------------------------#
    #   blend参数用于控制是否
    #   让识别结果和原图混合
    #--------------------------------#
    "blend"             : True,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
3、运行predict.py，输入    
```python
img/street.jpg
```
可完成预测。    
4、在predict.py里面进行设置可以进行fps测试、整个文件夹的测试和video视频检测。   

### mIOU&mPA
1、设置get_miou.py里面的num_classes为预测的类的数量加1。  
2、设置get_miou.py里面的name_classes为需要去区分的类别。  
3、运行get_miou.py即可获得miou大小。  

### Reference
https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bonlime/keras-deeplab-v3-plus
