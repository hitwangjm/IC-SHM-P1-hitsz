## Implementation of semantic segmentation model in pytoch
## DeepLabv3+：Encoder-Decoder with Atrous Separable Convolution 
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
    #   Training our own data sets must be modified
    #   The number of categories you need + 1, such as 2 + 1
    #   labcmp:num_classes = 9
    #   labdmg:num_classes = 4
    # -------------------------------#
    num_classes = 9
    # -------------------------------------------------------------------#
    #   Backbone network used：mobilenet、xception 
    #   Test model:Divided into “labcmp、labdmg、labdmg_puretex”
    #   When using Xception as the backbone network, it is recommended to reduce the learning rate in the part of training parameter setting, such as:
    #   Freeze_lr   = 3e-4
    #   Unfreeze_lr = 1e-5
    # -------------------------------------------------------------------#
    backbone = "mobilenet"
    model_path = "model_data\deeplab_mobilenetv2.pth"
    #   The multiples of down sampling can be 8 or 16
    #   If 8 is selected, the down sampling multiple is smaller and the effect is better in theory, but larger video memory is also required
    # ---------------------------------------------------------#
    downsample_factor = 8
    # ------------------------------#
    #   VOCdevkit_path is the data set path, and datafolder is the training object folder path under the path (labcmp, labdmg or labdmg_puretex)
    # ------------------------------#
    VOCdevkit_path = 'SHMdata'
    datafolder = 'labcmp'
    # ----------------------------------------------------#
    #   The training is divided into two stages: freezing stage and Unfreezing stage.
    #   Insufficient video memory has nothing to do with the size of the dataset. It indicates that insufficient video memory. Please turn down batch_size。
    #   Affected by BatchNorm layer, batch_ The minimum size is 2 and cannot be 1.
    # ----------------------------------------------------#
    # ----------------------------------------------------#
    #   Freeze stage training parameters
    #   At this time, the backbone of the model is frozen, and the feature extraction network does not change
    #   The occupied video memory is small, and only fine tune the network
    # ----------------------------------------------------#
    Init_Epoch = 50
    Freeze_Epoch = 50
    Freeze_batch_size = 16
    Freeze_lr = 3e-4
    # ----------------------------------------------------#
    #   Unfreeze stage training parameters
    #   At this time, the backbone of the model is not frozen, and the feature extraction network will change
    #   The occupied video memory is large, and all parameters of the network will change
    # ----------------------------------------------------#
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 2
    Unfreeze_lr = 2e-8
    # ---------------------------#
    #   Read TXT corresponding to data set
    #   You need to put relevant txt files in this directory for training
    # ---------------------------#
    with open(os.path.join(VOCdevkit_path, datafolder, "ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()

    with open(os.path.join(VOCdevkit_path, datafolder, "ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()


### How2predict
#### a、Use pretraining weights
1、After downloading the library, unzip it. If you want to use the backbone for mobile prediction, run predict.py is OK;
If you want to use backbone for xception prediction, Download deeplab_xception.pth on Baidu Cloud, put in model_data, modify backbone of deeplab.py and model_path and then run predict.py, enter.
```python
img/test.jpg
```
The forecast can be completed.   
2、After setting in predict.py, FPS test, whole folder test and video detection can be carried out.    
#### b、Use your training weights
1、Follow the training steps。    
2、In deeplab.py, modify the model in the following model_path、num_classes and backbone make them correspond to the trained files
**model_path corresponds to the weight file under the logs folder, num_classes represents the number of classes to be predicted plus 1. Backbone is the backbone feature extraction network used.** 
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
3、run predict.py, and input:    
```python
img/test.jpg
```
The forecast can be completed.  
4、After setting in predict.py, FPS test, whole folder test and video detection can be carried out. 

### mIOU&mPA
1、Set num_classes in get_miou.py as the number of predicted classes plus 1.
2、Set name_classes in get_miou.py are the categories that need to be distinguished.  
3、Run get_miou.py, then get mIOU and mPA。  

### Reference
https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bonlime/keras-deeplab-v3-plus   
https://github.com/bubbliiiing/deeplabv3-plus-pytorch  
(My github)https://github.com/hitwangjm/IC-SHM-P1
