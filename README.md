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
1、This project uses VOC format for training.(Because the author is Chinese, many comments on the program source code are written in Chinese, but actually it does not affect the use. Those requiring special comments have been annotated in English.)  
2、In this competition, excel2txt.py is used to convert the official CSV file into TXT format applicable to the algorithm, in which pred_path(.scvfile directory) and t_type(3 parameters are optional, labcmp、labdmg、labdmg_puretex) needs to be modified.
3、In the train.py, select the backbone model and down sampling factor you want to use. The backbone models provided in this paper are mobilenet and xception. The down sampling factor can be selected from 8 and 16. It should be noted that the pre training model should correspond to the backbone model.
4、Pay attention to modifying the num_classes of the train.py as the number of categories + 1.    
5、Run the train.py to start training (you need to modify the source program in the train.py according to the following requirements first).

#### Part to be modified in train.py
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
2、In predict.py, modify the model in the following model_path、num_classes and backbone make them correspond to the trained files
**model_path corresponds to the weight file under the logs folder, num_classes represents the number of classes to be predicted plus 1. Backbone is the backbone feature extraction network used.** 
```python
  # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到generate函数里修改self.colors即可
    #   get_mode表示测试的模型，分为“labcmp、labdmg、labdmg_puretex”
    # -------------------------------------------------------------------------#
    pr_model_path = r'logs\220102xception损伤检测桥梁ep100-loss0.065-val_loss0.083.pth'
    get_mode = 'labcmp'
    if get_mode == 'labcmp':
        # ------------------------------#
        #   分类个数+1、如2+1
        # ------------------------------#
        num_classes = 9
    else:
        num_classes = 4
        backbone = 'xception'
        downsample_factor = 8
    deeplab = DeeplabV3(pr_model_path, num_classes, backbone, downsample_factor)
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'txt_predict'表示训练集体输出（特殊模式），后期完善此注释！！！
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "txt_predict"
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    # -------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    # -------------------------------------------------------------------------#
    test_interval = 100
    # -------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "SHMdata/JPEGImages"
    dir_save_path = "img_out/"

 
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
