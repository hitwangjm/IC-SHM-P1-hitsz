## Implementation of semantic segmentation model in pytoch
## DeepLabv3+：Encoder-Decoder with Atrous Separable Convolution 
---

### CONTENTS
1. [Performance](#Performance)
2. [Environment](#Environment)
3. [Program introduction](#Program)
4. [Attention: Matters needing attention](#Attention)
5. [Download:File download](#Download)
6. [How2train:Training steps](#How2train)
7. [How2predict:Prediction steps](#How2predict)
8. [mIOU&mPA:Evaluation steps](#mIOU&mPA)
9. [Reference](#Reference)

### Performance
| Train dataset | Weight filename | Test dataset | Input picture size | mIOU |  mPA | 
| :-----: | :-----: | :------: | :------: | :------: | :------: |
| Tokaido_damage_dataset_train | [deeplab_mobilenetv2.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_mobilenetv2.pth) | Tokaido_damage_dataset_test | 512x512| 66.43 | 77.59 |
| Tokaido_damage_dataset_train | [deeplab_xception.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_xception.pth) | Tokaido_damage_dataset_test | 512x512| 65.9 | 75.53 |
| Tokaido_component_dataset_train | [deeplab_mobilenetv2.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_mobilenetv2.pth) | Tokaido_component_dataset_test | 512x512| 71.08 | 75.33 | 
| Tokaido_component_dataset_train | [deeplab_xception.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_xception.pth) | Tokaido_component_dataset_test | 512x512| 66.80 | 74.06 | 

### Environment
torch==1.2.0    
(CUDA9/CUDA10，which is supported by NVIDIA RTX20 Series graphics card and previous versions)(Please refer to requirements_cuda10.txt for other requirements)

torch==1.7.1    
(CUDA11.0, which is supported by NVIDIA RTX30 Series graphics card)

### Program
#### First, introduce the main py files
train.py：main training program      
predict.py:main predictor       
get_miou.py：program for calculating mIOU and mPA      
excel2txt.py：convert the CSV form given by the official competition to TXT format
deeplab.py：neural network program       
submission_helper.py：competition result submission procedure       
machine2man.py：color the mask (it can not be used)      
summary.py：program to view the results of neural network (it can not be used)      
fig_plot.py：draw damage function image (can not be used)      
json_to_dataset.py：data set format conversion (can not be used)      
voc_annotation.py：the program used to divide training set, test set and verification set (it can not be used)     

##### Then, introduce the main folders
logs：the folder where the training result file is saved  
model_data：the folder where the backbone network files are saved   
SHMdata：folder for saving train set, test set and validation set      
nets：the program corresponding to the backbone network       
utils：other important functions 

### Attention   
The deeplab_mobilenetv2.pth and deeplab_xception.pth in the code are based on VOC extended dataset training. Pay attention to modifying the backbone during training and prediction.

### Download
Backbone network file required for training: deeplab_mobilenetv2.pth and deeplab_xception.pth can be downloaded from Baidu Cloud or from Google drive.    
Baidu Cloud: https://pan.baidu.com/s/194LVYFNTtTutL_WgM4DjzA   
Extraction Code: 2021  
Google drive: https://drive.google.com/drive/folders/19HYqKXi662qB-hZd8NCeAfZP78sA56kW?usp=sharing  

Tokaido Dataset：Onedrive network disk of Tokaido Dataset is as follows (temporarily unavailable):  
UofH OneDrive: https://uofh-my.sharepoint.com/:f:/g/personal/vhoskere_cougarnet_uh_edu/EqwAVkOiGPhLrgw6bvoBWA8B4TpcCIgSYGhw8viH56RRpQ?e=ltL0Xo  

### How2train

**1、I have uploaded the first step to kaggle where you can choose to download your datasets directly on kaggle.  
kaggle link: https://www.kaggle.com/dataset/223739a153789866e7336123dae24bac12a13926a6089324c085c5aa956c1c3d

[Optional]
Competition officials gave Tokaido_dataset. Before training, you need to put the relevant data into the corresponding folder
a）Original picture: first, put all the picture in Tokaido_dataset/images_puretex and Tokaido_dataset/img_syn_raw to ./SHMdata/JPEGImages.
b）mask: then, put the mask in Tokaido_dataset/synthetic/labdmg and Tokaido_dataset/synthetic/test/labcmp to ./SHMdata/labcmp/SegmentationClass; put mask in Tokaido_dataset/synthetic/train/labdmg and Tokaido_dataset/synthetic/test/labdmg to ./SHMdata/labdmg/SegmentationClass(at the same time, make a copy and put it in ./SHMdata/labdmg_puretex/SegmentationClass).
c)mask: finally, put the mask in Tokaido_dataset/synthetic_puretex/train/labdmg to ./SHMdata/labdmg_puretex/SegmentationClass.
[Optional]

2、This project uses VOC format for training.(Because the author is Chinese, many comments on the program source code are written in Chinese, but actually it does not affect the use. Those requiring special comments have been annotated in English.)  
3、In this competition, excel2txt.py is used to convert the official CSV file into TXT format applicable to the algorithm, in which pred_path(.scvfile directory) and t_type(3 parameters are optional, labcmp、labdmg、labdmg_puretex) needs to be modified.      
4、In the train.py, select the backbone model and down sampling factor you want to use. The backbone models provided in this paper are mobilenet and xception. The down sampling factor can be selected from 8 and 16. It should be noted that the pre training model should correspond to the backbone model.      
5、Pay attention to modifying the num_classes of the train.py as the number of categories + 1.    
6、Run the train.py to start training (you need to modify the source program in the train.py according to the following requirements first).      

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
If you want to use backbone for xception prediction, Download deeplab_xception.pth on Baidu Cloud, put in model_data, modify backbone of deeplab.py and model_path and then run predict.py, enter:  
```python
img/test.jpg
```
The forecast can be completed.   

2、After setting in predict.py, FPS test, whole folder test and video detection can be carried out.    
#### b、Use your training weights
1、Follow the training steps。    
2、In predict.py, modify the model in the following model_path、num_classes and backbone make them correspond to the trained files.  **model_path corresponds to the weight file under the logs folder, num_classes represents the number of classes to be predicted plus 1. Backbone is the backbone feature extraction network used.** 
```python
    # -------------------------------------------------------------------------#
    #   If you want to modify the color of the corresponding category, you can modify self.colors in the generate function
    #   get_mode refers to the test model, which is divided into "labcmp, labdmg, labdmg_puretex"
    # -------------------------------------------------------------------------#
    pr_model_path = r'logs\220102xception损伤检测桥梁ep100-loss0.065-val_loss0.083.pth'
    get_mode = 'labcmp'
    if get_mode == 'labcmp':
        # ------------------------------#
        #   Number of classifications + 1, e.g. 2 + 1
        # ------------------------------#
        num_classes = 9
    else:
        num_classes = 4
        backbone = 'xception'
        downsample_factor = 8
    deeplab = DeeplabV3(pr_model_path, num_classes, backbone, downsample_factor)
    # ----------------------------------------------------------------------------------------------------------#
    #   mode specifies the mode of the test:
    #   'txt_predict'indicates the training collective output. The mode to be selected in the competition is this mode. Here, the result of semantic segmentation is added with distinguishable colors by default. If you need to use a mask, you can use 'dir_predict' mode, or use mIOU related programs.
    #   'predict' indicates the prediction of a single picture. If you want to modify the prediction process, such as saving pictures, intercepting objects, etc.
    #   'video' means video detection. You can call the camera or video for detection.
    #   'fps' means testing FPS. The image used is street.jpg in img folder.
    #   'dir_predict' means to traverse the folder for detection and save. Traverse img folder by default and save in img_out folder.
    # ----------------------------------------------------------------------------------------------------------#
    mode = "txt_predict"
    # ----------------------------------------------------------------------------------------------------------#
    #   video_ Path is used to specify the path of the video. When video_path=0, the camera is detected
    #   If you want to detect video, set it as video_path = "xxx.mp4" can be used to read xxx.mp4 in the root directory MP4 file.
    #   video_save_path indicates the path where the video is saved, and video_save_path="" means no saving
    #   If you want to save the video, set it as video_save_path = "yyy.mp4", which means yyy.mp4 is saved in the root directory.
    #   video_fps for saved fps of video
    #   video_path、video_save_path and video_fps is valid only when mode='video'.
    #   When saving the video, you need to Ctrl + C to exit or run to the last frame to complete the complete saving steps.
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    # -------------------------------------------------------------------------#
    #   test_interval is used to specify the number of image detection when FPS is measured
    #   Theoretically, the larger the test_interval, the more accurate the FPS is.
    # -------------------------------------------------------------------------#
    test_interval = 100
    # -------------------------------------------------------------------------#
    #   dir_origin_path specifies the folder path of the pictures to be detected
    #   dir_save_path specifies the path to save the detected image
    #   dir_origin_path and dir_save_path are valid only when mode='dir_predict.
    # -------------------------------------------------------------------------#
    dir_origin_path = "SHMdata/JPEGImages"
    dir_save_path = "img_out/"
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
3、Run get_miou.py, then get mIOU and mPA.   

### Reference
https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bonlime/keras-deeplab-v3-plus   
https://github.com/bubbliiiing/deeplabv3-plus-pytorch  
(My github)https://github.com/hitwangjm/IC-SHM-P1-hitsz
