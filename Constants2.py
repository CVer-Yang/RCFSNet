import os
import shutil

# ########修改输入的图像的大小
# Image_size = (448, 448)
# Image_size = (512, 512)
#Image_size = (1024, 1024)
Image_size = (1024, 1024)
# ########模仿他的数据的放置方式，文件夹的命名
# ROOT = './dataset/DRIVE'
# ROOT = './dataset/ROAD'
ROOT = './dataset/Mas'
# #######batch_size大小，根据训练的情况进行
BATCHSIZE_PER_CARD = 2
TOTAL_EPOCH = 300
INITAL_EPOCH_LOSS = 10000
NUM_EARLY_STOP = 6
NUM_UPDATE_LR = 3
BINARY_CLASS = 1
