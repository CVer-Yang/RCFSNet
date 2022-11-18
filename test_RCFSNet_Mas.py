import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

# import sklearn.metrics as metrics
import cv2
import os
import numpy as np

from time import time
from PIL import Image

import warnings

warnings.filterwarnings('ignore')

from networks.RCFSNet import RCFSNet
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

BATCHSIZE_PER_CARD = 32

#
# def calculate_auc_test(prediction, label):
#     # read images
#     # convert 2D array into 1D array
#     result_1D = prediction.flatten()
#     label_1D = label.flatten()
#
#
#     label_1D = label_1D / 255
#
#     auc = metrics.roc_auc_score(label_1D, result_1D)
#
#     # print("AUC={0:.4f}".format(auc))
#
#     return auc

def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    sen = TP / (TP + FN)
    iou = TP / (TP + FN + FP)
    pre = TP / (TP + FP + 1e-6)
    f1 = (2 * pre * sen) / (pre + sen + 1e-6)
    return acc, sen, iou, pre, f1, TP, FN, TN, FP

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        print(batchsize)
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        # img90 = np.array(np.rot90(img))
        # img1 = np.concatenate([img[None], img90[None]])
        # img2 = np.array(img1)[:, ::-1]
        # img3 = np.array(img1)[:, :, ::-1]
        # img4 = np.array(img2)[:, :, ::-1]
        img1 = img
        img2 = np.array(img1)[::-1]
        img3 = np.array(img1)[:, ::-1]
        img4 = np.array(img2)[:, ::-1]
        img1 = img1.transpose(2, 0, 1)
        img2 = img2.transpose(2, 0, 1)
        img3 = img3.transpose(2, 0, 1)
        img4 = img4.transpose(2, 0, 1)
        img1 = img1.reshape(-1, 3, 1024, 1024)
        img2 = img2.reshape(-1, 3, 1024, 1024)
        img3 = img3.reshape(-1, 3, 1024, 1024)
        img4 = img4.reshape(-1, 3, 1024, 1024)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask2 = maska + maskb[::-1] + maskc[:, ::-1] + maskd[::-1, ::-1]

        # img1 = img1.transpose(0, 3, 1, 2)
        # img2 = img2.transpose(0, 3, 1, 2)
        # img3 = img3.transpose(0, 3, 1, 2)
        # img4 = img4.transpose(0, 3, 1, 2)
        #
        # img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        # img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        # img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        # img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        # maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        # maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        # maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        # maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        #
        # mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        # mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        # 修改图片尺寸
        img = cv2.resize(img, (1024, 1024))

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model)


def test_ce_net_vessel():
    #source = '/data/guoxuejun/yangjialin/CE-Net/dataset/ROAD/test/images/'
    source = './dataset/Mas/test/images/'
    val = os.listdir(source)
    disc = 20
    # 使用多种模型进行测试
    solver = TTAFrame(RCFSNet)
    solver.load('./weights/lunwen3_12plus4Mas.th')
    # target = '/data/guoxuejun/yangjialin/CE-Net/submits/log_U_Net/'
    target = './submits/RCFSNet_Mas20/'

    tic = time()
    if not os.path.exists(target):
        os.mkdir(target)
    # gt_root = '/data/guoxuejun/yangjialin/CE-Net/dataset/ROAD/test/masks'
    gt_root = './dataset/Mas/test/masks'
    total_m1 = 0

    hausdorff = 0
    total_acc = []
    total_sen = []
    total_iou = []
    total_pre = []
    total_f1 = []
    tp = []
    fn= []
    tn= []
    fp= []

    threshold = 2
    total_auc = []

    for i, name in enumerate(val):
        # if i%10 == 0:
        #     print(i/10, '    ','%.2f'%(time()-tic))
        image_path = os.path.join(source, name)
        print(image_path)
        mask = solver.test_one_img_from_path(image_path)

        new_mask = mask.copy()

        mask[mask > threshold] = 255
        mask[mask <= threshold] = 0
        mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)

        ground_truth_path = os.path.join(gt_root, name.split('.')[0] + '.tif')

        # print(ground_truth_path)
        ground_truth = np.array(cv2.imread(ground_truth_path))[:, :, 1]

        mask = cv2.resize(mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))

        new_mask = cv2.resize(new_mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))
        # total_auc.append(calculate_auc_test(new_mask / 8., ground_truth))

        predi_mask = np.zeros(shape=np.shape(mask))
        predi_mask[mask > disc] = 1
        gt = np.zeros(shape=np.shape(ground_truth))
        gt[ground_truth > 0] = 1
        # print(gt.shape)

        acc, sen, iou, pre, f1, TP, FN, TN, FP= accuracy(predi_mask[:, :, 0], gt)
        total_acc.append(acc)
        total_sen.append(sen)
        total_iou.append(iou)
        total_pre.append(pre)
        total_f1.append(f1)
        tp.append(TP)
        fn.append(FN)
        tn.append(TN)
        fp.append(FP)

        # print(i + 1, acc, sen, calculate_auc_test(new_mask / 8., ground_truth))
        print(i + 1, acc, sen, iou, pre, f1)

        cv2.imwrite(target + name.split('.')[0] + 'mask.png', mask.astype(np.uint8))
    print(np.mean(total_acc), np.std(total_acc))
    print(np.mean(total_sen), np.std(total_sen))
    print(np.mean(total_iou), np.std(total_iou))
    print(np.mean(total_pre), np.std(total_pre))
    print(np.mean(total_f1), np.std(total_f1))
    print(np.sum(tp), np.mean(tp))
    print(np.sum(fn), np.mean(fn))
    print(np.sum(tn), np.mean(tn))
    print(np.sum(tp), np.mean(tp))

    # print(np.mean(total_auc), np.std(total_auc))

if __name__ == '__main__':
    test_ce_net_vessel()
