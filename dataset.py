from torch.utils.data import Dataset
import torch
import numpy as np
from imageio import imwrite, imread
import os
import torch.nn.functional as F
import cv2
def img_normalize(image):
    if len(image.shape)==2:
        channel = (image[:, :, np.newaxis] - 0.485) / 0.229
        image = np.concatenate([channel,channel,channel], axis=2)
    else:
        image = (image-np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3)))\
                /np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return image


class TrainDataset(Dataset):
    def __init__(self, paths):
        self.image = []
        self.label_gt = []
        self.label_edge = []
        self.label_sk = []
        self.count = {}
        for path in paths:
            self.list = os.listdir(os.path.join(path, "Imgs","Train",))
            for i in self.list:
                self.image.append(os.path.join(path,  "Imgs","Train", i))
                self.label_gt.append(os.path.join(path,"GT", "Train",  i.split(".")[0] + ".png"))
                self.label_edge.append(os.path.join(path,  "Edge","Train", i.split(".")[0] + ".png"))
                # self.label_sk.append(os.path.join(path, "skel", i.split(".")[0] + ".png"))
        print("Datasetsize:", len(self.image))
    def __len__(self):
        return len(self.image)
    def __getitem__(self, item):
        img = imread(self.image[item]).astype(np.float32)/255.
        label_gt = imread(self.label_gt[item]).astype(np.float32)/255.
        label_edge = imread(self.label_edge[item]).astype(np.float32) / 255.
        # label_sk = imread(self.label_sk[item]).astype(np.float32) / 255.
        ration = np.random.rand()
        if ration<0.25:
            img = cv2.flip(img, 1)
            label_gt = cv2.flip(label_gt, 1)
            label_edge = cv2.flip(label_edge, 1)
            # label_sk = cv2.flip(label_sk, 1)
        elif ration<0.5:
            img = cv2.flip(img, 0)
            label_gt = cv2.flip(label_gt, 0)
            label_edge = cv2.flip(label_edge, 0)
            # label_sk = cv2.flip(label_sk, 0)
        elif ration<0.75:
            img = cv2.flip(img, -1)
            label_gt = cv2.flip(label_gt, -1)
            label_edge = cv2.flip(label_edge, -1)
            # label_sk = cv2.flip(label_sk, -1)
        if len(label_gt.shape)==3:
            label_gt = label_gt[:,:,0]
            label_edge = label_edge[:, :, 0]
        label_gt=label_gt[:,:,np.newaxis]
        label_edge = label_edge[:, :, np.newaxis]
        # label_sk = label_sk[:, :, np.newaxis]
        return {"img": torch.from_numpy(img_normalize(img)).permute(2,0,1).unsqueeze(0),
                "label_gt":torch.from_numpy(label_gt).permute(2,0,1).unsqueeze(0),
                "label_edge": torch.from_numpy(label_edge).permute(2, 0, 1).unsqueeze(0),
                # "label_sk": torch.from_numpy(label_sk).permute(2, 0, 1).unsqueeze(0),
                }

class TestDataset(Dataset):
    def __init__(self, paths, size):
        self.size=size
        self.image = []
        self.label = []
        for path in paths:
            self.list = os.listdir(os.path.join(path, "Imgs","Val/"))
            self.count={}
            for i in self.list:
                self.image.append(os.path.join(path,"Imgs",  "Val/" , i))
                self.label.append(os.path.join(path,  "GT", "Val/" ,i.split(".")[0]+".png"))
    def __len__(self):
        return len(self.image)
    def __getitem__(self, item):
        img = imread(self.image[item]).astype(np.float32)/255.
        label = imread(self.label[item]).astype(np.uint8)
        if len(label.shape)==2:
            label=label[:,:,np.newaxis]
        return {"img": F.interpolate(torch.from_numpy(img_normalize(img)).permute(2,0,1).unsqueeze(0), (self.size, self.size), mode='bilinear', align_corners=True).squeeze(0),
                "label": torch.from_numpy(label).permute(2,0,1),
                'name': self.label[item]}

def my_collate_fn(batch):
    size = 384
    imgs=[]
    label_gt = []
    label_edge = []
    label_sk = []
    for item in batch:
        imgs.append(F.interpolate(item['img'], (size, size), mode='bilinear'))
        label_gt.append(F.interpolate(item['label_gt'], (size, size), mode='bilinear'))
        label_edge.append(F.interpolate(item['label_edge'], (size, size), mode='bilinear'))
        # label_sk.append(F.interpolate(item['label_sk'], (size, size), mode='bilinear'))
    return {'img': torch.cat(imgs, 0),
            'label_gt': torch.cat(label_gt, 0),
            'label_edge': torch.cat(label_edge, 0),
            # 'label_sk': torch.cat(label_sk, 0),
            }

