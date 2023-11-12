import torch
# import model
import model_ca3 as model
import dataset
import os
from torch.utils.data import DataLoader
import train_loss
import numpy as np
import torch.nn.functional as F
from imageio import imwrite
import sys
from py_sod_metrics.sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure
class CalTotalMetric(object):
    __slots__ = ["cal_mae", "cal_fm", "cal_sm", "cal_em", "cal_wfm"]

    def __init__(self):
        self.cal_mae = MAE()
        self.cal_fm = Fmeasure()
        self.cal_sm = Smeasure()
        self.cal_em = Emeasure()
        self.cal_wfm = WeightedFmeasure()

    def step(self, pred: np.ndarray, gt: np.ndarray):
        assert pred.ndim == gt.ndim and pred.shape == gt.shape, (pred.shape, gt.shape)
        assert pred.dtype == np.uint8, pred.dtype
        assert gt.dtype == np.uint8, gt.dtype

        self.cal_mae.step(pred, gt)
        self.cal_fm.step(pred, gt)
        self.cal_sm.step(pred, gt)
        self.cal_em.step(pred, gt)
        self.cal_wfm.step(pred, gt)

    def get_results(self, bit_width: int = 3) -> dict:
        fm = self.cal_fm.get_results()["fm"]
        wfm = self.cal_wfm.get_results()["wfm"]
        sm = self.cal_sm.get_results()["sm"]
        em = self.cal_em.get_results()["em"]
        mae = self.cal_mae.get_results()["mae"]
        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
        }

        def _round_w_zero_padding(_x):
            _x = str(_x.round(bit_width))
            _x += "0" * (bit_width - len(_x.split(".")[-1]))
            return _x

        results = {name: _round_w_zero_padding(metric) for name, metric in results.items()}
        return results
    
def metrix(pred, label,cal_total_seg_metrics):
    B,C,H,W = label.size()
    # out = out.squeeze(0).cpu().numpy()
    label = label.squeeze().cpu().numpy().astype(np.uint8)
    o = F.interpolate(out, (H,W), mode='bilinear', align_corners=True).detach().cpu().numpy()[0,0]
    pred = (o*255).astype(np.uint8)
    cal_total_seg_metrics.step(pred, label)

if __name__ =='__main__':
    batch_size = 1
    index = True
    net = model.Model(None, imgsize=384).cuda()
    ckpt="ckpt-CHAMELEON/model_191_loss_0.05234.pth"
    Dirs=['data/CHAMELEON']
    cal_total_seg_metrics = CalTotalMetric()
    pretrained_dict = torch.load(ckpt)

    pretrained_dict=pretrained_dict["model"]
    # net.load_state_dict(net_dict)
    # net_dict = net.state_dict() 
    # net_dict.update(pretrained_dict)
    net.load_state_dict(pretrained_dict)
    net.eval()
    for i in range(len(Dirs)):
        Dir = Dirs[i]
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(os.path.join("results", Dir.split("/")[-1])):
            os.mkdir(os.path.join("results", Dir.split("/")[-1]))
        Dataset = dataset.TestDataset([Dir], 384)
        Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=batch_size*2)
        count=0
        for data in Dataloader:
            count+=1
            print(count)
            img, label = data['img'].cuda(), data['label'].cuda()
            name = data['name'][0].split("/")[-1]
            with torch.no_grad():
#                     out = net(img)[0]
#                     out = net(img)
                out, edge = net(img)
#                     out = out[0]
                out=out[0]
#                 print(out)

            B,C,H,W = label.size()
            o = F.interpolate(out, (H,W), mode='bilinear', align_corners=True).detach().cpu().numpy()[0,0]
            pred = (o*255).astype(np.uint8)
            if index:
                label = label.squeeze().cpu().numpy().astype(np.uint8)
                cal_total_seg_metrics.step(pred, label)
            imwrite("./results/"+Dir.split("/")[-1]+"/"+name, pred)
    seg_results = cal_total_seg_metrics.get_results()
    print(f'"Smeasure": {seg_results["Smeasure"]}, "wFmeasure": {seg_results["wFmeasure"]} \
            ,"MAE": {seg_results["MAE"]}, "adpEm": {seg_results["adpEm"]} \
            ,"meanEm": {seg_results["meanEm"]}, "maxEm": {seg_results["maxEm"]} \
            ,"adpFm": {seg_results["adpFm"]}, "meanFm": {seg_results["meanFm"]} \
            ,"maxFm": {seg_results["maxFm"]}')
            # o = F.interpolate(out, (H,W), mode='bilinear', align_corners=True).detach().cpu().numpy()[0,0]
            # o = (o*255).astype(np.uint8)
            # imwrite("./results/"+Dir.split("/")[-1]+"/"+name, o)

