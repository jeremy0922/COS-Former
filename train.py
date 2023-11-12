import numpy as np
import torch
# import model_only_edge as model
import model_ca3 as  model
import dataset
import os
from torch.utils.data import DataLoader
import argparse
# import torch.distributed as dist
import train_loss
from py_sod_metrics.sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure
import tqdm
import torch.nn.functional as F
import cv2
import logging

# logger = logging.getLogger("logger")
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#     filename="log.txt",
#     filemode="w"  # 每次重启程序，覆盖之前的日志
# )

# console_handler = logging.StreamHandler()
# logger.addHandler(console_handler)
# 创建logger对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # log等级总开关

# log输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

# 控制台handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO) # log等级的开关
stream_handler.setFormatter(formatter)

# 文件handler
file_handler = logging.FileHandler("logging.log")
file_handler.setLevel(logging.INFO) # log等级的开关
file_handler.setFormatter(formatter)

# 添加到logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


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

def test_once(model, args, data_loader):
    model.eval()
    model.is_training = False
    cal_total_seg_metrics = CalTotalMetric()
    count=0
    with torch.no_grad():
        for data in data_loader:
            count+=1
            img, label = data['img'].cuda(), data['label'].cuda()
            name = data['name'][0].split("/")[-1]
            with torch.no_grad():
                out, edge = model(img)
                out=out[0]
            B,C,H,W = label.size()
            # out = out.squeeze(0).cpu().numpy()
            label = label.squeeze().cpu().numpy().astype(np.uint8)
            o = F.interpolate(out, (H,W), mode='bilinear', align_corners=True).detach().cpu().numpy()[0,0]
            pred = (o*255).astype(np.uint8)
            cal_total_seg_metrics.step(pred, label)
        fixed_seg_results = cal_total_seg_metrics.get_results()
    return fixed_seg_results

if __name__ =='__main__':
    parser = argparse.ArgumentParser("Unifying Global-Local Representations Transformer")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--batch_size_per_gpu", default=1)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--base_lr", default=1e-3)
    parser.add_argument("--path",default='data/COD10K', type=str)
    parser.add_argument("--pretrain", type=str)
    args = parser.parse_args()
    # print("local_rank", args.local_rank)
    # word_size = int(os.environ['WORLD_SIZE'])
    # print("word size:", word_size)
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    net = model.Model(args.pretrain, 384)
    # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    # net = torch.nn.parallel.DistributedDataParallel(net.cuda(args.local_rank), device_ids=[args.local_rank])
    net.cuda()
    Dir = [args.path]
    Dataset = dataset.TrainDataset(Dir)
    # Datasampler = torch.utils.data.distributed.DistributedSampler(Dataset, num_replicas=dist.get_world_size(), rank=args.local_rank, shuffle=True)
    Dataloader = DataLoader(Dataset, batch_size=args.batch_size_per_gpu, num_workers=args.batch_size_per_gpu * 2, collate_fn=dataset.my_collate_fn, drop_last=True,shuffle=True)
    
    Test_Dataset = dataset.TestDataset(Dir, 384)
    Test_Dataloader = DataLoader(Test_Dataset, batch_size=1, num_workers=args.batch_size_per_gpu*2)

    encoder_param=[]
    decoer_param=[]
    for name, param in net.named_parameters():
        if "encoder" in name:
            encoder_param.append(param)
        else:
            decoer_param.append(param)
    optimizer = torch.optim.SGD([{"params": encoder_param, "lr":args.base_lr*0.1},{"params":decoer_param, "lr":args.base_lr}], momentum=0.9, weight_decay=1e-5)
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    # for i in range(1, 200):
    for i in range(1, 201):
        if i==100 or i==150:
            for param_group in optimizer.param_groups:
                param_group['lr']= param_group['lr']*0.1
                print("Learning rate:", param_group['lr'])
        # Datasampler.set_epoch(i)
        net.train()
        running_loss, running_loss0=0., 0.
        loss0 = 0.
        count=0
        for data in Dataloader:
            count+=1
            img, label_gt, label_edge = data['img'].cuda(), data['label_gt'].cuda(), data['label_edge'].cuda()
            out, edge_p = net(img)
            # out = net(img)
            # loss = train_loss.multi_bce(out, label_gt, label_edge, label_sk)

            loss = train_loss.multi_edge(out, edge_p, label_gt, label_edge)
            # loss = train_loss.multi_edge(out,label_gt,label_edge)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if i == 1:
                loss0 = loss.item()
            if count%100==0:
                print("Epoch:{}, Iter:{}, loss:{:.5f}".format(i, count, running_loss/count))
        logger.info(f'"epoch":{i}, "loss": {running_loss/count}')
        if not os.path.exists("ckpt"):
            os.mkdir("ckpt")
        # if loss.item() <= loss0 and i>50:
        if (i-1) % 10 == 0:
            seg_results = test_once(net, args, Test_Dataloader)

            logger.info(f'"Smeasure": {seg_results["Smeasure"]}, "wFmeasure": {seg_results["wFmeasure"]} \
                         ,"MAE": {seg_results["MAE"]}, "adpEm": {seg_results["adpEm"]} \
                         ,"meanEm": {seg_results["meanEm"]}, "maxEm": {seg_results["maxEm"]} \
                         ,"adpFm": {seg_results["adpFm"]}, "meanFm": {seg_results["meanFm"]} \
                         ,"maxFm": {seg_results["maxFm"]}')
            torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./ckpt/model_{}_loss_{:.5f}.pth".format(i, running_loss/count))


