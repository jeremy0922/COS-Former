import torch
import torch.nn as nn
import torch.nn.functional as F
import vit


class decoder_stage(nn.Module):
    def __init__(self, infilter, outfilter):
        super(decoder_stage, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(infilter, outfilter, 1, bias=False),
                                   nn.BatchNorm2d(outfilter),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(outfilter, outfilter, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(outfilter),
                                   nn.ReLU(inplace=True),

                                   nn.Conv2d(outfilter, outfilter, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(outfilter),
                                   nn.ReLU(inplace=True))
    def forward(self, x):
        b,c,h,w = x.size()
        x= self.conv1(x)#F.interpolate(self.conv1(x), (h*4, w*4), mode='bilinear', align_corners=True)
        return x

class bridges4(nn.Module):
    def __init__(self, infilter, num):
        super(bridges4, self).__init__()
        outfilter=infilter*4
        self.modulelist = nn.ModuleList()
        self.num=num
        for i in range(num):
            self.modulelist.append(nn.Sequential(
                nn.Conv2d(infilter, outfilter, 1, bias=False),
                nn.BatchNorm2d(outfilter),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2)))
    def forward(self, features):
        assert self.num == len(features)
        feature=[]
        for i in range(self.num):
            feature.append(self.modulelist[i](features[i]))
        return feature

class bridges3(nn.Module):
    def __init__(self, infilter, num):
        super(bridges3, self).__init__()
        outfilter=infilter*2
        self.modulelist = nn.ModuleList()
        self.num=num
        for i in range(num):
            self.modulelist.append(nn.Sequential(
                nn.Conv2d(infilter, outfilter, 1, bias=False),
                nn.BatchNorm2d(outfilter),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.UpsamplingBilinear2d(scale_factor=2)))
    def forward(self, features):
        assert self.num == len(features)
        feature=[]
        for i in range(self.num):
            feature.append(self.modulelist[i](features[i]))
        return feature

class bridges2(nn.Module):
    def __init__(self, infilter, num):
        super(bridges2, self).__init__()
        outfilter = infilter*4
        self.modulelist = nn.ModuleList()
        self.num=num
        for i in range(num):
            self.modulelist.append(nn.Sequential(
                nn.Conv2d(infilter, outfilter, 1, bias=False),
                nn.BatchNorm2d(outfilter), nn.ReLU(inplace=True),
                nn.PixelShuffle(4),
                nn.UpsamplingBilinear2d(scale_factor=2)))
    def forward(self, features):
        assert self.num == len(features)
        feature=[]
        for i in range(self.num):
            feature.append(self.modulelist[i](features[i]))
        return feature

class bridges1(nn.Module):
    def __init__(self, infilter, num):
        super(bridges1, self).__init__()
        outfilter = infilter*2
        self.modulelist = nn.ModuleList()
        self.num=num
        for i in range(num):
            self.modulelist.append(nn.Sequential(
                nn.Conv2d(infilter, outfilter, 1, bias=False), nn.BatchNorm2d(outfilter), nn.ReLU(inplace=True), nn.PixelShuffle(4)))
    def forward(self, features):
        assert self.num == len(features)
        feature=[]
        for i in range(self.num):
            feature.append(F.interpolate(self.modulelist[i](features[i]), (192, 192), mode='bilinear'))
        return feature
class bridges5(nn.Module):
    def __init__(self, infilter, num):
        super(bridges5, self).__init__()
        outfilter=infilter*2
        # outfilter=1024
        self.modulelist = nn.ModuleList()
        self.num=num
        for i in range(num):
            self.modulelist.append(nn.Sequential(
                nn.Conv2d(infilter, outfilter, 1, bias=False),
                nn.BatchNorm2d(outfilter),
                nn.ReLU(inplace=True),
                ))
    def forward(self, features):
        assert self.num == len(features)
        feature=[]
        for i in range(self.num):
            feature.append(self.modulelist[i](features[i]))
        return feature

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)# Squeeze操作的定义
        self.fc = nn.Sequential(# Excitation操作的定义
            nn.Linear(channel, channel // reduction, bias=False),# 压缩
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),# 恢复
            nn.Sigmoid()# 定义归一化操作
        )

    def forward(self, x):
        b, c, _, _ = x.size()# 得到H和W的维度，在这两个维度上进行全局池化
        y = self.avg_pool(x).view(b, c)# Squeeze操作的实现
        y = self.fc(y).view(b, c, 1, 1)# Excitation操作的实现
        # 将y扩展到x相同大小的维度后进行赋权
        return y.expand_as(x) * x

class outs(nn.Module):
    def __init__(self, infilter, num, first, scale_factor=4):
        super(outs, self).__init__()
        self.modulelist = nn.ModuleList()
        self.num = num
        if first:
            self.modulelist.append(nn.Sequential(nn.Conv2d(infilter//2, infilter//2, 1, bias=False),
                                                 nn.BatchNorm2d(infilter//2),
                                                 nn.ReLU(inplace=True),
                                                 nn.UpsamplingBilinear2d(scale_factor=scale_factor),
                                                 nn.Conv2d(infilter//2, 1, 1),
                                                 nn.Sigmoid()))
        else:
            self.modulelist.append(nn.Sequential(nn.Conv2d(infilter, infilter, 1, bias=False),
                                                 nn.BatchNorm2d(infilter),
                                                 nn.ReLU(inplace=True),
                                                 nn.UpsamplingBilinear2d(scale_factor=scale_factor),
                                                 nn.Conv2d(infilter, 1, 1),
                                                 nn.Sigmoid()))
        for i in range(num-1):
            self.modulelist.append(nn.Sequential(nn.Conv2d(infilter, infilter, 1, bias=False),
                                                 nn.BatchNorm2d(infilter),
                                                 nn.ReLU(inplace=True),
                                                 nn.UpsamplingBilinear2d(scale_factor=scale_factor),
                                                 nn.Conv2d(infilter, 1, 1),
                                                 nn.Sigmoid()))

    def forward(self, features):
        assert self.num == len(features)
        # feature = []
        # for i in range(self.num):
        #     feature.append(self.modulelist[i](features[i]))
        return self.modulelist[0](features[0])

class Model(nn.Module):
    def __init__(self, ckpt, imgsize=384):
        super(Model, self).__init__()
        self.encoder = vit.deit_base_distilled_patch16_384()
        self.imgsize=imgsize
        if ckpt is not None:
            ckpt = torch.load(ckpt, map_location='cpu')
            msg = self.encoder.load_state_dict(ckpt["model"], strict=False)
            print(msg)

        self.out_edge3 = outs(768, 1, first=False, scale_factor=8)
        self.out3 = outs(768, 1, first=False, scale_factor=8)
        self.out_edge2 = outs(384, 1, first=False, scale_factor=4)
        self.out2 = outs(384, 1, first=False, scale_factor=4)
        self.out_edge1 = outs(192, 1, first=False, scale_factor=2)
        self.out1 = outs(192, 1, first=False, scale_factor=2)

        self.bridge4 = bridges5(768, 3)
        self.bridge3 = bridges4(768, 3)
        self.bridge2 = bridges3(768, 3)
        self.bridge1 = bridges2(768, 3)
        self.decoder12 = decoder_stage(1536, 768)
        self.decoder11 = decoder_stage(1536, 384)
        self.decoder10 = decoder_stage(1536, 192)
        self.decoder9 = decoder_stage(768, 768) 
        self.decoder8 = decoder_stage(768, 384)
        self.decoder7 = decoder_stage(768, 192)
        self.decoder6 = decoder_stage(384, 384 )
        self.decoder5 = decoder_stage(384, 384)
        self.decoder4 = decoder_stage(384, 192)
        self.decoder3 = decoder_stage(192, 192)
        self.decoder2 = decoder_stage(192, 192)
        self.decoder1 = decoder_stage(192, 192)

        self.decoder_p3 = decoder_stage(768 + 768, 768)
        self.decoder_p2 = decoder_stage(768 + 384 + 384, 384)
        self.decoder_p1 = decoder_stage(384 + 192 + 192, 192)

        self.decoder_edge3 = decoder_stage(768 + 768, 768)
        self.decoder_edge2 = decoder_stage(768 + 384 + 384, 384)
        self.decoder_edge1 = decoder_stage(384 + 192 + 192, 192)

        self.se2 = SELayer(768 + 768)
        self.se1 = SELayer(384 + 384)

        self.decoder_se2 = decoder_stage(768 + 768, 384)
        self.decoder_se1 = decoder_stage(384 + 384, 192)

    def decoder(self, feature):

        feature12 = self.decoder12(F.interpolate(feature[-1], (48, 48), mode='bilinear'))
        feature11 = self.decoder11(F.interpolate(feature[-2], (96, 96), mode='bilinear'))
        feature10 = self.decoder10(F.interpolate(feature[-3], (192, 192), mode='bilinear'))

        feature9 = self.decoder9(feature12 * feature[-4])
        feature8 = self.decoder8(F.interpolate(feature[-5], (96, 96), mode='bilinear'))
        feature7 = self.decoder7(F.interpolate(feature[-6], (192, 192), mode='bilinear'))

        feature6 = self.decoder6(feature11 * feature[-7])
        feature5 = self.decoder5(feature6 * feature[-8] * feature8)
        feature4 = self.decoder4(F.interpolate(feature[-9], (192, 192), mode='bilinear'))

        feature3 = self.decoder3(feature10 * feature[-10])
        feature2 = self.decoder2(feature7 * feature[-11] * feature3)
        feature1 = self.decoder1(feature2 * feature[-12] * feature4)

        p3 = self.decoder_p3(torch.cat((F.interpolate(feature12, (48, 48), mode='bilinear'), feature9), 1))
        edge3 = self.decoder_edge3(torch.cat((F.interpolate(feature12, (48, 48), mode='bilinear'), feature9), 1))

        SE2 = self.decoder_se2(self.se2(torch.cat((p3, edge3), 1)))
        SE2 = F.interpolate(SE2, (96, 96), mode='bilinear')     

        p2 = self.decoder_p2(torch.cat((F.interpolate(p3, (96, 96), mode='bilinear'), feature5, SE2), 1)) 
        edge2 = self.decoder_edge2(torch.cat((F.interpolate(edge3, (96, 96), mode='bilinear'),feature5, SE2), 1))
        
        SE1 = self.decoder_se1(self.se1(torch.cat((p2, edge2), 1)))
        SE1 = F.interpolate(SE1, (192, 192), mode='bilinear')

        p1 = self.decoder_p1(torch.cat((F.interpolate(p2, (192, 192), mode='bilinear'), feature1, SE1), 1))
        edge1 = self.decoder_edge1(torch.cat((F.interpolate(edge2, (192, 192), mode='bilinear'), feature1, SE1), 1))

        return [p1, p2, p3], [edge1, edge2, edge3]

    def forward(self, img):
        B, C, H, W = img.size()
        x = self.encoder(img)
        feature = []  #3, 6, 9, 12
        for x0 in x:
            feature.append(x0[:, 2:, :].permute(0, 2, 1).view(B, 768, int(self.imgsize/16), int(self.imgsize/16)).contiguous())
        feature = self.bridge1(feature[:3])+self.bridge2(feature[3:6])+self.bridge3(feature[6:9])+self.bridge4(feature[9:])
        feature, edge = self.decoder(feature)
        return [self.out1([feature[0]]), self.out2([feature[1]]), self.out3([feature[2]])], \
            [self.out_edge1([edge[0]]), self.out_edge2([edge[1]]), self.out_edge3([feature[2]])]
    

if __name__ =='__main__':
    img = torch.rand((1,3,384,384))
    print(img.shape)
    net = Model(None)
    a,b = net(img)
    print("1")