import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvt import pvt_v2_b2
from math import log
from thop import profile
from torch import Tensor
from typing import List
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            )


class ChannelAttention(nn.Module):
    def __init__(self, ch, ratio=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // ratio, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x.mean(dim=(2, 3), keepdim=True))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg, max_val], dim=1)
        return x * self.sigmoid(self.conv(concat))

class Net(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(Net, self).__init__()
        self.backbone = pvt_v2_b2()
        path = 'F:\\BBNet--CoCOD-main\\pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.trans1 = nn.Conv2d(64, 64, 1)  # 修改输入通道数为512
        self.trans2 = nn.Conv2d(128, 128, 1)  # 保持原样
        self.trans3 = nn.Conv2d(320, 256, 1)  # 保持原样
        self.trans4 = nn.Conv2d(512, 512, 1)  # 保持原样

        # self.MAM_1 = CoordAtt(512, 512)
        # self.MAM_2 = CoordAtt(256, 256)
        # self.MAM_3 = CoordAtt(128, 128)
        # self.MAM_4 = CoordAtt(64, 64)

        self.PCM1 = MLPBlock(dim=64)
        self.PCM2 = MLPBlock(dim=64)
        self.PCM3 = MLPBlock(dim=64)
        self.PCM4 = MLPBlock(dim=64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.deconv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
        )
        self.deconv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
        )
        self.deconv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
        )
        self.predict_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.dwc3 = conv3x3_bn_relu(128, 64, k=1, s=1, p=0)
        self.dwc2 = conv3x3_bn_relu(256, 128, k=1, s=1, p=0)
        self.dwc1 = conv3x3_bn_relu(512, 256, k=1, s=1, p=0)
        self.dwcon_2 = conv3x3_bn_relu(256, 256)
        self.dwcon_3 = conv3x3_bn_relu(128, 128)
        self.dwcon_4 = conv3x3_bn_relu(64, 64)

        self.xf_11 = nn.Conv2d(512, 64, kernel_size=1)
        self.xf_22 = nn.Conv2d(256, 64, kernel_size=1)
        self.xf_33 = nn.Conv2d(128, 64, kernel_size=1)

        self.eam = EAM()  # PPU

        self.efm1 = EFM(512)  #KPM
        self.efm2 = EFM(256)
        self.efm3 = EFM(128)
        self.efm4 = EFM(64)

        self.clff_d = CLFF()

    def forward(self, x):
        image_shape = x.size()[2:]
        pvt = self.backbone(x)
        x1 = pvt[0]  # [B, 512, H/4, W/4] - 实际输出通道512
        x2 = pvt[1]  # [B, 128, H/8, W/8]
        x3 = pvt[2]  # [B, 320, H/16, W/16]
        x4 = pvt[3]  # [B, 512, H/32, W/32]

        # 修正后的转换层处理
        r1 = self.trans4(x1)  # [B, 512,11,11]
        r2 = self.trans3(x2)  # [B, 256, 22, 22]
        r3 = self.trans2(x3)  # [B, 128, 44, 44]
        r4 = self.trans1(x4)  # [B, 64, 88, 88]

        edge = self.eam(r1, r4)
        edge_att = torch.sigmoid(edge)

        x1a = self.efm1(r1, edge_att)  # [B, 512, H/32, W/32]
        x2a = self.efm2(r2, edge_att)  # [B, 256, H/16, W/16]
        x3a = self.efm3(r3, edge_att)  # [B, 128, H/8, W/8]
        x4a = self.efm4(r4, edge_att)  # [B, 64, H/4, W/4]

        xf_1 = (x1a)  # [B, 512, H/32, W/32]

        # 使用动态尺寸
        target_size_2 = x2a.size()[2:]
        r1_up = F.interpolate(self.dwc1(xf_1), size=target_size_2, mode='bilinear')
        # r1_up, _ = torch.split(r1_up, [128, 128], dim=1)
        # x2a, _ = torch.split(x2a, [128, 128], dim=1)
        # r2_con = torch.cat((x2a, r1_up), 1)
        # r2_con = self.dwcon_2(r2_con)
        xf_2 = (r1_up)  # [B, 256, H/16, W/16]

        target_size_3 = x3a.size()[2:]
        r2_up = F.interpolate(self.dwc2(xf_2), size=target_size_3, mode='bilinear')
        # r2_up, _ = torch.split(r2_up, [64, 64], dim=1)
        # x3a, _ = torch.split(x3a, [64, 64], dim=1)
        # r3_con = torch.cat((x3a, r2_up), 1)
        # r3_con = self.dwcon_3(r3_con)
        xf_3 = (r2_up)  # [B, 128, H/8, W/8]

        target_size_4 = x4a.size()[2:]
        r3_up = F.interpolate(self.dwc3(xf_3), size=target_size_4, mode='bilinear')
        # r3_up, _ = torch.split(r3_up, [32, 32], dim=1)
        # x4a, _ = torch.split(x4a, [32, 32], dim=1)
        # r4_con = torch.cat((x4a, r3_up), 1)
        # r4_con = self.dwcon_4(r4_con)
        xf_4 = (r3_up)  # [B, 64, H/4, W/4]

        xf_11 = self.xf_11(xf_1)  # [B, 64, H/32, W/32]

        xf_12 = F.interpolate(xf_11, size=xf_2.size()[2:], mode='bilinear')  # [B, 64, H/16, W/16]

        xf_22 = self.xf_22(xf_2)  # [B, 64, H/16, W/16]
        xf_21 = F.interpolate(xf_22, size=xf_1.size()[2:], mode='bilinear')  # [B, 64, H/32, W/32]
        xf_23 = F.interpolate(xf_22, size=xf_3.size()[2:], mode='bilinear')  # [B, 64, H/8, W/8]

        xf_33 = self.xf_33(xf_3)  # [B, 64, H/8, W/8]
        xf_32 = F.interpolate(xf_33, size=xf_2.size()[2:], mode='bilinear')  # [B, 64, H/16, W/16]
        xf_34 = F.interpolate(xf_33, size=xf_4.size()[2:], mode='bilinear')  # [B, 64, H/4, W/4]

        xf_44 = xf_4  # [B, 64, H/4, W/4]
        xf_43 = F.interpolate(xf_4, size=xf_3.size()[2:], mode='bilinear')  # [B, 64, H/8, W/8]

        xf_4 = xf_44 + xf_34  # [B, 64, H/4, W/4]
        xf_3 = xf_33 + xf_23 + xf_43  # [B, 64, H/8, W/8]
        xf_2 = xf_22 + xf_12 + xf_32  # [B, 64, H/16, W/16]
        xf_1 = xf_11 + xf_21  # [B, 64, H/32, W/32]

        fd1 = self.clff_d(xf_1, xf_2)
        xf_1 = self.PCM1(fd1)

        xc_1_2 = torch.cat((xf_1, xf_2), 1)
        df_f_2 = self.deconv_layer_2(xc_1_2)
        df_f_2 = self.PCM2(df_f_2)

        xc_1_3 = torch.cat((df_f_2, xf_3), 1)
        df_f_3 = self.deconv_layer_3(xc_1_3)
        df_f_3 = self.PCM3(df_f_3)

        xc_1_4 = torch.cat((df_f_3, xf_4), 1)
        df_f_4 = self.deconv_layer_4(xc_1_4)
        df_f_4 = self.PCM4(df_f_4)

        y1 = self.predict_layer_1(df_f_4)
        y2 = F.interpolate(self.predtrans2(df_f_3), size=image_shape, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_f_2), size=image_shape, mode='bilinear')
        y4 = F.interpolate(self.predtrans4(xf_1), size=image_shape, mode='bilinear')

        return y1, y2, y3, y4
