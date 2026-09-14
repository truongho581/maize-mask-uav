# Muc dich: dinh nghia kien truc Attention U-Net cho bai toan segmentation.
# Dau vao: tensor anh va cac tham so so lop, dropout, normalization.
# Dau ra: logits mask segmentation co kich thuoc khop voi anh dau vao.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv, AttentionBlock


def match_size(x, ref):
    """Resize tensor x để cùng kích thước không gian (H, W) với tensor ref"""
    return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, norm='bn'):
        super().__init__()
        self.norm = norm  # 👈 lưu kiểu normalization

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64, norm=self.norm)
        self.enc2 = DoubleConv(64, 128, norm=self.norm)
        self.enc3 = DoubleConv(128, 256, norm=self.norm)
        self.enc4 = DoubleConv(256, 512, dropout=True, norm=self.norm)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, dropout=True, norm=self.norm)

        # Decoder + Attention
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256, norm=self.norm)
        self.dec1 = DoubleConv(1024, 512, norm=self.norm)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128, norm=self.norm)
        self.dec2 = DoubleConv(512, 256, norm=self.norm)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64, norm=self.norm)
        self.dec3 = DoubleConv(256, 128, norm=self.norm)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32, norm=self.norm)
        self.dec4 = DoubleConv(128, 64, norm=self.norm)

        # Output
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder + Attention
        d1 = self.up1(b)
        d1 = match_size(d1, e4)
        e4_att = self.att1(d1, e4)
        d1 = torch.cat([e4_att, d1], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = match_size(d2, e3)
        e3_att = self.att2(d2, e3)
        d2 = torch.cat([e3_att, d2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = match_size(d3, e2)
        e2_att = self.att3(d3, e2)
        d3 = torch.cat([e2_att, d3], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = match_size(d4, e1)
        e1_att = self.att4(d4, e1)
        d4 = torch.cat([e1_att, d4], dim=1)
        d4 = self.dec4(d4)

        out = self.out_conv(d4)
        return out
