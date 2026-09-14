# Muc dich: chua cac block co ban dung de lap Attention U-Net.
# Dau vao: feature map tu encoder/decoder cua mang.
# Dau ra: feature map da qua convolution hoac attention gate.
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Khối cơ bản 2x(Conv + Norm + ReLU)"""
    def __init__(self, in_channels, out_channels, dropout=False, norm='bn'):
        super().__init__()
        def norm_layer(channels):
            return nn.GroupNorm(8, channels) if norm == 'gn' else nn.BatchNorm2d(channels)

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.3))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionBlock(nn.Module):
    """Khối Attention Gate cho skip connection"""
    def __init__(self, F_g, F_l, F_int, norm='bn'):
        super().__init__()

        def norm_layer(channels):
          # Nếu dùng GroupNorm và số kênh < num_groups => fallback sang LayerNorm
          if norm == 'gn':
              num_groups = min(8, channels)
              if channels % num_groups != 0:
                  # nếu không chia hết thì giảm num_groups cho hợp lệ
                  for g in reversed(range(1, num_groups + 1)):
                      if channels % g == 0:
                          num_groups = g
                          break
              return nn.GroupNorm(num_groups, channels)
          else:
              return nn.BatchNorm2d(channels)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            norm_layer(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            norm_layer(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            norm_layer(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
