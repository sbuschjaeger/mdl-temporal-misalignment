from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x).squeeze(-1)
        return self.fc(self.dropout(x))


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, layers=(2, 2, 2, 2)):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(ResNetBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.avgpool(x).squeeze(-1))


class BranchCNN(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, emb_dim, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(emb_dim)
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.global_pool(x).squeeze(-1)


class LateFusionCNN(nn.Module):
    def __init__(self, sensor_groups: dict[int, list[int]], num_classes: int, emb_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.sensor_groups = sensor_groups
        self.group_keys = sorted(sensor_groups)
        self.branches = nn.ModuleList([
            BranchCNN(len(sensor_groups[key]), emb_dim)
            for key in self.group_keys
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        emb = None
        for branch, key in zip(self.branches, self.group_keys):
            out = branch(x[:, self.sensor_groups[key], :])
            emb = out if emb is None else emb + out
        return self.fc(self.dropout(emb))


class MagnitudeWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, log_scale: bool = True):
        super().__init__()
        self.backbone = backbone
        self.log_scale = log_scale

    def forward(self, x):
        mag = torch.abs(torch.fft.rfft(x, dim=-1))
        if self.log_scale:
            mag = torch.log1p(mag)
        return self.backbone(mag)

class SyncNet(nn.Module):
    class Branch(nn.Module):
        def __init__(self, in_channels: int, window_size: int, num_filters: int = 64, filter_size: int = 5):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, num_filters, filter_size)
            self.bn1 = nn.BatchNorm1d(num_filters)
            self.conv2 = nn.Conv1d(num_filters, num_filters, filter_size)
            self.bn2 = nn.BatchNorm1d(num_filters)
            self.pool2 = nn.MaxPool1d(2)
            self.conv3 = nn.Conv1d(num_filters, num_filters * 2, filter_size)
            self.bn3 = nn.BatchNorm1d(num_filters * 2)
            self.conv4 = nn.Conv1d(num_filters * 2, num_filters * 2, filter_size)
            self.bn4 = nn.BatchNorm1d(num_filters * 2)
            self.pool4 = nn.MaxPool1d(2)

            with torch.no_grad():
                dim = self._forward_convs(torch.zeros(1, in_channels, window_size)).shape[1]
            self.fc5 = nn.Linear(dim, 128)
            self.fc6 = nn.Linear(128, 128)
            self.fc7 = nn.Linear(128, 128)

        def _forward_convs(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool4(F.relu(self.bn4(self.conv4(x))))
            return x.flatten(1)

        def forward(self, x):
            x = F.relu(self.fc5(self._forward_convs(x)))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.fc6(x))
            x = F.dropout(x, p=0.5, training=self.training)
            return self.fc7(x)

    def __init__(
        self,
        window_size: int,
        misalign_indices: list[int],
        aligned_indices: list[int],
        num_filters: int = 64,
        filter_size: int = 5,
    ):
        super().__init__()
        self.register_buffer("misalign_index_tensor", torch.tensor(misalign_indices, dtype=torch.long))
        self.register_buffer("aligned_index_tensor", torch.tensor(aligned_indices, dtype=torch.long))
        self.misalign_indices = list(misalign_indices)
        self.aligned_indices = list(aligned_indices)
        self.misalign_cnn = self.Branch(len(misalign_indices), window_size, num_filters, filter_size)
        self.aligned_cnn = self.Branch(len(aligned_indices), window_size, num_filters, filter_size)

    def forward(self, x):
        mis = x.index_select(1, self.misalign_index_tensor)
        aligned = x.index_select(1, self.aligned_index_tensor)
        return (
            F.normalize(self.misalign_cnn(mis), p=2, dim=1),
            F.normalize(self.aligned_cnn(aligned), p=2, dim=1),
        )


def make_model(model_name: str, *, in_channels: int, num_classes: int, sensor_groups=None, window_size=None, misalign_indices=None, aligned_indices=None):
    model_name = model_name.lower()
    if model_name == "cnn":
        return CNNBaseline(in_channels=in_channels, num_classes=num_classes)
    if model_name == "resnet":
        return ResNet1D(in_channels=in_channels, num_classes=num_classes)
    if model_name == "late_fusion":
        return LateFusionCNN(sensor_groups=sensor_groups, num_classes=num_classes)
    if model_name == "magnitude_cnn":
        return MagnitudeWrapper(CNNBaseline(in_channels=in_channels, num_classes=num_classes))
    if model_name == "syncnet":
        return SyncNet(
            window_size=window_size,
            misalign_indices=misalign_indices,
            aligned_indices=aligned_indices,
        )
    raise ValueError(f"unknown model: {model_name}")
