import torch
import torch.nn as nn

def conv3d(in_channels, out_channels, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )

class X3D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, temporal_stride=1):
        super(X3D_Block, self).__init__()
        self.conv1 = conv3d(in_channels, mid_channels, (1, 3, 3), (temporal_stride, 1, 1), (0, 1, 1))
        self.conv2 = conv3d(mid_channels, mid_channels, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        self.conv3 = conv3d(mid_channels, out_channels, (1, 3, 3), (1, 1, 1), (0, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class X3D(nn.Module):
    def __init__(self, input_channels, num_classes, expansion=1.0):
        super(X3D, self).__init__()
        base_channels = [45, 64, 100, 180, 256]
        base_channels = [int(expansion * ch) for ch in base_channels]

        self.stem = nn.Sequential(
            conv3d(input_channels, base_channels[0], (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            conv3d(base_channels[0], base_channels[0], (3, 3, 3), (1, 1, 1), (1, 1, 1)),
        )

        self.layer1 = nn.Sequential(
            X3D_Block(base_channels[0], base_channels[1], base_channels[0], 1),
            X3D_Block(base_channels[1], base_channels[1], base_channels[0], 1),
        )

        self.layer2 = nn.Sequential(
            X3D_Block(base_channels[1], base_channels[2], base_channels[1], 2),
            X3D_Block(base_channels[2], base_channels[2], base_channels[1], 1),
        )

        self.layer3 = nn.Sequential(
            X3D_Block(base_channels[2], base_channels[3], base_channels[2], 2),
            X3D_Block(base_channels[3], base_channels[3], base_channels[2], 1),
        )

        self.layer4 = nn.Sequential(
            X3D_Block(base_channels[3], base_channels[4], base_channels[3], 2),
            X3D_Block(base_channels[4], base_channels[4], base_channels[3], 1),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(base_channels[4], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def x3d_small(input_channels, num_classes):
    return X3D(input_channels, num_classes, expansion=0.5)

def x3d_medium(input_channels, num_classes):
    return X3D(input_channels, num_classes, expansion=1.0)

def x3d_large(input_channels, num_classes):
    return X3D(input_channels, num_classes, expansion=2.0)

if __name__ == '__main__':
    num_classes = 400  # Change this according to your dataset
    model = x3d_small(3, num_classes)
    print(model)

    # Test input
    input_tensor = torch.randn(1, 3, 16, 112, 112)
    output_tensor = model(input_tensor)
    print("Output shape: ", output_tensor.shape)
