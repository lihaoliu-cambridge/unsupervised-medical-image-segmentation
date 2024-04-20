import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
from torchvision.models import vgg16, resnet50
from torch.distributions import normal
from torch.nn import init
import SimpleITK as sitk
from torch.distributions.normal import Normal


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel // 2, out_channels=16)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(16, 32)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(32, 64)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)

        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear_x = torch.nn.Linear(64, 64)
        self.linear_y = torch.nn.Linear(64, 64)

        # Bottleneck
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
            torch.nn.BatchNorm3d(mid_channel * 2),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel * 2, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )

        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, 3)

        self._init_weight()

    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            # For Probablisitic Model - Softmax is Better
            torch.nn.Softmax() 
            # # For Non-Probablisitic Model - Relu is Better
            # torch.nn.ReLU()
        )
        return block

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose3d):
                init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.uniform_(0.0, 1.0)
                m.bias.data.fill_(0)

    def forward(self, x, y):
        # Encode 1
        encode_block1_x = self.conv_encode1(x)
        encode_pool1_x = self.conv_maxpool1(encode_block1_x)
        encode_block2_x = self.conv_encode2(encode_pool1_x)
        encode_pool2_x = self.conv_maxpool2(encode_block2_x)
        encode_block3_x = self.conv_encode3(encode_pool2_x)
        encode_pool3_x = self.conv_maxpool3(encode_block3_x)
        f_x = self.avgpool(encode_pool3_x)
        f_x = f_x.squeeze()
        f_x = self.linear_x(f_x)
        f_x = f_x / f_x.norm(dim=-1, keepdim=True)

        # Encode 2
        encode_block1_y = self.conv_encode1(y)
        encode_pool1_y = self.conv_maxpool1(encode_block1_y)
        encode_block2_y = self.conv_encode2(encode_pool1_y)
        encode_pool2_y = self.conv_maxpool2(encode_block2_y)
        encode_block3_y = self.conv_encode3(encode_pool2_y)
        encode_pool3_y = self.conv_maxpool3(encode_block3_y)
        f_y = self.avgpool(encode_pool3_y)
        f_y = f_y.squeeze()
        f_y = self.linear_y(f_y)
        f_y = f_y / f_y.norm(dim=-1, keepdim=True)

        # Bottleneck
        bottleneck1 = self.bottleneck(torch.cat((encode_pool3_x, encode_pool3_y), 1))

        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, torch.cat((encode_block3_x, encode_block3_y), 1))
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, torch.cat([encode_block2_x, encode_block2_y], 1))
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, torch.cat([encode_block1_x, encode_block1_y], 1))
        final_layer = self.final_layer(decode_block1)

        return final_layer, f_x, f_y


class ProbabilisticModel(nn.Module):
    def __init__(self, is_training=True):
        super(ProbabilisticModel, self).__init__()

        self.mean = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.log_sigma = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, padding=1)

        # Manual Initialization
        self.mean.weight.data.normal_(0, 1e-5)
        self.log_sigma.weight.data.normal_(0, 1e-10)
        self.log_sigma.bias.data.fill_(-10.)

        self.is_training=is_training

    def forward(self, final_layer):
        flow_mean = self.mean(final_layer)
        flow_log_sigma = self.log_sigma(final_layer)
        noise = torch.randn_like(flow_mean).cuda()

        if self.is_training:
            flow = flow_mean + flow_log_sigma * noise 
        else:
            flow = flow_mean + flow_log_sigma # No noise at testing time

        return flow, flow_mean, flow_log_sigma


class VoxelMorph3d(nn.Module):
    def __init__(self, in_channels=2, use_gpu=False, is_training=True, img_size=(144, 192, 144)):
        super(VoxelMorph3d, self).__init__()
        self.unet = UNet(in_channels, 3)
        self.probabilistic_model = ProbabilisticModel(is_training=is_training)
        self.spatial_transform = SpatialTransformer(img_size)

        if use_gpu:
            self.unet = self.unet.cuda()
            self.probabilistic_model = self.probabilistic_model.cuda()
            self.spatial_transform = self.spatial_transform.cuda()

    def forward(self, moving_image, fixed_image, moving_atlas):
        flow, f_x, f_y = self.unet(moving_image, fixed_image)

        deformation_matrix, flow_mean, flow_log_sigma = self.probabilistic_model(flow)
        warped_image = self.spatial_transform(moving_image, deformation_matrix)
        warped_image_atlas = self.spatial_transform(moving_atlas, deformation_matrix, mode="nearest")

        return warped_image, warped_image_atlas, deformation_matrix, flow_mean, flow_log_sigma, f_x, f_y


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, mode='bilinear'):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=mode)
