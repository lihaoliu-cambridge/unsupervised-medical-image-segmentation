import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models
import torch.nn.functional as F
import numpy as np
import math


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal'):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('probabilistic_model') != -1:
            pass
        else:
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data)
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            elif classname.find('BatchNorm3d') != -1:
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    return init_func


def init_net_Seg_data_parallel(net, init_type='kaiming', gpu_ids=None):
    if gpu_ids is None:
        gpu_ids = []

    # custom_init_weights(net, init_type=init_type)
    net.apply(init_weights(init_type))

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    else:
        raise ValueError("No GPU.")

    return net


def define_registration_model(gpu_ids=None, is_training=True, model_parallel=False, mode='lpba40', img_size=(144, 192, 144)):
    if model_parallel:
        print("model_parallel.")
        raise NotImplementedError
    else:
        print("dataloaders parallel.")
        if gpu_ids is None:
            gpu_ids = []

        from .model_architecture.model_contrastive_learning import VoxelMorph3d as model
        if mode == 'lpba40':
            netSeg = model(is_training=is_training, img_size=img_size)
        else:
            raise NotImplementedError

        return init_net_Seg_data_parallel(netSeg, gpu_ids=gpu_ids)


# CVPR 2018 Version 
class smooth_loss(nn.Module):
    def __init__(self):
        super(smooth_loss, self).__init__()

    def __call__(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return 0.01 * (d / 3.0)


class recon_loss(nn.Module):
    def __init__(self,):
        super(recon_loss, self).__init__()

    def __call__(self, x, y):
        return torch.mean( (x - y) ** 2 )
        

# MICCAI 2018 Version - Adpopted from Tensorflow/Keras Implementation
# class kl_loss(nn.Module):
#     def __init__(self):
#         super(kl_loss, self).__init__()
#         # self.image_sigma = 0.02
#         # self.prior_lambda = 10.0
#         self.image_sigma = 0.01
#         self.prior_lambda = 25.0

#     def __call__(self, flow_mean, flow_sigma):
#         # print(1, flow_mean.sum())
#         # print(2, flow_sigma.sum())
#         ndims = len(flow_mean.shape[2:])

#         D = self._degree_matrix(flow_mean.shape[2:]).cuda()
#         # print(3, D.sum())

#         # sigma terms
#         sigma_term = torch.mean(self.prior_lambda * D * torch.exp(flow_sigma.cuda()) - flow_sigma.cuda())
#         # print(4, sigma_term.sum())

#         # precision terms
#         # note needs 0.5 twice, one here (inside self.prec_loss), one below
#         prec_term = self.prior_lambda * self.prec_loss(flow_mean)
#         # print(5, self.prior_lambda, prec_term)

#         # combine terms
#         return 0.5 * ndims * (sigma_term + prec_term)  # ndims because we averaged over dimensions as well

#     def _adj_filt(self, ndims):
#         filt_inner = np.zeros([3] * ndims)
#         for j in range(ndims):
#             o = [[1]] * ndims
#             o[j] = [0, 2]
#             filt_inner[np.ix_(*o)] = 1

#         # full filter, that makes sure the inner filter is applied
#         # ith feature to ith feature
#         filt = np.zeros([ndims, ndims] + [3] * ndims)
#         for i in range(ndims):
#             filt[i, i, ...] = filt_inner

#         return filt

#     def _degree_matrix(self, vol_shape):
#         # get shape stats
#         ndims = len(vol_shape)

#         conv_fn = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
#         conv_fn.weight = torch.nn.Parameter(torch.from_numpy(self._adj_filt(ndims)).float())
#         conv_result = conv_fn(torch.ones([1] + [ndims, *vol_shape]))

#         return conv_result

#     def prec_loss(self, y_pred):
#         vol_shape = y_pred.shape[2:]
#         ndims = len(vol_shape)

#         x = y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]
#         x2 = torch.mean(torch.mul(x, x))

#         y = y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]
#         y2 = torch.mean(torch.mul(y, y))

#         z = y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]
#         z2 = torch.mean(torch.mul(z, z))

#         return 0.5 * torch.add(torch.add(x2, y2), z2) / ndims


class kl_loss(nn.Module):
    def __init__(self):
        super(kl_loss, self).__init__()

    def __call__(self, flow_mean, flow_log_sigma):
        kl_div = -0.5 * torch.sum(1 + flow_log_sigma - flow_mean.pow(2) - flow_log_sigma.exp(), dim=[1, 2, 3, 4]) 
        return 0.01 * (kl_div.mean() / (flow_mean.size(0) * flow_mean.size(1) * flow_mean.size(2) * flow_mean.size(3) * flow_mean.size(4)))


class lamda_mse_loss(nn.Module):
    def __init__(self):
        super(lamda_mse_loss, self).__init__()
        self.image_sigma=1

    def __call__(self, x, y):
        return 1.0 / (self.image_sigma ** 2) * torch.mean( (x - y) ** 2 )


class contrastive_loss(nn.Module):
    def __init__(self, batch_size=8, temperature=0.5, use_cosine_similarity=True):
        super(contrastive_loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.cuda()

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def __call__(self, zis, zjs):
        # print(1, zjs.shape)
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        # print('logits:', logits)

        labels = torch.zeros(2 * self.batch_size).cuda().long()
        loss = self.criterion(logits, labels)

        return 0.01 * loss / (2 * self.batch_size)
