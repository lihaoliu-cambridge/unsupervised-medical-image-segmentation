import torch
import torch.nn as nn
from .utils import transform


class SpatialTransformer(nn.Module):
    def __init__(self, interp_method='linear'):
        """
        Parameters:
            interp_method: 'linear' or 'nearest'
            single_transform: whether a single transform supplied for the whole batch
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow
                (along last axis) flipped compared to 'ij' indexing
        """
        self.interp_method = interp_method
        self.ndims = 3
        self.inshape = (160, 192, 224)

    def forward(self, vol, trf):
        """
        Parameters
            inputs: list with two entries
        """
        vol = vol.view(vol.size(0), -1)
        trf = trf.view(trf.size(0), -1)

        # prepare location shift
        return transform(vol, trf, interp_method=self.interp_method)
