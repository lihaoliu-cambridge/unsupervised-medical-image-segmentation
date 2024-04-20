import os.path
from dataloaders.base_dataset import BaseDataset
from dataloaders.base_dataset import get_transform_contrastive_learning as get_transform_train
from dataloaders.base_dataset import get_transform_contrastive_learning_test as get_transform_test
from PIL import Image
import SimpleITK as sitk
import random
import numpy as np
import copy
import torch
from skimage import exposure


good_labels_FL = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
good_labels_PL = [41, 42, 43, 44, 45]
good_labels_OL = [61, 62, 63, 64, 65, 66, 67, 68]
good_labels_TL = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]
good_labels_CL = [101, 102, 121, 122]
good_labels_Ptm = [163, 164]
good_labels_Hpcp = [165, 166]
good_labels_list = [good_labels_FL, good_labels_PL, good_labels_OL, good_labels_TL, good_labels_CL, good_labels_Ptm, good_labels_Hpcp]


class LPBA40(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.image_path = self.opt.dataroot

        self.is_training = opt.phase == "train"

        # ~~~~~~~~~~~~~~~~~~~ paths ~~~~~~~~~~~~~~~~~~~
        if self.is_training:
            self.constrain = list(range(31, 41, 1))
        else:
            self.constrain = list(range(1, 31, 1))

        self.moving_path = []
        list_dirs = os.walk(self.image_path)
        for root, dirs, files in list_dirs:
            for f in files:
                if (f.endswith(".hdr") or f.endswith(".nii")) and f.startswith("l"):
                    for c in self.constrain:
                        if "l{}_".format(str(c)) in str(f) or "_l{}.".format(str(c)) in str(f):
                            break
                    else:
                        self.moving_path.append(os.path.join(self.image_path, f))

        self.moving_fixed = {}
        for i in self.moving_path:
            fixed_name = i.strip().split("/")[-1].split(".")[-2].split("_")[-1]
            fixed_suffix = i.strip().split("/")[-1].split(".")[-1]
            if not (fixed_suffix == "hdr" or fixed_suffix == "nii"):
                raise Exception("Suffix not hdr or nii.")
            self.moving_fixed[i] = os.path.join(self.image_path, "{}_to_{}.{}".format(str(fixed_name), str(fixed_name), str(fixed_suffix)))
        self.fixed_path = list(set(self.moving_fixed.values()))

        # ~~~~~~~~~~~~~~~~~~~ volume ~~~~~~~~~~~~~~~~~~~
        # Images
        self.fixed_img = {x: self.readVol(x) for x in self.fixed_path}
        self.fixed_img_whiten = {k: self.whitening(v) for k, v in self.fixed_img.items()}

        # Transformation
        if "large" in self.image_path:
            self.transform_train = get_transform_train(img_size=(144, 192, 144))
            self.transform_test = get_transform_test(img_size=(144, 192, 144))
        elif "small" in self.image_path:
            self.transform_train = get_transform_train(img_size=(72, 96, 72))
            self.transform_test = get_transform_test(img_size=(72, 96, 72))

    def name(self):
        return 'LPBA40_contrastive_learning'

    def __len__(self):
        return len(self.moving_path) * 2 if self.is_training else len(self.moving_path)

    def __getitem__(self, index):
        # get train or validation dataloaders
        if self.is_training:
            img_index = int(index % len(self.moving_path))

            moving_img = self.whitening(self.readVol(self.moving_path[img_index]))
            fixed_img = self.fixed_img_whiten[self.moving_fixed[self.moving_path[img_index]]]

            moving_atlas = self.readVol(
                self.moving_path[img_index].replace("LPBA40_rigidly_registered_pairs_histogram_standardization",
                                                    "LPBA40_rigidly_registered_label_pairs").replace('.nii', '.hdr'))
            fixed_atlas = self.readVol(
                self.moving_fixed[self.moving_path[img_index]].replace("LPBA40_rigidly_registered_pairs_histogram_standardization",
                                                                       "LPBA40_rigidly_registered_label_pairs").replace('.nii', '.hdr'))

            moving_img_pytorch, fixed_img_pytorch, moving_atlas_pytorch, fixed_atlas_pytorch = self.transform_train(
                [moving_img, fixed_img, moving_atlas, fixed_atlas])

            return {'A': moving_img_pytorch,
                    'A_atlas': moving_atlas_pytorch,
                    'A_paths': self.moving_path[img_index],
                    'B': fixed_img_pytorch,
                    'B_atlas': fixed_atlas_pytorch,
                    'B_paths': self.moving_fixed[self.moving_path[img_index]]}
        else:
            img_index = int(index % len(self.moving_path))

            moving_img = self.whitening(self.readVol(self.moving_path[img_index]))
            fixed_img = self.fixed_img_whiten[self.moving_fixed[self.moving_path[img_index]]]

            moving_atlas = self.readVol(self.moving_path[img_index].replace("LPBA40_rigidly_registered_pairs_histogram_standardization",
                                                                       "LPBA40_rigidly_registered_label_pairs").replace('.nii', '.hdr'))
            fixed_atlas = self.readVol(self.moving_fixed[self.moving_path[img_index]].replace("LPBA40_rigidly_registered_pairs_histogram_standardization",
                                                                       "LPBA40_rigidly_registered_label_pairs").replace('.nii', '.hdr'))

            # Fuse small regions to a big one.
            for idx, i in enumerate(good_labels_list, start=1):
                for j in i:
                    moving_atlas[moving_atlas == j] = idx
                    fixed_atlas[fixed_atlas == j] = idx
            moving_atlas[moving_atlas > 7] = 0
            fixed_atlas[fixed_atlas > 7] = 0

            moving_img_pytorch, fixed_img_pytorch, moving_atlas_pytorch, fixed_atlas_pytorch = self.transform_test(
                [moving_img, fixed_img, moving_atlas, fixed_atlas])

            return {'A': moving_img_pytorch,
                    'A_atlas': moving_atlas_pytorch,
                    'A_paths': self.moving_path[img_index],
                    'B': fixed_img_pytorch,
                    'B_atlas': fixed_atlas_pytorch,
                    'B_paths': self.moving_fixed[self.moving_path[img_index]]}

    def readVol(self, volpath):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(volpath))).swapaxes(0, 2)

    def whitening(self, image):
        """Not real Whitening. Just standardize image to 0-1."""
        image = image.astype(np.float32)

        return (np.clip(image, 50., 100.) - 50.) / (100 - 50)


def readVol(volpath):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(volpath))).swapaxes(0, 2)
