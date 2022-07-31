import os
import subprocess
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.stats as stats

DEFAULT_CUTOFF = 0.01, 0.99
STANDARD_RANGE = 0, 100


def _get_percentiles(percentiles_cutoff):
    quartiles = np.arange(25, 100, 25).tolist()
    deciles = np.arange(10, 100, 10).tolist()
    all_percentiles = list(percentiles_cutoff) + quartiles + deciles
    percentiles = sorted(set(all_percentiles))
    return np.array(percentiles)


def _get_average_mapping(percentiles_database):
    """Map the landmarks of the database to the chosen range.

    Args:
        percentiles_database: Percentiles database over which to perform the
            averaging.
    """
    # Assuming percentiles_database.shape == (num_data_points, num_percentiles)
    pc1 = percentiles_database[:, 0]
    pc2 = percentiles_database[:, -1]
    s1, s2 = STANDARD_RANGE
    slopes = (s2 - s1) / (pc2 - pc1)
    slopes = np.nan_to_num(slopes)
    intercepts = np.mean(s1 - slopes * pc1)
    num_images = len(percentiles_database)
    final_map = slopes.dot(percentiles_database) / num_images + intercepts
    return final_map


def _standardize_cutoff(cutoff):
    """Standardize the cutoff values given in the configuration.

    Computes percentile landmark normalization by default.

    """
    cutoff = np.asarray(cutoff)
    cutoff[0] = max(0., cutoff[0])
    cutoff[1] = min(1., cutoff[1])
    cutoff[0] = np.min([cutoff[0], 0.09])
    cutoff[1] = np.max([cutoff[1], 0.91])
    return cutoff


def normalize(array, landmarks, mask, cutoff=None, epsilon=1e-5):
    cutoff_ = DEFAULT_CUTOFF if cutoff is None else cutoff
    mapping = landmarks

    data = array
    shape = data.shape
    data = data.reshape(-1).astype(np.float32)

    # if mask is None:
    #     mask = np.ones_like(data, np.bool)
    mask = mask.reshape(-1)

    range_to_use = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]

    quantiles_cutoff = _standardize_cutoff(cutoff_)
    percentiles_cutoff = 100 * np.array(quantiles_cutoff)
    percentiles = _get_percentiles(percentiles_cutoff)
    percentile_values = np.percentile(data[mask], percentiles)

    # Apply linear histogram standardization
    range_mapping = mapping[range_to_use]
    range_perc = percentile_values[range_to_use]
    diff_mapping = np.diff(range_mapping)
    diff_perc = np.diff(range_perc)

    # Handling the case where two landmarks are the same
    # for a given input image. This usually happens when
    # image background is not removed from the image.
    diff_perc[diff_perc < epsilon] = np.inf

    affine_map = np.zeros([2, len(range_to_use) - 1])

    # Compute slopes of the linear models
    affine_map[0] = diff_mapping / diff_perc

    # Compute intercepts of the linear models
    affine_map[1] = range_mapping[:-1] - affine_map[0] * range_perc[:-1]

    bin_id = np.digitize(data, range_perc[1:-1], right=False)
    lin_img = affine_map[0, bin_id]
    aff_img = affine_map[1, bin_id]
    new_img = lin_img * data + aff_img
    new_img = new_img.reshape(shape)
    new_img = new_img.astype(np.float32)

    return new_img


def calculate_landmarks(image_path):
    quantiles_cutoff = DEFAULT_CUTOFF
    percentiles_cutoff = 100 * np.array(quantiles_cutoff)
    percentiles_database = []
    percentiles = _get_percentiles(percentiles_cutoff)

    count = 1
    for (dirpath, dirnames, filenames) in os.walk(image_path):
        print(dirnames)
        dirnames = ['NKI-RS-22_volumes', 'NKI-TRT-20_volumes', 'OASIS-TRT-20_volumes']
        for dir in dirnames:
            print(dir)
            second_dirpath = os.path.join(dirpath, dir)
            for (dirpath2, dirnames2, filenames2) in os.walk(second_dirpath):
                for dir2 in dirnames2:
                    img_path = os.path.join(second_dirpath, dir2, 't1weighted_brain.MNI152.nii.gz')
                    atlas_path = os.path.join(second_dirpath, dir2, 'labels.DKT31.manual.MNI152.nii.gz')
                    atlas_all_path = os.path.join(second_dirpath, dir2, 'labels.DKT31.manual+aseg.MNI152.nii.gz')
                    print(img_path)

                    if os.path.exists(img_path) and os.path.exists(atlas_path):
                        img_sitk = sitk.ReadImage(str(img_path))
                        img_np = sitk.GetArrayFromImage(img_sitk).swapaxes(0, 2)

                        atlas_all_sitk = sitk.ReadImage(str(atlas_all_path))
                        atlas_all_np = sitk.GetArrayFromImage(atlas_all_sitk).swapaxes(0, 2)
                        atlas_all_np[atlas_all_np == 7] = 0
                        atlas_all_np[atlas_all_np == 8] = 0
                        atlas_all_np[atlas_all_np == 15] = 0
                        atlas_all_np[atlas_all_np == 16] = 0
                        atlas_all_np[atlas_all_np == 46] = 0
                        atlas_all_np[atlas_all_np == 47] = 0

                        percentile_values = np.percentile(img_np[atlas_all_np > 0], percentiles)
                        percentiles_database.append(percentile_values)
                        count += 1
                    else:
                        raise Exception

    percentiles_database = np.vstack(percentiles_database)
    mapping = _get_average_mapping(percentiles_database)
    print(mapping)

    np.save('../../dataset/Mindboggle101/mapping.npy', mapping)

    return mapping


def resample_image(itk_image, out_spacing=(2.0, 2.0, 2.0), is_label=False):

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def center_crop(img, size_ratio):
    x, y, z = img.shape
    size_ratio_x, size_ratio_y, size_ratio_z = size_ratio
    size_x = size_ratio_x
    size_y = size_ratio_y
    size_z = size_ratio_z

    if x < size_x or y < size_y and z < size_z:
        raise ValueError

    x1 = int((x - size_x) / 2)
    y1 = int((y - size_y) / 2)
    z1 = int((z - size_z) / 2)

    img_crop = img[x1: x1 + size_x, y1: y1 + size_y, z1: z1 + size_z]

    return img_crop


def histogram_stardardization(mapping,
                              image_path,
                              output_path_hs):

    if not os.path.exists(str(output_path_hs)):
        os.makedirs(str(output_path_hs))

    count = 1

    for (dirpath, dirnames, filenames) in os.walk(image_path):
        print(dirnames)
        dirnames = ['NKI-RS-22_volumes', 'NKI-TRT-20_volumes', 'OASIS-TRT-20_volumes']
        for dir in dirnames:
            print(dir)
            second_dirpath = os.path.join(dirpath, dir)
            for (dirpath2, dirnames2, filenames2) in os.walk(second_dirpath):
                for dir2 in dirnames2:
                    img_path = os.path.join(second_dirpath, dir2, 't1weighted_brain.MNI152.nii.gz')
                    atlas_path = os.path.join(second_dirpath, dir2, 'labels.DKT31.manual.MNI152.nii.gz')
                    atlas_all_path = os.path.join(second_dirpath, dir2, 'labels.DKT31.manual+aseg.MNI152.nii.gz')
                    print(img_path)

                    if os.path.exists(img_path) and os.path.exists(atlas_path):
                        img_sitk = sitk.ReadImage(str(img_path))
                        img_np = sitk.GetArrayFromImage(img_sitk).swapaxes(0, 2)

                        atlas_sitk = sitk.ReadImage(str(atlas_path))
                        atlas_np = sitk.GetArrayFromImage(atlas_sitk).swapaxes(0, 2)

                        atlas_all_sitk = sitk.ReadImage(str(atlas_all_path))
                        atlas_all_np = sitk.GetArrayFromImage(atlas_all_sitk).swapaxes(0, 2)
                        atlas_all_np[atlas_all_np == 7] = 0
                        atlas_all_np[atlas_all_np == 8] = 0
                        atlas_all_np[atlas_all_np == 15] = 0
                        atlas_all_np[atlas_all_np == 16] = 0
                        atlas_all_np[atlas_all_np == 46] = 0
                        atlas_all_np[atlas_all_np == 47] = 0

                        # HS
                        img_hs = normalize(img_np * (atlas_all_np > 0), mapping, atlas_all_np > 0)

                        # Center Crop
                        img_hs_crop = center_crop(img=img_hs, size_ratio=(160, 192, 160)).swapaxes(0, 2)
                        new_img = sitk.GetImageFromArray(img_hs_crop)
                        new_img.SetSpacing(img_sitk.GetSpacing())
                        new_img.SetDirection(img_sitk.GetDirection())
                        new_img.SetOrigin(img_sitk.GetOrigin())

                        atlas_crop = center_crop(img=atlas_np, size_ratio=(160, 192, 160)).swapaxes(0, 2)
                        new_atlas = sitk.GetImageFromArray(atlas_crop)
                        new_atlas.SetSpacing(atlas_sitk.GetSpacing())
                        new_atlas.SetDirection(atlas_sitk.GetDirection())
                        new_atlas.SetOrigin(atlas_sitk.GetOrigin())

                        output_path_hs_img = os.path.join(output_path_hs, 'brain_{}.nii.gz'.format(count))
                        output_path_hs_atlas = os.path.join(output_path_hs, 'atlas_{}.nii.gz'.format(count))

                        sitk.WriteImage(new_img, str(output_path_hs_img))
                        sitk.WriteImage(new_atlas, str(output_path_hs_atlas))

                        print(count)
                        count+=1


def resample_image_and_label(input_path='../../dataset/Mindboggle101/mindboggle101_hs',
                             output_path_image_hs_re ='../../dataset/Mindboggle101/mindboggle101_hs_re',
                             output_path_mask_hs_re ='../../dataset/Mindboggle101/mindboggle101_hs_re'):

    if not os.path.exists(str(output_path_image_hs_re)):
        os.makedirs(str(output_path_image_hs_re))
    if not os.path.exists(str(output_path_mask_hs_re)):
        os.makedirs(str(output_path_mask_hs_re))

    for i in list(range(1, 63, 1)):
        volpath = os.path.join(input_path, 'brain_{}.nii.gz'.format(i))
        img_sitk = sitk.ReadImage(str(volpath))
        img_sitk_hs_re = resample_image(img_sitk)

        atlas_path = os.path.join(input_path, 'atlas_{}.nii.gz'.format(i))
        atlas_sitk = sitk.ReadImage(str(atlas_path))
        atlas_resampled = resample_image(atlas_sitk, is_label=True)

        output_path_hs_img = os.path.join(output_path_image_hs_re, 'brain_{}.nii.gz'.format(i))
        output_path_hs_atlas = os.path.join(output_path_mask_hs_re, 'atlas_{}.nii.gz'.format(i))

        sitk.WriteImage(img_sitk_hs_re, str(output_path_hs_img))
        sitk.WriteImage(atlas_resampled, str(output_path_hs_atlas))


def plot_hist(image_path_hs):
    for i in list(range(1, 63, 1)):
        volpath = os.path.join(image_path_hs, 'brain_{}.nii.gz'.format(i))
        img_sitk = sitk.ReadImage(str(volpath))
        img_np = sitk.GetArrayFromImage(img_sitk).swapaxes(0, 2)
        data = np.reshape(img_np[img_np>0], -1)

        print(np.min(img_np), np.max(img_np))

        density = stats.gaussian_kde(data)
        xs = np.linspace(0, 150, 300)
        density.covariance_factor = lambda: .25
        density._compute_covariance()

        plt.plot(xs, density(xs))
    plt.show()


if __name__ == '__main__':
    # mapping = calculate_landmarks(image_path='../../dataset/Mindboggle101/Mindboggle101_volumes')
    mapping = np.asarray([[0., 31.01795976, 41.76938504, 45.2994302,  48.52944574,
                           55.18918239, 62.38044758, 69.7584229,  76.85375058, 80.15567369,
                           83.3061535,  89.67194882,  100.]])

    histogram_stardardization(mapping=mapping,
                              image_path='../../dataset/Mindboggle101/Mindboggle101_volumes',
                              output_path_hs='../../dataset/Mindboggle101/mindboggle101_hs')

    resample_image_and_label(input_path='../../dataset/Mindboggle101/mindboggle101_hs',
                             output_path_image_hs_re='../../dataset/Mindboggle101/mindboggle101_hs_re',
                             output_path_mask_hs_re='../../dataset/Mindboggle101/mindboggle101_hs_re')

    plot_hist(image_path_hs="../../dataset/Mindboggle101/mindboggle101_hs_re")
