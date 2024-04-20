import os
import subprocess
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.stats as stats

DEFAULT_CUTOFF = 0.01, 0.99
STANDARD_RANGE = 0, 100


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

    x1 = 0
    y1 = int((y - size_y) / 2)
    z1 = int((z - size_z) / 2)

    img_crop = img[x1: x1 + size_x, y1: y1 + size_y, z1: z1 + size_z]

    return img_crop


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


def normalize(array, landmarks, mask=None, cutoff=None, epsilon=1e-5):
    cutoff_ = DEFAULT_CUTOFF if cutoff is None else cutoff
    mapping = landmarks

    data = array
    shape = data.shape
    data = data.reshape(-1).astype(np.float32)

    if mask is None:
        mask = np.ones_like(data, np.bool)
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


def calculate_landmarks(image_path='../datasets/LPBA40/LPBA40_rigidly_registered_pairs'):
    quantiles_cutoff = DEFAULT_CUTOFF
    percentiles_cutoff = 100 * np.array(quantiles_cutoff)
    percentiles_database = []
    percentiles = _get_percentiles(percentiles_cutoff)

    count = 1
    for i in list(range(1, 41, 1)):
        for j in list(range(1, 41, 1)):
            img_path = os.path.join(image_path, 'l{}_to_l{}.hdr'.format(str(i), str(j)))
            atlas_path = os.path.join(
                image_path.replace('LPBA40_rigidly_registered_pairs', 'LPBA40_rigidly_registered_label_pairs'),
                'l{}_to_l{}.hdr'.format(str(i), str(j)))

            if os.path.exists(img_path) and os.path.exists(atlas_path):
                img_sitk = sitk.ReadImage(str(img_path))
                img_np = sitk.GetArrayFromImage(img_sitk).swapaxes(0, 2)

                mask = img_np > 0

                percentile_values = np.percentile(img_np[mask], percentiles)
                percentiles_database.append(percentile_values)
                count += 1
            else:
                raise FileNotFoundError

    percentiles_database = np.vstack(percentiles_database)
    mapping = _get_average_mapping(percentiles_database)
    print(mapping)

    np.save('../datasets/LPBA40/mapping.npy', mapping)

    return mapping


def histogram_stardardization_resample_center_crop(mapping,
                                                   input_path='../datasets/LPBA40/LPBA40_rigidly_registered_pairs',
                                                   output_path_hs_small='../datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small',
                                                   output_path_mask='../datasets/LPBA40/LPBA40_rigidly_registered_label_pairs_small'):
    if not os.path.exists(str(output_path_hs_small)):
        os.makedirs(str(output_path_hs_small))
    if not os.path.exists(str(output_path_mask)):
        os.makedirs(str(output_path_mask))

    for i in list(range(1, 41, 1)):
        for j in list(range(1, 41, 1)):
            # ~~~~~~~~~~~~~~~ images ~~~~~~~~~~~~~~~
            volpath = os.path.join(input_path, 'l{}_to_l{}.nii'.format(str(i), str(j)))
            img_sitk = sitk.ReadImage(str(volpath))
            img_np = sitk.GetArrayFromImage(img_sitk).swapaxes(0, 2)

            mask = img_np > 0

            # 1. histogram_stardardization
            img_np_hs = normalize(img_np, mapping, mask)

            img_sitk_hs = sitk.GetImageFromArray(img_np_hs.swapaxes(0, 2))
            img_sitk_hs.SetSpacing(img_sitk.GetSpacing())
            img_sitk_hs.SetDirection(img_sitk.GetDirection())
            img_sitk_hs.SetOrigin(img_sitk.GetOrigin())

            # 2. resample
            img_sitk_hs_small = resample_image(img_sitk_hs)
            img_np_hs_small = sitk.GetArrayFromImage(img_sitk_hs_small).swapaxes(0, 2)

            # 3. center_crop
            img_crop = center_crop(img=img_np_hs_small, size_ratio=(80, 106, 80)).swapaxes(0, 2)
            new_img = sitk.GetImageFromArray(img_crop)
            new_img.SetSpacing(img_sitk_hs_small.GetSpacing())
            new_img.SetDirection(img_sitk_hs_small.GetDirection())
            new_img.SetOrigin(img_sitk_hs_small.GetOrigin())

            output_path_hs_small_img = os.path.join(output_path_hs_small, 'l{}_to_l{}.nii'.format(str(i), str(j)))
            sitk.WriteImage(new_img, str(output_path_hs_small_img))

            # ~~~~~~~~~~~~~~~ masks ~~~~~~~~~~~~~~~
            atlas_path = volpath.replace('LPBA40_rigidly_registered_pairs', 'LPBA40_rigidly_registered_label_pairs')
            atlas = sitk.ReadImage(str(atlas_path))

            # 1. resample
            atlas_resampled = resample_image(atlas, is_label=True)
            atlas_np = sitk.GetArrayFromImage(atlas_resampled).swapaxes(0, 2)

            # 2. center_crop
            atlas_crop = center_crop(img=atlas_np, size_ratio=(80, 106, 80)).swapaxes(0, 2)
            new_atlas = sitk.GetImageFromArray(atlas_crop)
            new_atlas.SetSpacing(atlas_resampled.GetSpacing())
            new_atlas.SetDirection(atlas_resampled.GetDirection())
            new_atlas.SetOrigin(atlas_resampled.GetOrigin())

            output_path_hs_small_atlas = os.path.join(output_path_mask, 'l{}_to_l{}.nii'.format(str(i), str(j)))
            sitk.WriteImage(new_atlas, str(output_path_hs_small_atlas))


def histogram_stardardization_center_crop(mapping,
                                                   input_path='../datasets/LPBA40/LPBA40_rigidly_registered_pairs',
                                                   output_path_hs='../datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_large',
                                                   output_path_mask='../datasets/LPBA40/LPBA40_rigidly_registered_label_pairs_large'):
    if not os.path.exists(str(output_path_hs)):
        os.makedirs(str(output_path_hs))
    if not os.path.exists(str(output_path_mask)):
        os.makedirs(str(output_path_mask))

    for i in list(range(1, 41, 1)):
        for j in list(range(1, 41, 1)):
            # ~~~~~~~~~~~~~~~ images ~~~~~~~~~~~~~~~
            volpath = os.path.join(input_path, 'l{}_to_l{}.nii'.format(str(i), str(j)))
            img_sitk = sitk.ReadImage(str(volpath))
            img_np = sitk.GetArrayFromImage(img_sitk).swapaxes(0, 2)

            mask = img_np > 0

            # 1. histogram_stardardization
            img_np_hs = normalize(img_np, mapping, mask)

            # 2. center_crop
            img_crop = center_crop(img=img_np_hs, size_ratio=(160, 212, 160)).swapaxes(0, 2)
            new_img = sitk.GetImageFromArray(img_crop)
            new_img.SetSpacing(img_sitk.GetSpacing())
            new_img.SetDirection(img_sitk.GetDirection())
            new_img.SetOrigin(img_sitk.GetOrigin())

            output_path_hs_img = os.path.join(output_path_hs, 'l{}_to_l{}.nii'.format(str(i), str(j)))
            sitk.WriteImage(new_img, str(output_path_hs_img))

            # ~~~~~~~~~~~~~~~ masks ~~~~~~~~~~~~~~~
            atlas_path = volpath.replace('LPBA40_rigidly_registered_pairs', 'LPBA40_rigidly_registered_label_pairs')
            atlas = sitk.ReadImage(str(atlas_path))

            atlas_np = sitk.GetArrayFromImage(atlas).swapaxes(0, 2)

            # 1. center_crop
            atlas_crop = center_crop(img=atlas_np, size_ratio=(160, 212, 160)).swapaxes(0, 2)
            new_atlas = sitk.GetImageFromArray(atlas_crop)
            new_atlas.SetSpacing(atlas.GetSpacing())
            new_atlas.SetDirection(atlas.GetDirection())
            new_atlas.SetOrigin(atlas.GetOrigin())

            output_path_hs_atlas = os.path.join(output_path_mask, 'l{}_to_l{}.nii'.format(str(i), str(j)))
            sitk.WriteImage(new_atlas, str(output_path_hs_atlas))


def plot_hist(image_path_hs='../datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small'):
    for i in list(range(1, 41, 1)):
        for j in list(range(1, 41, 1)):
            volpath = os.path.join(image_path_hs, 'l{}_to_l{}.nii'.format(str(i), str(j)))
            img_sitk = sitk.ReadImage(str(volpath))
            img_np = np.clip(sitk.GetArrayFromImage(img_sitk).swapaxes(0, 2), 50, 100)
            data = np.reshape(img_np[img_np>50], -1)
            print(data.shape)
            density = stats.gaussian_kde(data)
            xs = np.linspace(40, 110, 70)
            density.covariance_factor = lambda: .25
            density._compute_covariance()

            plt.plot(xs, density(xs))
    plt.show()


if __name__ == '__main__':
    mapping = calculate_landmarks(image_path='../datasets/LPBA40/LPBA40_rigidly_registered_pairs')
    # mapping = np.asarray([1.77635684e-15, 4.02863140e+01, 5.86044434e+01, 6.33688576e+01, 6.66438972e+01, 7.12987107e+01, 7.53526276e+01, 7.96537020e+01, 8.43034770e+01, 8.67112286e+01, 8.91208850e+01, 9.35115887e+01, 1.00000000e+02])
    # mapping = np.load('../datasets/LPBA40/mapping.npy')
    print(mapping)

    histogram_stardardization_center_crop(mapping=mapping,
                                          input_path='../datasets/LPBA40/LPBA40_rigidly_registered_pairs',
                                          output_path_hs='../datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_large',
                                          output_path_mask='../datasets/LPBA40/LPBA40_rigidly_registered_label_pairs_large')

    histogram_stardardization_resample_center_crop(mapping=mapping,
                                                   input_path='../datasets/LPBA40/LPBA40_rigidly_registered_pairs',
                                                   output_path_hs_small='../datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small',
                                                   output_path_mask='../datasets/LPBA40/LPBA40_rigidly_registered_label_pairs_small')

    plot_hist()
