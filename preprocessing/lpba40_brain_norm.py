import os, glob
import numpy as np
import SimpleITK as sitk

def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):

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


t1_fn = glob.glob('/media/lhliu/Datas1/public_dataset/7_REGISTRATION/Data/LPBA40/LPBA40_rigidly_registered_pairs/*hdr')
for i in t1_fn:
    # image
    print(i)
    t1 = sitk.ReadImage(i)
    t1_image_masked = sitk.GetArrayFromImage(t1).astype(np.float32)

    # Whitening Norm
    signs = (t1_image_masked == 0.0)

    mean1 = np.mean(t1_image_masked[t1_image_masked != 0.0])
    std1 = np.std(t1_image_masked[t1_image_masked != 0.0])
    t1_image_masked -= mean1
    t1_image_masked /= std1
 
    # Clip extreme values and outliers, and then to [0,1]. (Noted, this step is not necessary if Whiten is used.)
    t1_image_masked = (np.clip(t1_image_masked, -3., 3.) + 3.0) / 6.0

    t1_image_masked[signs] = 0
    t1_masked = sitk.GetImageFromArray(t1_image_masked)
    t1_masked.SetSpacing(t1.GetSpacing())
    t1_masked.SetDirection(t1.GetDirection())
    t1_masked.SetOrigin(t1.GetOrigin())

    # t1_1mm = resample_image(t1_masked)

    output_dir = "/media/lhliu/Datas1/public_dataset/7_REGISTRATION/Data/LPBA40/new_LPBA40_rigidly_registered_pairs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sitk.WriteImage(t1_masked, os.path.join(output_dir, i.split("/")[-1]))
