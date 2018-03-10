import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

import random

class dvf():
    def __init__(self, img):
        self.image_fixed = img
        self.image_fixed_data =  None
        self.DeformedIM = None
        self.P = None
        self.sigma_dvf = None
        self.sigma_image = None
        self.displacement = None
        self.deformedDVF= None



    def transform(self,num_img):

        self.image_fixed_data = sitk.GetArrayFromImage(self.image_fixed)

        self.deformedDVF = np.zeros([self.image_fixed_data.shape[0],self.image_fixed_data.shape[1],
                                self.image_fixed_data.shape[2],3], dtype=np.float64)

        DVF_x = np.zeros(self.image_fixed_data.shape,dtype=np.float64)
        DVF_y = np.zeros(self.image_fixed_data.shape, dtype=np.float64)
        DVF_z = np.zeros(self.image_fixed_data.shape, dtype=np.float64)



        for num_p in range(1, self.P):
            posx = random.randint(0, self.image_fixed_data.shape[2] - 1);
            posy = random.randint(0, self.image_fixed_data.shape[1] - 1);
            posz = random.randint(0, self.image_fixed_data.shape[0] - 1);
            DVF_x[posz, posy, posx] = random.randint(-self.displacement, self.displacement);
            DVF_y[posz, posy, posx] = random.randint(-self.displacement, self.displacement);
            DVF_z[posz, posy, posx] = random.randint(-self.displacement, self.displacement);

        DVF_xb = gaussian_filter(DVF_x, self.sigma_dvf)
        DVF_yb = gaussian_filter(DVF_y, self.sigma_dvf)
        DVF_zb = gaussian_filter(DVF_z, self.sigma_dvf)

        DVF_xb = (np.max(abs(DVF_x))/np.max(abs(DVF_xb)))*DVF_xb
        DVF_yb = (np.max(abs(DVF_y)) / np.max(abs(DVF_yb))) * DVF_yb
        DVF_zb = (np.max(abs(DVF_z)) / np.max(abs(DVF_zb))) * DVF_zb

        self.deformedDVF[:, :, :, 0] = DVF_xb
        self.deformedDVF[:, :, :, 1] = DVF_yb
        self.deformedDVF[:, :, :, 2] = DVF_zb
        #deformedDVF = deformedDVF*100

        deformedDVF_im = sitk.GetImageFromArray(self.deformedDVF)

        # print(deformedDVF_im.GetOrigin())
        # print (deformedDVF_im.GetWidth(), deformedDVF_im.GetHeight(), deformedDVF_im.GetDepth())
        # deformedDVF_im.SetOrigin(_FixedIm.GetOrigin())
        # deformedDVF_im.SetOrigin(image.GetOrigin())

        sitk.WriteImage(sitk.Cast(deformedDVF_im,sitk.sitkVectorFloat32),"DVF/"+str(num_img)+".nii")

        DVF_T = sitk.DisplacementFieldTransform(deformedDVF_im)

        Deformed_im_Clean = sitk.Resample(sitk.GetImageFromArray(self.image_fixed_data),DVF_T)

        #print(np.sum(abs(image_fixed_data-Deformed_im_Clean)))
        self.DeformedIM = sitk.AdditiveGaussianNoise(Deformed_im_Clean,self.sigma_image)
        self.DeformedIM.SetDirection(self.image_fixed.GetDirection())

        #DeformedIM.SetOrigin(image_fixed.GetOrigin())
        #DeformedIM_data = sitk.GetArrayFromImage(self.DeformedIM)
        #return self.DeformedIM








