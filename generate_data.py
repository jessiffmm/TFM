import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import random
from dvf import *

class generate_data():
    def __init__(self):
        self.num_img = 19
        self.img_in =  sitk.ReadImage('brain.nii')
        #self.DVF = []

    def generate_dvf(self):
        #Low frequency
        for i in range(0, self.num_img):
            DVF_low = dvf(self.img_in)
            DVF_low.sigma_dvf = 35
            DVF_low.P = 100
            DVF_low.displacement = 15
            DVF_low.sigma_image = 5
            DVF_low.transform(i)
            sitk.WriteImage(sitk.Cast(DVF_low.DeformedIM, sitk.sitkInt16), "Images/"+ str(i) + ".nii")


        #Medium frequency
        for i in range(self.num_img, self.num_img*2):
            DVF_medium = dvf(self.img_in)
            DVF_medium.sigma_dvf = 25
            DVF_medium.P = 100
            DVF_medium.displacement = 15
            DVF_medium.sigma_image = 5
            DVF_medium.transform(i)
            sitk.WriteImage(sitk.Cast(DVF_medium.DeformedIM, sitk.sitkInt16), "Images/" + str(i) + ".nii")

        # High frequency
        for i in range(self.num_img*2, self.num_img*3):
            DVF_high = dvf(self.img_in)
            DVF_high.sigma_dvf = 20
            DVF_high.P = 100
            DVF_high.displacement = 15
            DVF_high.sigma_image = 5
            DVF_high.transform(i)
            sitk.WriteImage(sitk.Cast(DVF_high.DeformedIM, sitk.sitkInt16), "Images/" + str(i) + ".nii")



        # New displacement
        for i in range(0,self.num_img*3):
            img_transform = sitk.ReadImage("Images/" + str(i) + ".nii")
            DVF_low_1 = dvf(img_transform)
            DVF_low_1.sigma_dvf = 35
            DVF_low_1.P = 100
            DVF_low_1.displacement = 15
            DVF_low_1.sigma_image = 3
            DVF_low_1.transform(i+self.num_img*3)
            sitk.WriteImage(sitk.Cast(DVF_low_1.DeformedIM, sitk.sitkInt16), "Images/" + str(i+self.num_img*3) + ".nii")
