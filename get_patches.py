import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import random
import glob
import scipy.ndimage

class get_patches():
    def __init__(self, img,num_patches):
        self.image_fixed = img
        self.X_27_moving = None
        self.X_29_moving = None
        self.X_27_fixed = None
        self.X_29_fixed = None
        self.Y = None
        self.num_patches = num_patches

    def patches(self,):
        data_image_fixed= sitk.GetArrayFromImage(self.image_fixed)
        archivos_img = glob.glob("Images/*")
        archivos_img.sort()
        archivos_dvf = glob.glob("DVF/*")
        archivos_dvf.sort()
        longitud = len(archivos_img)
        X_27_moving = []
        X_29_moving = []
        X_27_fixed = []
        X_29_fixed = []
        Y = []


        
        for i in range(0, longitud):
            image_moving = sitk.ReadImage(archivos_img[i])
            data_image_moving = sitk.GetArrayFromImage(image_moving)
            dvf = sitk.ReadImage(archivos_dvf[i])
            dvf_data = sitk.GetArrayFromImage(dvf)
            width = image_moving.GetWidth()
            height = image_moving.GetHeight()
            depth = image_moving.GetDepth()
            

            ################################
            ### VOXELS 54X54X54#############
            ### VOXELS 29X29X29#############
            ################################
            voxels = 54
            voxels1 = 29
            voxels_moving54 = np.zeros((54, 54, 54))
            voxels_moving29 = np.zeros((29, 29, 29))
            voxels_fixed54 = np.zeros((54, 54, 54))
            voxels_fixed29 = np.zeros((29, 29, 29))


            for j in range(0, self.num_patches):
                z = random.randint(int(voxels / 2 - 1), int(depth - voxels / 2 - 1))
                y = random.randint(int(voxels / 2 - 1), int(height - voxels / 2 - 1))
                x = random.randint(int(voxels / 2 - 1), int(width - voxels / 2 - 1))
                voxels_moving54 = data_image_moving[int(z - (voxels / 2 - 1)): int(z + voxels / 2) + 1,
                            int(y - (voxels / 2 - 1)):int(y + voxels / 2) + 1,
                            int(x - (voxels / 2 - 1)): int(x + voxels / 2) + 1]
                voxels_moving29 = data_image_moving[int(z - (voxels1 - 1) / 2): int(z + (voxels1 - 1) / 2) + 1,
                            int(y - (voxels1 - 1) / 2): int(y + (voxels1 - 1) / 2) + 1,
                            int(x - (voxels1 - 1) / 2): int(x + (voxels1 - 1) / 2) + 1,np.newaxis]

                voxels_fixed54 = data_image_fixed[int(z - (voxels / 2 - 1)): int(z + voxels / 2) + 1,
                                  int(y - (voxels / 2 - 1)):int(y + voxels / 2) + 1,
                                  int(x - (voxels / 2 - 1)): int(x + voxels / 2) + 1]
                voxels_fixed29 = data_image_fixed[int(z - (voxels1 - 1) / 2): int(z + (voxels1 - 1) / 2) + 1,
                                  int(y - (voxels1 - 1) / 2): int(y + (voxels1 - 1) / 2) + 1,
                                  int(x - (voxels1 - 1) / 2): int(x + (voxels1 - 1) / 2) + 1,np.newaxis]

                voxels_moving27 = scipy.ndimage.zoom(voxels_moving54, 0.5, order=3)
                voxels_fixed27 = scipy.ndimage.zoom(voxels_fixed54, 0.5, order=3)

                voxels_moving27 =voxels_moving27[:,:,:,np.newaxis]
                voxels_fixed27 = voxels_fixed27[:, :, :, np.newaxis]

                y_data = dvf_data[z:z+1,y:y+1,x:x+1,0:3]

                X_29_moving.append(voxels_moving29)
                X_27_moving.append(voxels_moving27)
                X_29_fixed.append(voxels_fixed29)
                X_27_fixed.append(voxels_fixed27)
                Y.append(y_data)

        self.X_27_moving =np.stack(X_27_moving,axis=0)
        self.X_29_moving = np.stack(X_29_moving,axis=0)
        self.X_27_fixed = np.stack(X_27_fixed,axis=0)
        self.X_29_fixed = np.stack(X_29_fixed,axis=0)
        self.Y = np.stack(Y,axis=0)
        return self.X_27_moving,self.X_29_moving,self.X_27_fixed,self.X_29_fixed,self.Y


# if __name__ == '__main__':
#     x_moving29 = []
#     x_moving27 = []
#     x_fixed29 = []
#     x_fixed27 = []
#
#     image_fixed = sitk.ReadImage("brain.nii")
#     x_moving29,x_moving27,x_fixed29,x_fixed27 = get_patches(image_fixed)




