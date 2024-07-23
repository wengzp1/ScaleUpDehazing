import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# do gamma correction
#img_gamma1 = np.power(img, gamma).clip(0,255).astype(np.uint8)

# gamma correction
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def do_gamma_correction(folder, gamma_B, gamma_G, gamma_R):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        # do gamma correction
        gammaImg_B = gammaCorrection(img[:,:, 0], gamma_B)
        gammaImg_G = gammaCorrection(img[:,:, 1], gamma_G)
        gammaImg_R = gammaCorrection(img[:,:, 2], gamma_R)
        gammaImg = np.dstack((gammaImg_B,gammaImg_G, gammaImg_R))

        # save to gamma folder
        cv2.imwrite(os.path.join(savepath  , filename), gammaImg)


readpath="D:\\Download\\RSID\\hazy"
savepath="D:\\Download\\RSID\\gammahazy"
do_gamma_correction(readpath, gamma_B=1.91, gamma_G=1.91, gamma_R=1.90)
