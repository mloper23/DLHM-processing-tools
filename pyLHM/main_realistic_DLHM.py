from myfunctions import *
import numpy as np
import cv2 as cv

img = np.array(cv.imread('/Users/mjloperaa/Library/CloudStorage/OneDrive-UniversidadEAFIT/EAFIT/DLHM-data/04172023/sample2.png'))[:,:,0]

holo, ref = realisticDLHM(img, 40e-3, 7e-3, 4.71e-3, 405e-9, 1e-6, 2, 256)
fig = px.imshow(holo)
fig.show()
