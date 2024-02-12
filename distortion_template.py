import pyLHM.myfunctions as LHM
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import imageio as io
import time
import cv2
import os
import pandas as pd
from scipy.interpolate import interp1d
from PIL import Image

def get_files_in_folder(folder_path):
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        for file_name in filenames:
            files.append(os.path.join(root, file_name))
    return files

metrics = LHM.metrics()




kreuzer_in = True
NAs = np.linspace(0.1,0.8,8)
NAs = [round(n,1) for n in NAs]



# props_names = [r'realistic',r'AS',r'kreuzer',r'SAASM']
# profiles = [493,493,483]

props_names = [r'AS',r'kreuzer',r'realistic']
kreuzer = [True,True, False, False, False, False, False,False]

distortions_array = np.zeros((8,3))
for NA_local in NAs:
    NA_idx = NAs.index(NA_local)
    kreuzer_in = kreuzer[NAs.index(NA_local)]
    NA = r'\ '.replace(' ','') + str(NA_local).replace('.','')
    folder = r'F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Recs\Grid'+ NA
    files = get_files_in_folder(folder)
    if kreuzer_in == True:
        distortions = [0,0,0]
        for file in files:
            if 'SAASM' in file:
                continue
            idx = files.index(file)
            I = LHM.open_image(file)
            distortions[idx] = metrics.measure_distortion_improved(I)
            # noise = metrics.measure_noise(I)
        distortions_array[NA_idx] = distortions

    else:
        distortions = [0,0]
        for file in files:
            idx = files.index(file)
            I = LHM.open_image(file)
            distortions[idx] = metrics.measure_distortion_improved(I)
        distortions_array[NA_idx, 0] = distortions[0]
        distortions_array[NA_idx, 2] = distortions[1]




print(distortions)


