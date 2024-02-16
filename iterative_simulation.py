import pyLHM.myfunctions as LHM
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import imageio as io
import time
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt



# Simulation parameters

df = pd.read_excel(r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Holos_list.xlsx")
df.describe()

reconstruct = LHM.reconstruct()



file_paths = [r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\USAF-sampled.png",
              r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\Grid_show.png",
              r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\Fringes.png"]

output_paths = [r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Holos\USAF",
                r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Holos\Grid",
                r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Holos\Fringes"]

output_names = [r'\usaf_',
                r'\grid_',
                r'\fringes_']
#------------------------------ Iterative Calculation ----------------------------

for i in range(df.shape[0]):
    if df['Tipo Muestra'][i] == 0:
        amplitude = LHM.open_image(file_paths[df['Archivo'][i]])
        phase = np.zeros_like(amplitude)
    else:
        phase = LHM.open_image(file_paths[df['Archivo'][i]])
        amplitude = np.ones_like(phase)

    NA = str(df['NA'][i])
    NA_str = NA.replace('.','')
    wvl = df['λ [nm]'][i] * 10**(-9)
    k = 2*np.pi/wvl
    So_sc = df['L [mm]'][i] * 10**(-3)
    So_Sa = df['Z [mm]'][i] * 10**(-3)
    out_width = df['W [mm]'][i] * 10**(-3)
    out_height = out_width
    Magn = So_sc/So_Sa
    in_width = out_width / Magn          # Width of the output plane [m]
    in_height = out_height / Magn        # Height of the output plane [m]
    index = df['Archivo'][i]
    sample = amplitude * np.exp(1j * phase)
    holo, ref = reconstruct.realisticDLHM(sample, So_sc, So_Sa, out_width, wvl, 1e-6, 2, 256)
    name_holo = output_names[index] + NA_str+'.bmp'
    name_ref  = output_names[index] + NA_str + '_ref.bmp'
    LHM.save_image(holo,output_paths[index]+name_holo)
    LHM.save_image(ref,output_paths[index]+name_ref)    

    
    



