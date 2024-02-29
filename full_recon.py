import pyLHM.myfunctions as LHM
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import imageio as io
import cv2
import time
import pandas as pd
from PIL import Image
import os

def get_files_in_folder(folder_path):
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        for file_name in filenames:
            files.append(os.path.join(root, file_name))
    return files



def find_in(input_string, template):
    return template in input_string


# Simulation parameters

df = pd.read_excel(r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\focus_dist.xlsx")
df.describe()

df_keys = ['AS z \nfocus [mm]', 'kreuzer z\nfocus [mm]','Rayleigh z focus [mm]']

reconstruct = LHM.reconstruct()

gen_inp_path = r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Holos"

gen_out_path = r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Recs"

propagators = ['angularSpectrum','kreuzer_reconstruct','rayleigh_convolutional']
props_names = [r'AS',r'kreuzer',r'RS1']

file_paths = [r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\USAF-sampled.png",
              r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\Grid_show.png",
              r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\Fringes.png"]

output_paths = [r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Holos\USAF",
                r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Holos\Grid",
                r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Holos\Fringes"]

samples = [r'\Usaf',r'\Grid',r'\Fringes']
NAS = np.linspace(0.1,0.8,8)
NAS = [round(n,1) for n in NAS]
#------------------------------ Iterative Calculation ----------------------------


M = N = 1024
wvl = 411e-9                       # wavelength [m]
in_width = 3e-3                    # Width of the input plane[m]
in_height = in_width               # Height of the input plane [m]
input_pitch = in_width/M
#------------------------------ Asumptions from the geometry----------------------------------

k_wvl = 2*np.pi/wvl
reconstruction_times = [[],[],[],[]]

for sample in range(3):
    if sample !=2: # ERRASE THIS LINE IF YOU WANT TO RECONSTRUCT ALL THE HOLOGRAMS AGAIN
        continue
    folder = gen_inp_path + samples[sample]
    files = get_files_in_folder(folder)
    files = [i for i in files if 'ref' not in i]
    
    for i in range(df.shape[0]):
        holopath = files[i]
        holo = LHM.open_image(holopath)
        for prop in propagators:
            idx = propagators.index(prop)
            So_sc = df['L [mm]'][i]*10**-3                        # L parameter in the microscope setup [m]
            z_micro = df[df_keys[idx]][i] *10**-3
            if np.isnan(z_micro):
                continue
            Magn = So_sc/(z_micro)       # Magnification of the microscope system (L/Z) [#]
            out_width = in_width/Magn          # Width of the output plane [m]
            out_height = in_height/Magn        # Height of the output plane [m]
            output_pitch = out_width/N
            focus_params =[[z_micro, holo, wvl, input_pitch, So_sc],
                           [z_micro, holo, wvl, So_sc, in_width,np.zeros_like(holo)],
                           [z_micro, holo, wvl, So_sc, in_width]]

            start_rec = time.time()
            reconstruction = reconstruct.autocall(prop,focus_params[idx])
            stop_rec = time.time()
            rectime = stop_rec - start_rec
            reconstruction_times[idx].append(rectime)
            NA_file = str(NAS[i]).replace('.','')
            NA_str = (r'\ ' + NA_file).replace(' ','')
            filename = (r'\ ' + props_names[idx] + NA_file + '.bmp').replace(' ','')
            outfolder = gen_out_path + samples[sample] + NA_str
            path_fin = outfolder+filename
            if sample == 2:
                # image = reconstruct.norm_bits(np.angle(reconstruction),256)
                # point_src = np.exp(-i * k_wvl)
                tramitance = np.ones_like(holo)
                ref_params =[[z_micro, tramitance, wvl, input_pitch, So_sc],
                             [z_micro, tramitance, wvl, So_sc, in_width,np.zeros_like(tramitance)],
                             [z_micro, tramitance, wvl, So_sc, in_width]]
                reference = reconstruct.autocall(prop,ref_params[idx])
                if prop == 'rayleigh_convolutional':
                    desenv = reconstruction * np.conjugate(reference)
                else:
                    desenv = np.conjugate(reconstruction) * (reference)
                image = np.angle(desenv)
            else:
                # image = reconstruct.norm_bits(np.abs(reconstruction)**2,256)
                image = np.abs(reconstruction) ** 2

            if  prop == 'kreuzer_reconstruct' or prop == 'rayleigh_convolutional':
                image = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_AREA)
            LHM.save_image(image,path_fin)
        # break # ERRASE THIS LINE TOO---------------------------------------------------------------
    print(sample+1)



# df = pd.DataFrame(data={'realistic':reconstruction_times[0],
#                         'AS':reconstruction_times[1],
#                         'kreuzer':reconstruction_times[2],
#                         'SAASM':reconstruction_times[3]})
# df.to_csv(r'F:\OneDrive - Universidad EAFIT\Semestre X\TDG\reconstruction_times.csv')


        

    
    



