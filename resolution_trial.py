import pyLHM.myfunctions as LHM
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import imageio as io
import time
import cv2
from PIL import Image



holo = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\PLUGIN\Holo_411_3500_5_1_1.bmp"
wvl = 411e-9                       # wavelength [m]
rec_dis = 6.28e-3                 # Reconstruction distance [m]
So_sc = 5e-3                       # L parameter in the microscope setup [m]
in_width = 1e-3                    # Width of the input plane[m]
in_height = in_width               # Height of the input plane [m]

#------------------------------ Asumptions from the geometry----------------------------------
Magn = So_sc/(rec_dis)       # Magnification of the microscope system (L/Z) [#]
# Magn = 1
out_width = in_width/Magn          # Width of the output plane [m]
out_height = in_height/Magn        # Height of the output plane [m]




#------------------------------ Library parameters preparation ----------------------------
# holo = LHM.open_image(ref)-LHM.open_image(holo)
holo = LHM.open_image(holo)
# ref = LHM.open_image(r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\PLUGIN\usaf_ref.bmp")
# holo = holo - ref

M,N = np.shape(holo)
input_pitch = [in_width/N,in_height/M]
output_pitch = [out_width/N,out_height/M]

focus_params = [holo, wvl, input_pitch, output_pitch]


focus = LHM.focus('amp')
reconstructor = LHM.reconstruct()
metrics = LHM.metrics()


rec = reconstructor.angularSpectrum(rec_dis,holo,wvl,input_pitch,output_pitch)
# rec = reconstructor.convergentSAASM(rec_dis,holo,wvl,input_pitch,output_pitch)
# focus.manual_focus('angularSpectrum', focus_params, 6.2e-3,6.4e-3,11)
image = np.abs(rec)**2
image = image - np.amin(image)
image = image/np.amax(image)



# LHM.complex_show(rec)

metrics.measure_resolution(image)




