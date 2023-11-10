import pyLHM.myfunctions as LHM
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import imageio as io



# Reconstruction parameters

# holo = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Holos\Grid\grid_01.bmp"
holo = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Holos\USAF\usaf_08.bmp"


wvl = 411e-9                       # wavelength [m]
k_wvl = 2 * np.pi/wvl
# rec_dis = 3.444e-3                  # Reconstruction distance [m]
So_sc = 1.46e-3                    # L parameter in the microscope setup [m]
# z_micro = So_sc-rec_dis
z_micro = 0.94e-3
in_width = 3e-3                    # Width of the input plane[m]
in_height = in_width               # Height of the input plane [m]

#------------------------------ Asumptions from the geometry----------------------------------
Magn = So_sc/(z_micro)       # Magnification of the microscope system (L/Z) [#]
# Magn = 1
out_width = in_width/Magn          # Width of the output plane [m]
out_height = in_height/Magn        # Height of the output plane [m]


#------------------------------ Library parameters preparation ----------------------------
# holo = LHM.open_image(ref)-LHM.open_image(holo)
holo = LHM.open_image(holo)
propagators = ['angularSpectrum','kreuzer_reconstruct','rayleigh_convolutional']

# holo = LHM.open_image(holo)





M,N = np.shape(holo)
input_pitch = in_width/M
output_pitch = out_width/N
outshape = (M,N)
parameters = [z_micro, holo, wvl, So_sc, in_width, np.zeros_like(holo)]

focus_params =[[holo, wvl, input_pitch, So_sc],
               [holo, wvl, So_sc, in_width,np.zeros_like(holo)],
               [holo, wvl, So_sc, in_width]]


propagator = LHM.reconstruct()
# solution = propagator.autocall('angularSpectrum',parameters)
# solution = propagator.kreuzer_reconstruct(*parameters)
# M,N = np.shape(solution)
# x = (np.arange(M)-M/2)
# X,Y = np.meshgrid(x*output_pitch,x*output_pitch)
# r = np.sqrt(z_micro **2 + X**2  + Y**2)
# psrc = np.exp(-1j * k_wvl * r)
# solution = propagator.convergentSAASM(*parameters)



# M,N = np.shape(solution)
# im = LHM.complex_show(solution[int(M/4):int(3*M/4),int(N/4):int(3*N/4)],negative=False)
# im = LHM.complex_show(solution,negative=False)




idx = 2
focusing = LHM.focus('amp')
# xar = focusing.manual_focus(propagators[idx],focus_params[idx],0.1*So_sc,So_sc,11)
xar = focusing.manual_focus(propagators[idx],focus_params[idx],0.38e-3,0.39e-3,11)
# gif = LHM.save_gif(xar)
# focusing.manual_focus('convergentSAASM',focus_params,10e-3,20e-3,11)
# fig = px.imshow(np.angle(solution),color_continuous_scale='gray')
# fig.show()



