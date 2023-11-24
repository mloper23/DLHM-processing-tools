import pyLHM.myfunctions as LHM
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import imageio as io


# Reconstruction parameters

# holo = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\PLUGIN\Holo_411_3500_5_1_1.bmp"
holo = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Holos\Grid\grid_06.bmp"
wvl = 411e-9                       # wavelength [m]
# rec_dis = 2.49e-3                   # Reconstruction distance [m]
z_micro = 3.85e-3
So_sc = 2.49e-3                       # L parameter in the microscope setup [m]
# So_sc = 20e-3
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
ref = np.zeros_like(holo)


M,N = np.shape(holo)
input_pitch = [in_width/N,in_height/M]
output_pitch = [out_width/N,out_height/M]



# parameters = [So_sc, holo, wvl, input_pitch, output_pitch]
parameters = [z_micro, holo, wvl, So_sc, in_width,np.zeros_like(holo)]
focus_params = [holo,ref,So_sc,in_width,wvl]
# focus_params = [holo, wvl, input_pitch,output_pitch,So_sc]


propagator = LHM.reconstruct()

solution = propagator.kreuzer_reconstruct(*parameters)


# M,N = np.shape(solution)
# im = LHM.complex_show(solution[int(M/4):int(3*M/4),int(N/4):int(3*N/4)],negative=False)
im = LHM.complex_show(solution,negative=False)



# focusing = LHM.focus('amp')
# xar = focusing.manual_focus('kreuzer_reconstruct',focus_params,0.5e-3,0.7e-3,11)
# gif = LHM.save_gif(xar)
# focusing.manual_focus('convergentSAASM',focus_params,10e-3,20e-3,11)
# fig = px.imshow(np.angle(solution),color_continuous_scale='gray')
# fig.show()



