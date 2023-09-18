# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pyLHM.myfunctions as LHM
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import imageio as io

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# Reconstruction parameters
holo = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\PLUGIN\scale_Holo_411_3500_5_1_1.bmp"                         # Hologram path
ref = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\PLUGIN\usaf_ref.bmp"
# holo = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\PLUGIN\Holo_633_3500_5_1_1.bmp"                         # Hologram path
# ref = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\PLUGIN\ref.bmp"
wvl = 633e-9                       # wavelength [m]
rec_dis = 3.5e-3                   # Reconstruction distance [m]
So_sc = 5e-3                       # L parameter in the microscope setup [m]
in_width = 1e-3                    # Width of the input plane[m]
in_height = in_width               # Height of the input plane [m]

#------------------------------ Asumptions from the geometry----------------------------------
Magn = So_sc/(So_sc-rec_dis)       # Magnification of the microscope system (L/Z) [#]
# Magn = 1
out_width = in_width/Magn          # Width of the output plane [m]
out_height = in_height/Magn        # Height of the output plane [m]




#------------------------------ Library parameters preparation ----------------------------
# holo = LHM.open_image(ref)-LHM.open_image(holo)
holo = LHM.open_image(holo)


M,N = np.shape(holo)
input_pitch = [in_width/N,in_height/M]
output_pitch = [out_width/N,out_height/M]
outshape = (M,N)
parameters = [rec_dis, holo, wvl, input_pitch, output_pitch]
focus_params = [holo, wvl, input_pitch, output_pitch]
propagator = LHM.reconstruct()
# solution = propagator.autocall('convergentSAASM',parameters)
solution = propagator.rayleigh1Free(*parameters,outshape)

M,N = np.shape(solution)


im = LHM.complex_show(solution[int(M/4):int(3*M/4),int(N/4):int(3*N/4)],negative=False)




# focusing = LHM.focus()
# xar = focusing.manual_focus('angularSpectrum',focus_params,5.6e-3,6.6e-3,11)
# gif = LHM.save_gif(xar)
# focusing.manual_focus('convergentSAASM',focus_params,2.16e-3,2.18e-3,11)
# fig = px.imshow(np.angle(solution),color_continuous_scale='gray')
# fig.show()



