import pyLHM.myfunctions as LHM
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import imageio as io
import time
from PIL import Image


# Simulation parameters
# amplitude = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\USAF-1951.png"                         # Hologram path
amplitude = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\USAF-1951_256.bmp"
# phase = r""
wvl = 633e-9                       # wavelength [m]
So_Sa = 5e-3                   # Reconstruction distance [m]
So_sc = 15e-3                       # L parameter in the microscope setup [m]
out_width = 1e-3                    # Width of the input plane[m]
out_height = out_width               # Height of the input plane [m]

#------------------------------ Asumptions from the geometry----------------------------------
Magn = So_sc/So_Sa                  # Magnification of the microscope system (L/Z) [#]
# Magn = 1
in_width = out_width / Magn          # Width of the output plane [m]
in_height = out_height / Magn        # Height of the output plane [m]




#------------------------------ Library parameters preparation ----------------------------

amplitude = LHM.open_image(amplitude)
M,N = np.shape(amplitude)
input_pitch = [in_width/N,in_height/M]
output_pitch = [out_width/N,out_height/M]
outshape = (512,512)
x = np.arange(0, N, 1)  # array x
y = np.arange(0, M, 1)  # array y
X_in, Y_in = np.meshgrid((x - (N / 2)) * input_pitch[0], (y - (M / 2)) * input_pitch[1], indexing='xy')
r = np.sqrt(X_in**2 + Y_in**2 + So_Sa**2)
phase = np.exp(1j * (2*np.pi / wvl) * r) / r


sample = amplitude * phase
# sample = amplitude
parameters = [So_sc-So_Sa, sample, wvl, input_pitch, output_pitch]
focus_params = [sample, wvl, input_pitch, output_pitch]
propagator = LHM.reconstruct()
# solution = propagator.autocall('convergentSAASM',parameters)
start = time.time()
# solution = propagator.rayleigh1Free(*parameters,outshape)
solution = propagator.angularSpectrum(*parameters)
stop = time.time()
print('Runtime = ',stop-start,' s')

intensity = np.abs(solution)**2
# im = LHM.save_image(intensity,'raleygh_633_500_5_1_1.bmp')

M,N = np.shape(solution)
# im = LHM.complex_show(solution[int(M/4):int(3*M/4),int(N/4):int(3*N/4)],negative=False)
im = LHM.complex_show(solution,negative=False)




# focusing = LHM.focus()
# xar = focusing.manual_focus('angularSpectrum',focus_params,5.6e-3,6.6e-3,11)
# gif = LHM.save_gif(xar)
# focusing.manual_focus('convergentSAASM',focus_params,2.16e-3,2.18e-3,11)
# fig = px.imshow(np.angle(solution),color_continuous_scale='gray')
# fig.show()



