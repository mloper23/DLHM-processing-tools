# DISTORTION: For more information look at https://www.edmundoptics.com/knowledge-center/application-notes/imaging/distortion/
# RESOLUTION: Measurement of the MTF (Take different profiles in the USAF groups to get an uncertainty)
# CONTRAST: Measure sharpness in a certain area
# NOISE: What is background? Std deviation of the background: An alternative is https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image 
# RECONSTRUCTION TIME: Timer on each propagator
# DISTANCE: Percentage error in the reconstruction distance for the focused image
# PHASE SENSITIVITY: Yet to be discussed

import pyLHM.myfunctions as lhm
import numpy as np

# Reconstruction parameters
holo = r""                         # Hologram path
wvl = 633e-9                       # wavelength [m]
rec_dis = 1e-3                     # Reconstruction distance [m]
So_sc = 2e-3                       # L parameter in the microscope setup [m]
in_width = 1e-3                    # Width of the input plane[m]
in_height = in_width               # Height of the input plane [m]

#------------------------------ Asumptions from the geometry----------------------------------
Magn = So_sc/(So_sc-rec_dis)       # Magnification of the microscope system (L/Z) [#]
out_width = Magn*in_width          # Width of the output plane [m]
out_height = Magn*in_height        # Height of the output plane [m]




#------------------------------ Library parameters preparation ----------------------------
holo = lhm.open_image(holo)
M,N = np.shape(holo)
input_pitch = [in_width/N,in_height/M]
output_pitch = [out_width/N,out_height/M]
outshape = (M,N)
FC = 0
parameters = [rec_dis, holo, wvl, input_pitch, output_pitch]

#------------------------------ SHOULD I INCLUDE THE AUTOFOCUS CODE HERE????----------------





#___________________________________________________________________________________________







# Variables initialization
noise = {}
distortion = {}
resolution = {}
contrast = {}
time = {}
rec_distance = {}       # [mm]
rel_rec_distance={}
phase_sensitivity={}





# Setting all the propagators names in the list to reconstruct the same image through all the propagators
propagators_list = ['angularSpectrum','rayleigh1Free','convergentSAASM','kreuzer3F','realisticAS']

reconstructed_images = {}

for propagator in propagators_list:
    params = parameters
    if propagator == 'kreuzer3f':
        params = params + [So_sc,FC]
    elif propagator == 'rayleigh1Free':
        params = params + [outshape]
    rec = lhm.reconstruct(propagator,params)
    reconstructed_images[propagator] = rec

for propagator in propagators_list:
    im = reconstructed_images[propagator]
    noise[propagator] = lhm.measure_noise(im)
    # -------------- All the other metrics shall be calculated here --------------------






    #___________________________________________________________________________________





