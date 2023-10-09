import pyLHM.myfunctions as LHM
import numpy as np
import cv2 as cv
import plotly.express as px

# img = np.array(cv.imread('/Users/mjloperaa/Library/CloudStorage/OneDrive-UniversidadEAFIT/EAFIT/DLHM-data/04172023/sample2.png'))[:,:,0]
# holo = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\PLUGIN\Holo_gridshow_633_3500_5_1_1.bmp"                         # Hologram path
# path = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Fringes.png"
path = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\USAF-sampled.png"

img0 = LHM.open_image(path)
img = img0
# img = np.ones_like(img0)*np.exp(1j*img0*8)
reconstruct = LHM.reconstruct()

wvl = 532e-9
L = 65e-3
Z = 40e-3
w_c = 20.48e-3

holo, ref = reconstruct.realisticDLHM(img, wvl, L, Z, w_c,  1e-9, 2, 256)
# im = LHM.save_image(holo,'realistic_phase.bmp')
# im = LHM.save_image(ref,'ref_realistic_phase.bmp')
# reconstruction = reconstruct.convergentSAASM_pad(0.96e-3,holo-ref,wvl,np.divide(w_c,np.shape(holo)),np.divide(w_c,np.shape(holo)))
# LHM.complex_show(reconstruction)
foc = LHM.focus('amp')
foc_params = [holo, wvl, np.divide(w_c,np.shape(holo)),np.divide(w_c,np.shape(holo))]
# z = foc.manual_focus('convergentSAASM_pad',foc_params,0.8e-3,1.2e-3,11)

# reconstruction = reconstruct.angularSpectrum(5e-3,holo-ref,405e-9,np.divide(1e-3,np.shape(holo)),np.divide(1e-3,np.shape(holo)))
# fig = px.imshow(holo)
# fig.show()
# fig = px.imshow(reconstruct.norm_bits(np.angle(reconstruction),255)[200:800,200:800])
# fig.show()

# rec = rec.convergentSAASM(33e-3,holo-ref,405e-9,[7e-3/])



