import pyLHM.myfunctions as LHM
import numpy as np
import cv2 as cv
import plotly.express as px

# img = np.array(cv.imread('/Users/mjloperaa/Library/CloudStorage/OneDrive-UniversidadEAFIT/EAFIT/DLHM-data/04172023/sample2.png'))[:,:,0]
# holo = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\PLUGIN\Holo_gridshow_633_3500_5_1_1.bmp"                         # Hologram path
holo = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Fringes.png"
img = LHM.open_image(holo)
img = np.ones_like(img)*np.exp(1j*2*np.pi*img/(405e-9))
reconstruct = LHM.reconstruct()

holo, ref = reconstruct.realisticDLHM(img, 405e-9, 15e-3, 10e-3, 1e-3,  1e-6, 2, 256)
# im = LHM.save_image(holo,'realistic_phase.bmp')
# im = LHM.save_image(ref,'ref_realistic_phase.bmp')
# reconstruction = reconstruct.convergentSAASM(2e-3,holo-ref,405e-9,np.divide(1e-3,np.shape(holo)),np.divide(1e-3,np.shape(holo)))
foc = LHM.focus('phase')
foc_params = [holo, 405e-9, np.divide(1e-3,np.shape(holo)),np.divide(1e-3,np.shape(holo))]
z = foc.manual_focus('angularSpectrum',foc_params,10.5e-3,10.7e-3,21)

reconstruction = reconstruct.angularSpectrum(12e-3,holo-ref,405e-9,np.divide(1e-3,np.shape(holo)),np.divide(1e-3,np.shape(holo)))
fig = px.imshow(holo)
fig.show()
# fig = px.imshow(reconstruct.norm_bits(np.angle(reconstruction),255)[200:800,200:800])
# fig.show()

# rec = rec.convergentSAASM(33e-3,holo-ref,405e-9,[7e-3/])



