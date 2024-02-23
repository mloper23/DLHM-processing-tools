import pyLHM.myfunctions as LHM
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import imageio as io
import time
from PIL import Image
import cv2

Ls = np.array([14.95, 7.4, 4.85, 3.55, 2.75, 2.19, 1.78, 1.46])


file = r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Recs\Grid\08\AS08.bmp"
I = LHM.open_image(file)
I = (I * 255).astype(np.uint8)
metrics = LHM.metrics()
comp = cv2.adaptiveThreshold(I, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
# print(sample_width)
distortion,_ = metrics.measure_distortion_improved(I,Ls[7])

print('Distortion: ',distortion)





