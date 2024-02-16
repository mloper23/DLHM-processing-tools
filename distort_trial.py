import pyLHM.myfunctions as LHM
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import imageio as io
import time
from PIL import Image
import cv2



file = r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Recs\Grid\08\AS08.bmp"
I = LHM.open_image(file)
metrics = LHM.metrics()
comp = cv2.adaptiveThreshold(I, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
fig = px.imshow(I,color_continuous_scale='gray')
fig.show()
fig2 = px.imshow(comp,color_continuous_scale='gray')
fig2.show()
# print(sample_width)
# distortion = metrics.measure_distortion_improved(I)

# print('Distortion: ',distortion)





