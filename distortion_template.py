import pyLHM.myfunctions as LHM
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import imageio as io
import time
import cv2
import os
import pandas as pd
from scipy.interpolate import interp1d
from PIL import Image
from plotly.subplots import make_subplots
import math as mt
from scipy.optimize import curve_fit

def distortion_behaviour(NA,a,b):
    return((a * np.power(NA,2) + b) )


def get_files_in_folder(folder_path):
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        for file_name in filenames:
            files.append(os.path.join(root, file_name))
    return files

def hex_rgba(hex, transparency):
    col_hex = hex.lstrip('#')
    col_rgb = list(int(col_hex[i:i+2], 16) for i in (0, 2, 4))
    col_rgb.extend([transparency])
    areacol = tuple(col_rgb)
    return areacol




metrics = LHM.metrics()



sheets = ['01','02','03','04','05','06','07','08']
NA_arr = np.linspace(0.1,0.8,8)
kr_arr = [True,True,True,True,True,False,False,False]
showlegend = True
dashing = ['longdash','solid','dash']
markers = ['square','x','circle']
names = ['Angular Spectrum','Kreuzer','Convolutional Rayleigh']
fig = go.Figure()

Ls = np.array([14.95, 7.4, 4.85, 3.55, 2.75, 2.19, 1.78, 1.46]) * 10**-3
distortions = [[],[],[]]
for index in range(len(kr_arr)):
    L = Ls[index]
    row = mt.floor(index/3)+1
    col = index%3 + 1
    NA_num = round(NA_arr[index],1)
    kreuzer_in = kr_arr[index]
    

    NAS = str(NA_num)
    NA = r'\ '+NAS.replace('.','')
    NA = NA.replace(' ','')
    folder = r'C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Recs\Grid'+ NA
    files = get_files_in_folder(folder)

    props_names = [r'AS',r'kreuzer',r'RS1']
    '''This lines allow to select specifically the columns in the dataframes
    that will be evaluated to pass to the function
    '''

    










    max_px = 0
    for file in files:
        idx = files.index(file)
        if kreuzer_in == False and 'kreuzer' in file:
            distortions[idx].append(np.NaN)
            continue
        I = LHM.open_image(file)
        local_dist,_ = metrics.measure_distortion_improved(I,L)
        distortions[idx].append(local_dist)
        


# ---------------------------- Interpolation ---------------------------

fits = distortions.copy()
x_fit = np.linspace(0.095,0.805,100)
for i in range(len(distortions)):
    NA_fit = NA_arr.copy()
    distortions_fit = distortions[i]
    if i == 1:
        NA_fit = [0.1,0.2,0.3,0.4,0.5]
        distortions_fit = distortions_fit[0:5]
    popt, pcov = curve_fit(distortion_behaviour, NA_fit, distortions_fit,bounds=(0,np.inf),method='trf',max_nfev=1000,diff_step=1e-3)
    fits[i] = distortion_behaviour(x_fit,*popt)



# ------------------------------- PLOTING ----------------------------
df = pd.DataFrame({
    'x':NA_arr,
    'AS':distortions[0],
    'kreuzer': distortions[1],
    'RS1':distortions[2],
})

dfit = pd.DataFrame({
    'x' : x_fit,
    'AS_fit': fits[0],
    'KR_fit': fits[1],
    'RS_fit': fits[2],
})
    

color_palette =['#CEE719',
                '#4D94AD',
                '#E69FFF']

rgba = ['rgba'+str(hex_rgba(c, transparency=0.3)) for c in color_palette]

for i, column in enumerate(df):
    if column=='x':
        continue
    x = list(df['x'])
    y1 = df[column]

    fig.add_trace(go.Scatter(x=x,
                            y=y1,
                            line=dict(color=color_palette[i-1]),
                            marker = dict(symbol=markers[i-1]),
                            mode='markers',
                            marker_line_width=1,
                            name=names[i-1]))
    
    fig.add_trace(go.Scatter(x=x,
                            y=y1,
                            line=dict(color=color_palette[i-1], width=2.5,dash=dashing[i-1]),
                            mode='lines',
                            marker_line_width=1,
                            name=names[i-1]))

# for i, column in enumerate(dfit):
#     if column=='x':
#         continue
#     x = dfit['x']
#     y1 = dfit[column]

#     fig.add_trace(go.Scatter(x=x,
#                             y=y1,
#                             line=dict(color=color_palette[i-1], width=2.5,dash=dashing[i-1]),
#                             mode='lines',
#                             name=names[i-1]))

fig.update_xaxes(linecolor='black',gridcolor='lightgrey', range=[NA_arr[0]-0.05,NA_arr[7]+0.05], mirror=True)
fig.update_yaxes(linecolor='black',gridcolor='lightgrey', mirror=True)

        
fig.update_xaxes(title_text= 'Numerical Aperture')
fig.update_yaxes(title_text= 'Distortion [%]',
                 showexponent = 'all',
                 exponentformat = 'power')


fig.update_layout(legend=dict(title='Reconstruction\nmethod'), showlegend=showlegend,plot_bgcolor='white',title_x=0.5)
fig.update_layout(
                title=dict(text='Percentual distortion on every NA', font=dict(size=40), yref='paper'),
                font_family='Arial',
                font_color="black",
                title_font_family="Arial",
                title_font_color="black",
                )                                 
fig.update_layout(
font=dict(
    family="Arial",
    size=30,  # Set the font size here
    color="black"))


config = {
  'toImageButtonOptions': {
    'format': 'svg', # one of png, svg, jpeg, webp
    'filename': 'Distortion_behaviour',
    'height': 700,
    'width': 1024,
    'scale':50 # Multiply title/legend/axis/canvas sizes by this factor
  }
}

# fig.show()
fig.show(config=config)



