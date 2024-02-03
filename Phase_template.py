import pyLHM.myfunctions as LHM
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import imageio as io
import time
import cv2
import os
import pandas as pd
from scipy.interpolate import interp1d
from PIL import Image

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

def smoothen(arr, winsize=5):
    max_px = len(arr)
    x_supreme = [i for i in range(max_px)]
    arr = np.array(pd.Series(arr).rolling(winsize).mean())[winsize-1:]
    x = np.linspace(0,max_px,len(arr))
    inter = interp1d(x, arr, kind='linear')
    arr = inter(x_supreme)
    return arr

metrics = LHM.metrics()
sample_path = r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\Fringes.png"
sample = LHM.open_image(sample_path)
profile_sample = 2 * np.pi * metrics.measure_phase_sensitivity(sample)
x = [i for i in range(len(profile_sample))]

NAS = '0.1'
kreuzer_in = True

NA = r'\ '+NAS.replace('.','')
NA = NA.replace(' ','')
folder = r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Recs\Fringes"+ NA
files = get_files_in_folder(folder)



props_names = [r'AS',r'kreuzer',r'RS1']

profile_lists = [0,0,0,profile_sample]
smooth_profiles = [0,0,0]
steps_lists = [0,0,0]
error_lists = steps_lists.copy()

steps_indexes = [[(162,186),(230,250),(295,315),(361,379),(426,445),(491,510),(557,576),(624,642)],
                 [(130,151),(203,221),(283,305),(355,375),(430,450),(505,523),(579,597),(649,670)],
                 [(162,188),(229,251),(295,315),(361,380),(426,446),(492,511),(558,574),(623,643)]]


reference = [0.0985598, 0.2218, 0.3696, 0.4682, 0.617, 0.7392, 0.8624, 1.0102]

max_px = len(profile_sample)
for file in files:
    idx = files.index(file)
    I = LHM.open_image(file)
    profile_lists[idx] = 2*np.pi*metrics.measure_phase_sensitivity(I)
    if len(profile_lists[idx])>max_px:
        max_px = len(profile_lists[idx])
    smooth_profiles[idx] = smoothen(profile_lists[idx],7)   #errase this line to avoid filtering

#Calculations of the mean values on the steps
for i in range(3):
    steps = [0]*8
    error_list = steps.copy()
    for j in range(8):
        step_start = steps_indexes[i][j][0]
        step_stop = steps_indexes[i][j][1]
        step_mean = np.mean(profile_lists[i][step_start:step_stop])
        error_list[j] = np.abs((step_mean-reference[j])/reference[j])
        steps[j] = step_mean
    error_lists[i] = error_list
    steps_lists[i] = steps


errors = [np.amax(i) for i in error_lists]



x_supreme = [(7.32e-4 *(i-int(max_px/2))) for i in range(max_px)]


color_palette =['#000000',
               '#CEE719',
               '#4D94AD',
               '#E69FFF']

show_raw = True
transparency = 0.75
rgba = ['rgba'+str(hex_rgba(c, transparency=transparency)) for c in color_palette]
rgba = rgba + ['rgba'+str(hex_rgba(c, transparency=transparency-0.5)) for c in color_palette]
showlegend = False

ind = 0
# for profile in profile_lists:
#     x = np.linspace(0,max_px,len(profile))
#     inter = interp1d(x, profile, kind='linear')
#     profile_lists[ind] = inter(x_supreme)
#     ind = ind+1

if kreuzer_in == True:
    df = pd.DataFrame({
        'x':x_supreme,
        'Angular Spectrum':profile_lists[0],
        'Kreuzer': profile_lists[1],
        'Convolutional Rayleigh':profile_lists[2],
        'Sample':profile_lists[3],
        'SAS':smooth_profiles[0],
        'SKR':smooth_profiles[1],
        'SRS':smooth_profiles[2],
    })

    fig = px.line(df, x='x', y=['Sample','SAS','SKR','SRS'], title='Profile comparison')
    fig.update_traces(line=dict(color=rgba[0],dash='dash'), selector=dict(mode='lines', name='Sample'))
    fig.update_traces(line=dict(color=rgba[1]), selector=dict(mode='lines', name='SAS'))
    fig.update_traces(line=dict(color=rgba[2]), selector=dict(mode='lines', name='SKR'))
    fig.update_traces(line=dict(color=rgba[3]), selector=dict(mode='lines', name='SRS'))

else:
    df = pd.DataFrame({
        'x':x_supreme,
        'Angular Spectrum':profile_lists[0],
        'Convolutional Rayleigh':profile_lists[2],
        'Sample':profile_lists[3],
        'SAS':smooth_profiles[0],
        'SKR':smooth_profiles[1],
        'SRS':smooth_profiles[2],
    })

    fig = px.line(df, x='x', y=['Sample','SAS','SRS'], title='Profile comparison')
    fig.update_traces(line=dict(color=rgba[0],dash='dash'), selector=dict(mode='lines', name='Sample'))
    # fig.update_traces(line=dict(color=rgba[1]), selector=dict(mode='lines', name='Angular Spectrum'))
    # fig.update_traces(line=dict(color=rgba[3]), selector=dict(mode='lines', name='Convolutional Rayleigh'))
    fig.update_traces(line=dict(color=rgba[1]), selector=dict(mode='lines', name='SAS'))
    fig.update_traces(line=dict(color=rgba[3]), selector=dict(mode='lines', name='SRS'))

fig.update_xaxes(title_text= 'Radius [mm]',linecolor='black',gridcolor='lightgrey',mirror=True)
fig.update_yaxes(title_text='Phase [rad]',linecolor='black',gridcolor='lightgrey', mirror=True)
fig.update_layout(legend=dict(title='Reconstruction\nmethod'), showlegend=showlegend,plot_bgcolor='white')
fig.update_layout(
                title=dict(text='Phase ladder profile comparison NA '+NAS, font=dict(size=40), yref='paper'),
                font_family='Arial',
                font_color="black",
                title_font_family="Arial",
                title_font_color="black",
                )                                 
fig.update_layout(
font=dict(
    family="Arial",
    size=30,  # Set the font size here
    color="black"
)
)
# print(errors)
folder_name = r'C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Graphs\Phase'+NA+'profile.html'
fig.write_html(folder_name)




