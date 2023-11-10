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

metrics = LHM.metrics()
sample_path = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\Fringes.png"
sample = LHM.open_image(sample_path)
profile_sample = 2 * np.pi * metrics.measure_phase_sensitivity(sample,True)
x = [i for i in range(len(profile_sample))]



kreuzer_in = True
NA = r'\01'
folder = r'F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Recs\Fringes'+ NA
files = get_files_in_folder(folder)

# props_names = [r'realistic',r'AS',r'kreuzer',r'SAASM']
# profiles = [493,493,483]

props_names = [r'AS',r'kreuzer',r'realistic']
if kreuzer_in == True:
    profile_lists = [0,0,0,profile_sample]
else:
    profile_lists = [0,0,profile_sample]







max_px = len(profile_sample)
for file in files:
    if 'SAASM' in file:
        continue
    idx = files.index(file)
    I = LHM.open_image(file)
    profile_lists[idx] = metrics.measure_phase_sensitivity(I, False)
    if len(profile_lists[idx])>max_px:
        max_px = len(profile_lists[idx])

x_supreme = [i for i in range(max_px)]

ind = 0
for profile in profile_lists:
    x = np.linspace(0,max_px,len(profile))
    inter = interp1d(x, profile, kind='linear')
    profile_lists[ind] = inter(x_supreme)
    ind = ind+1

if kreuzer_in == True:
    df = pd.DataFrame({
        'x':x_supreme,
        'AS':profile_lists[0],
        'kreuzer': profile_lists[1],
        'realistic':profile_lists[2],
        'sample':profile_lists[3]
    })
    fig = px.line(df, x='x', y=['AS', 'kreuzer', 'realistic','sample'], title='Profile comparisson')
    fig.update_xaxes(title_text= 'Pixels [px]')
    fig.update_yaxes(title_text='Phase [rad]')
    fig.update_traces(line=dict(color='#6C0E5D'), selector=dict(mode='lines', name='AS'))
    fig.update_traces(line=dict(color='#E66C51'), selector=dict(mode='lines', name='kreuzer'))
    fig.update_traces(line=dict(color='#0061A8'), selector=dict(mode='lines', name='realistic'))
    fig.update_layout(legend=dict(title='Reconstruction\nmethod'))
    folder_name = r'F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Graphs\Phase'+NA+'profile.html'
    fig.write_html(folder_name)




else:
    df = pd.DataFrame({
        'x':x_supreme,
        'AS':profile_lists[0],
        'realistic':profile_lists[1],
        'sample':profile_lists[2]
    })
    fig = px.line(df, x='x', y=['AS','realistic','sample'], title='Profile comparisson')
    fig.update_xaxes(title_text= 'Pixels [px]')
    fig.update_yaxes(title_text='Phase [rad]')
    fig.update_traces(line=dict(color='#6C0E5D'), selector=dict(mode='lines', name='AS'))
    fig.update_traces(line=dict(color='#0061A8'), selector=dict(mode='lines', name='realistic'))
    fig.update_layout(legend=dict(title='Reconstruction\nmethod'))
    folder_name = r'F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Graphs\Phase'+NA+'profile.html'
    fig.write_html(folder_name)




