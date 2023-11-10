import pyLHM.myfunctions as LHM
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
sample_path = r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\USAF-sampled.png"
sample = LHM.open_image(sample_path)
profile_sample = sample[393:561,494]
x = [i for i in range(len(profile_sample))]



kreuzer_in = True
NA = r'\01'
folder = r'F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Recs\USAF'+ NA
files = get_files_in_folder(folder)

# props_names = [r'realistic',r'AS',r'kreuzer',r'SAASM']
# profiles = [493,493,483]

props_names = [r'AS',r'kreuzer',r'realistic']
if kreuzer_in == True:
    profiles = [493,483,493]
    profile_lists = [0,0,0,profile_sample]
    mtfs = [0,0,0]
    groups = [[(393,420),(433,455),(470,487),(504,518),(533,542),(557,561)],
          [(330,374),(390,427),(445,474),(500,521),(544,558),(580,585)],
          [(393,420),(433,455),(470,487),(504,518),(533,542),(557,561)]]
else:
    profiles = [493,493]
    profile_lists = [0,0,profile_sample]
    mtfs = [0,0]
    groups = [[(393,420),(433,455),(470,487),(504,518),(533,542),(557,561)],
          [(393,420),(433,455),(470,487),(504,518),(533,542),(557,561)]]


# groups = [[(393,420),(433,455),(470,487),(504,518),(533,542),(557,561)],
#           [(393,420),(433,455),(470,487),(504,518),(533,542),(557,561)],
#           [(330,374),(390,427),(445,474),(500,521),(544,558),(580,585)]]



max_px = 0
for file in files:
    if 'SAASM' in file:
        continue
    idx = files.index(file)
    I = LHM.open_image(file)
    mtfs[idx],_,profile_lists[idx] = metrics.measure_resolution(I,profiles[idx],groups[idx])
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
    fig.update_xaxes(title_text='Spatial Frequency [1/px]')
    fig.update_yaxes(title_text='Contrast')
    fig.update_layout(legend=dict(title='Reconstruction\nmethod'))
    folder_name = r'F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Graphs\Resolution\Profiles'+NA+'profile.html'
    fig.write_html(folder_name)

    x = [1/12, 1/10, 1/8, 1/6, 1/4, 1/2]
    df = pd.DataFrame({
        'x':x,
        'AS':mtfs[0],
        'kreuzer': mtfs[1],
        'realistic':mtfs[2],
    })

    fig = px.line(df, x='x', y=['AS', 'kreuzer', 'realistic'], title='Frequency Response')
    fig.update_layout(legend=dict(title='Reconstruction\nmethod'))
    fig.update_xaxes(title_text='Spatial Frequency [1/px]')
    fig.update_yaxes(title_text='Contrast')
    fig.update_traces(line=dict(color='#6C0E5D'), selector=dict(mode='lines', name='AS'))
    fig.update_traces(line=dict(color='#E66C51'), selector=dict(mode='lines', name='kreuzer'))
    fig.update_traces(line=dict(color='#0061A8'), selector=dict(mode='lines', name='realistic'))
    folder_name = r'F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Graphs\Resolution\MTF'+NA+'profile.html'
    fig.write_html(folder_name)
else:
    df = pd.DataFrame({
        'x':x_supreme,
        'AS':profile_lists[0],
        'realistic':profile_lists[1],
        'sample':profile_lists[2]
    })
    fig = px.line(df, x='x', y=['AS','realistic','sample'], title='Profile comparisson')
    fig.update_layout(legend=dict(title='Reconstruction method'))
    fig.update_xaxes(title_text='Spatial Frequency [1/px]')
    fig.update_yaxes(title_text='Contrast')
    folder_name = r'F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Graphs\Resolution\Profiles'+NA+'profile.html'
    fig.write_html(folder_name)

    x = [1/12, 1/10, 1/8, 1/6, 1/4, 1/2]
    df = pd.DataFrame({
        'x':x,
        'AS':mtfs[0],
        'realistic':mtfs[1],
    })

    fig = px.line(df, x='x', y=['AS', 'realistic'], title='Frequency Response')
    fig.update_layout(legend=dict(title='Reconstruction\nmethod'))
    fig.update_xaxes(title_text='Spatial Frequency [1/px]')
    fig.update_yaxes(title_text='Contrast')
    folder_name = r'F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Graphs\Resolution\MTF'+NA+'profile.html'
    fig.write_html(folder_name)


