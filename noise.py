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

def write_excel(mat,cols,file_path,sheetname):
    final_df = pd.DataFrame(mat,columns=cols)
    with pd.ExcelWriter(file_path,mode='a') as writer:
        final_df.to_excel(writer,sheet_name=sheetname,index=False,header=False)

metrics = LHM.metrics()

sample_selector = 1
samples_names = ['USAF',
                 'x pattern']
samples = ['USAF',
           'Grid']
ranges_usaf = [[540,623],[500,600]]
ranges_grid = [[404,632],[245,475]]
ranges = [ranges_usaf,ranges_grid]
ranges = ranges[sample_selector]

names = ['Angular Spectrum','Kreuzer','Convolutional Rayleigh']


noises = [[],[],[]]

folder = r'C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Recs'
files = get_files_in_folder(folder)

for file in files:
    if 'Fringes' in file:
        continue
    if samples[sample_selector] in file:
        I = LHM.open_image(file)
        local_noise = metrics.measure_noise(I,ranges[0],ranges[1])
        if 'AS' in file:
            noises[0].append(local_noise)
        elif 'kreuzer' in file:
            if '06' in file or '07' in file or '08' in file:
                local_noise = np.NaN
            noises[1].append(local_noise)
        elif 'RS1' in file:
            noises[2].append(local_noise)

df = pd.DataFrame({
    'AS': noises[0],
    'kreuzer': noises[1],
    'RS1':noises[2]
})


#----------------------------  Plot -------------------------------------
colors = ['#CEE719',
          '#4D94AD',
          '#E69FFF']

rgba = ['rgba'+str(hex_rgba(c, transparency=0.2)) for c in colors]
width = 1

fig = make_subplots(rows=1,cols=3)
fig.add_trace(go.Box(y=df['AS'],
                     name='Angular Spectrum',
                     line=dict(color=colors[0], width=width),
                     boxpoints='all',
                     boxmean='sd'),
                     row=1,col=1)
fig.add_trace(go.Box(y=df['kreuzer'],
                     name='Kreuzer',
                     line=dict(color=colors[1], width=width),
                     boxpoints='all',
                     boxmean='sd'),
                     row=1,col=2)
fig.add_trace(go.Box(y=df['RS1'],
                     name='Convolutional Rayleigh',
                     line=dict(color=colors[2], width=width),
                     boxpoints='all',
                     boxmean='sd'),
                     row=1,col=3)
fig.update_yaxes(title_text='Reconstruction time [s]',row=1, col=1)
fig.update_layout(legend=dict(title='Reconstruction\nmethod'),
                  showlegend=False,
                  title=dict(text='Noise in ' + samples_names[sample_selector] +' sample', font=dict(size=30), yref='paper'),
                  font_family='Arial',
                  font_color="black",
                  title_font_family="Arial",
                  title_font_color="black",
                  title_x=0.5
                  )                                 
fig.update_layout(
    font=dict(
        family="Arial",
        size=20,  # Set the font size here
        color="black"
    )
)
# fig.write_html(file_path)
config = {
  'toImageButtonOptions': {
    'format': 'svg', # one of png, svg, jpeg, webp
    'filename': 'rec_times',
    'height': 700,
    'width': 1024,
    'scale':100 # Multiply title/legend/axis/canvas sizes by this factor
  }
}


fig.show(config=config)


    
    



