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

def hex_rgba(hex, transparency):
    col_hex = hex.lstrip('#')
    col_rgb = list(int(col_hex[i:i+2], 16) for i in (0, 2, 4))
    col_rgb.extend([transparency])
    areacol = tuple(col_rgb)
    return areacol


metrics = LHM.metrics()
sample_path = r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\USAF-sampled.png"
sample = LHM.open_image(sample_path)
profile_sample = sample[393:561,494]
x = [i for i in range(len(profile_sample))]


showlegend=False
kreuzer_in = False
NAS = '0.8'
NA = r'\ '+NAS.replace('.','')
NA = NA.replace(' ','')
index = 0 # Only change this (on increments of 1) for NA>=0.6 if kreuzer is in the graph
folder = r'C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Recs\USAF'+ NA
files = get_files_in_folder(folder)

props_names = [r'AS',r'kreuzer',r'RS1']
names = ['Angular Spectrum','Kreuzer','Rayleigh']
profiles = [493,483,493]




kreuzer_groups = [[(373,409),(420,450),(460,483),(502,519),(534,547),(560,566)],
                  [(340,382),(398,433),(448,477),(500,522),(542,557),(577,583)],
                  [(335,378),(395,432),(448,475),(500,522),(542,557),(577,584)],
                  [(331,375),(393,430),(445,475),(499,522),(543,559),(577,585)]]


mtfs = [0,0,0]
stdv = np.zeros((6,3))
profile_lists = [0,0,0,profile_sample]
groups = [[(391,420),(432,457),(467,487),(504,518),(533,542),(556,561)],
          kreuzer_groups[index],
          [(393,421),(432,457),(467,487),(504,518),(532,542),(555,561)]]

# groups = [[(391,420),(432,457),(467,487),(504,518),(533,542),(556,561)],
#           [(393,420),(433,455),(470,487),(504,518),(533,542),(557,561)],
#           [(393,421),(432,457),(467,487),(504,518),(532,542),(555,561)]]


max_px = 0
for file in files:
    idx = files.index(file)
    I = LHM.open_image(file)
    mtfs[idx],stdv[:,idx],profile_lists[idx] = metrics.measure_resolution(I,profiles[idx],groups[idx])
    if len(profile_lists[idx])>max_px:
        max_px = len(profile_lists[idx])

x_supreme = [i for i in range(max_px)]

ind = 0
for profile in profile_lists:
    x = np.linspace(0,max_px,len(profile))
    inter = interp1d(x, profile, kind='linear')
    profile_lists[ind] = inter(x_supreme)
    ind = ind+1


# df = pd.DataFrame({
#     'x':x_supreme,
#     'AS':profile_lists[0],
#     'kreuzer': profile_lists[1],
#     'RS1':profile_lists[2],
#     'sample':profile_lists[3]
# })
# fig = px.line(df, x='x', y=['AS', 'kreuzer', 'realistic','sample'], title='Profile comparisson')
# fig.update_xaxes(title_text='Spatial Frequency [1/px]')
# fig.update_yaxes(title_text='Contrast')
# fig.update_layout(legend=dict(title='Reconstruction\nmethod'))
# folder_name = r'F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Graphs\Resolution\Profiles'+NA+'profile.html'
# fig.write_html(folder_name)

x = [1/12, 1/10, 1/8, 1/6, 1/4, 1/2]
x = [i/(7.32e-7 * 10**6) for i in x]
df = pd.DataFrame({
    'x':x,
    'AS':mtfs[0],
    'kreuzer': mtfs[1],
    'RS1':mtfs[2],
})

fig = go.Figure()

color_palette =['#CEE719',
                '#4D94AD',
                '#E69FFF']

rgba = ['rgba'+str(hex_rgba(c, transparency=0.3)) for c in color_palette]

for i, col in enumerate(df):
    if col=='x':
        continue
    if kreuzer_in==False and col=='kreuzer':
        continue
    x = list(df['x'])
    y1 = df[col]
    y1_upper = list(y1+stdv[:,i-1])
    y1_lower = list(y1-stdv[:,i-1])
    y1_lower = y1_lower[::-1]
    fig.add_traces(go.Scatter(x=x,
                              y=y1,
                              line=dict(color=color_palette[i-1], width=2.5),
                              mode='lines',
                              name=names[i-1])
                                )
    
    fig.add_traces(go.Scatter(x=x+x[::-1],
                                y=y1_upper+y1_lower,
                                fill='tozerox',
                                fillcolor=rgba[i-1],
                                line=dict(color=color_palette[i-1],width=0),
                                marker=dict(opacity=0),
                                showlegend=True,
                                name=names[i-1]+' std'))
    
    
    
fig.update_xaxes(title_text= 'Spatial Frequency [1/\u03bcm]',range=[x[0],x[5]],linecolor='black',gridcolor='lightgrey',mirror=True)
fig.update_yaxes(title_text='Contrast',range=[-0.01,1.01],linecolor='black',gridcolor='lightgrey', mirror=True)
fig.update_layout(legend=dict(title='Reconstruction\nmethod'), showlegend=showlegend,plot_bgcolor='white',title_x=0.5)
fig.update_layout(
                title=dict(text='Frequency responses at NA '+NAS, font=dict(size=40), yref='paper'),
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

folder_name = r'C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Graphs\Resolution\MTF'+NA+'profile.html'
fig.write_html(folder_name)



