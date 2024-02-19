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




metrics = LHM.metrics()
sample_path = r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\USAF-sampled.png"
sample = LHM.open_image(sample_path)
profile_sample = sample[393:561,494]
x = [i for i in range(len(profile_sample))]

profiles = 492
cord_file = r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\USAF_cords.xlsx"
sheets = ['01','02','03','04','05','06','07','08']
NA_arr = np.linspace(0.1,0.8,8)
kr_arr = [True,True,True,True,True,False,False,False]
showlegend = False
names = ['Angular Spectrum','Kreuzer','Convolutional Rayleigh']
fig = make_subplots(rows=3, cols=3)
color_palette =['#CEE719',
                '#4D94AD',
                '#E69FFF']

for index in range(len(kr_arr)):
    row = mt.floor(index/3)+1
    col = index%3 + 1
    NA_num = round(NA_arr[index],1)
    kreuzer_in = kr_arr[index]
    group_cords = pd.read_excel(cord_file,sheets[index])
    group_cords.describe()


    NAS = str(NA_num)
    NA = r'\ '+NAS.replace('.','')
    NA = NA.replace(' ','')
    folder = r'C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Recs\USAF'+ NA
    files = get_files_in_folder(folder)

    props_names = [r'AS',r'kreuzer',r'RS1']
    '''This lines allow to select specifically the columns in the dataframes
    that will be evaluated to pass to the function
    '''
    # keys = [key for key in min_coords.keys()]

    # if kreuzer_in == False:
    #     keys.remove('KR0')
    #     keys.remove('KR1')

    # min_coords_arr = min_coords[keys].to_numpy()
    # max_coords_arr = max_coords[keys].to_numpy()
    
    AS = [(group_cords['AS0'][i],group_cords['AS1'][i]) for i in range(6)]
    KR = [(group_cords['KR0'][i],group_cords['KR1'][i]) for i in range(6)]
    RS = [(group_cords['RS0'][i],group_cords['RS1'][i]) for i in range(6)]
    groups=[AS,
            KR,
            RS]






    mtfs = [0,0,0]
    stdv = np.zeros((6,3))
    profile_lists = [0,0,0,profile_sample]



    max_px = 0
    for file in files:
        idx = files.index(file)
        if kreuzer_in == False and 'kreuzer' in file:
            mtfs[idx] == np.NaN
            continue
        I = LHM.open_image(file)
        mtfs[idx],stdv[:,idx],profile_lists[idx] = metrics.measure_resolution(I,profiles,groups[idx])
        if len(profile_lists[idx])>max_px:
            max_px = len(profile_lists[idx])

    x_supreme = [i for i in range(max_px)]

    ind = 0
    # for profile in profile_lists:
    #     idx = profile.index(profile_lists)
    #     if kreuzer_in == False and 'kreuzer' in file:
    #         continue
        
    #     x = np.linspace(0,max_px,len(profile))
    #     inter = interp1d(x, profile, kind='linear')
    #     profile_lists[ind] = inter(x_supreme)
    #     ind = ind+1




    x = [1/12, 1/10, 1/8, 1/6, 1/4, 1/2]
    x = [i/(7.32e-7 * 10**6) for i in x]
    df = pd.DataFrame({
        'x':x,
        'AS':mtfs[0],
        'kreuzer': mtfs[1],
        'RS1':mtfs[2],
    })

    

    color_palette =['#CEE719',
                    '#4D94AD',
                    '#E69FFF']

    rgba = ['rgba'+str(hex_rgba(c, transparency=0.3)) for c in color_palette]

    for i, column in enumerate(df):
        if column=='x':
            continue
        if kreuzer_in==False and column=='kreuzer':
            continue
        x = list(df['x'])
        y1 = df[column]
        y1_upper = list(y1+stdv[:,i-1])
        y1_lower = list(y1-stdv[:,i-1])
        y1_lower = y1_lower[::-1]
        fig.add_trace(go.Scatter(x=x,
                                y=y1,
                                line=dict(color=color_palette[i-1], width=2.5),
                                mode='lines',
                                name=names[i-1]),
                                row=row,col=col)
        
        fig.add_trace(go.Scatter(x=x+x[::-1],
                                    y=y1_upper+y1_lower,
                                    fill='tozerox',
                                    fillcolor=rgba[i-1],
                                    line=dict(color=color_palette[i-1],width=0),
                                    marker=dict(opacity=0),
                                    showlegend=True,
                                    name=names[i-1]+' std'),
                                    row=row,col=col)
        fig.update_xaxes(linecolor='black',gridcolor='lightgrey', range=[x[0],x[5]], mirror=True,row=row,col=col)
        fig.update_yaxes(linecolor='black',gridcolor='lightgrey',range=[-0.01,1.01], mirror=True,row=row,col=col)
        # fig.update_layout(title=dict(text=NAS),title_x=0.5,row=row,col=col)
        
fig.update_xaxes(title_text= 'Spatial Frequency [1/\u03bcm]',row=3,col=1)
fig.update_xaxes(title_text= 'Spatial Frequency [1/\u03bcm]',row=3,col=2)
fig.update_xaxes(title_text= 'Spatial Frequency [1/\u03bcm]',row=2,col=3)
fig.update_yaxes(title_text='Contrast',row=1,col=1)
fig.update_yaxes(title_text='Contrast',row=2,col=1)
fig.update_yaxes(title_text='Contrast',row=3,col=1)
        

fig.update_layout(legend=dict(title='Reconstruction\nmethod'), showlegend=showlegend,plot_bgcolor='white',title_x=0.5)
fig.update_layout(
                title=dict(text='Frequency responses at different NA', font=dict(size=40), yref='paper'),
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
    'filename': 'MTFS_graph',
    'height': 1400,
    'width': 2048,
    'scale':50 # Multiply title/legend/axis/canvas sizes by this factor
  }
}

fig.show(config=config)
# folder_name = r'C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Graphs\Resolution\MTF'+NA+'profile.html'
# fig.write_html(folder_name)



