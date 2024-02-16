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

def smoothen(arr, winsize=5):
    max_px = len(arr)
    x_supreme = [i for i in range(max_px)]
    arr = np.array(pd.Series(arr).rolling(winsize).median())[winsize-1:]
    x = np.linspace(0,max_px,len(arr))
    inter = interp1d(x, arr, kind='linear')
    arr = inter(x_supreme)
    return arr



metrics = LHM.metrics()
sample_path = r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Samples\Fringes.png"
sample = LHM.open_image(sample_path)
profile_sample = 2 * np.pi * sample[844,280:734]
x = [i for i in range(len(profile_sample))]




showlegend = False

Numerical_appertures = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8']
kr_arr = [True,True,True,True,False,False,False,False,]
fig = make_subplots(rows=3, cols=3)
error_lists = [[],[],[]]
for i in range(len(kr_arr)):
    row = mt.floor(i/3)+1
    col = i%3 + 1
    NAS = Numerical_appertures[i]
    kreuzer_in = kr_arr[i]

    NA = r'\ '+NAS.replace('.','')
    NA = NA.replace(' ','')
    folder = r"C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Recs\Fringes"+ NA
    files = get_files_in_folder(folder)



    props_names = [r'AS',r'kreuzer',r'RS1']

    profile_lists = [0,0,0,profile_sample]
    smooth_profiles = [0,0,0]
    steps_lists = [0,0,0]
    

    indexing = [[844,280,734],
                [886,250,704],
                [844,280,734]]
    




    max_px = len(profile_sample)

    for file in files:
        idx = files.index(file)
        I = LHM.open_image(file)
        bg = np.mean(I[512:640,128:256])
        profile_lists[idx] = 2*np.pi*(I[indexing[idx][0],indexing[idx][1]:indexing[idx][2]]-bg)
        if len(profile_lists[idx])>max_px:
            max_px = len(profile_lists[idx])
        smooth_profiles[idx] = profile_lists[idx]
        # smooth_profiles[idx] = smoothen(profile_lists[idx],7)   #errase this line to avoid filtering

    # Calculations of the mean values on the steps
    for i in range(3):
        error_instance = np.mean(np.abs(profile_lists[i]-profile_lists[3]))
        if i==1 and kreuzer_in==False:
            continue
        error_lists[i].append(error_instance)


    errors = [np.amax(i) for i in error_lists]
    ind = 0

    # for profile in smooth_profiles:
    #     smooth_profiles[ind] = profile[129:675]

    #     ind = ind+1
    #     max_px = 546



    x_supreme = [(7.32e-4 *(i-int(max_px/2))) for i in range(max_px)]


    color_palette =['#000000',
                '#CEE719',
                '#4D94AD',
                '#E69FFF']

    show_raw = True
    transparency = 0.75
    rgba = ['rgba'+str(hex_rgba(c, transparency=transparency)) for c in color_palette]
    rgba = rgba + ['rgba'+str(hex_rgba(c, transparency=transparency-0.5)) for c in color_palette]

    ind = 0
    # for profile in profile_lists:
    #     x = np.linspace(0,max_px,len(profile))
    #     inter = interp1d(x, profile, kind='linear')
    #     profile_lists[ind] = inter(x_supreme)
    #     ind = ind+1

    if kreuzer_in == True:
        df = pd.DataFrame({
            'x':x_supreme,
            'Sample':profile_lists[3],
            'SAS':smooth_profiles[0],
            'SKR':smooth_profiles[1],
            'SRS':smooth_profiles[2],
        })
        fig.add_trace(go.Scatter(x=df['x'],y=df['Sample'],
                                 mode='lines',
                                 line=dict(color=rgba[0],dash='dash')
                                 ),row=row,col=col)
        fig.add_trace(go.Scatter(x=df['x'],y=df['SAS'],
                                 mode='lines',
                                 line=dict(color=rgba[1])
                                 ),row=row,col=col)
        fig.add_trace(go.Scatter(x=df['x'],y=df['SKR'],
                                 mode='lines',
                                 line=dict(color=rgba[2])
                                 ),row=row,col=col)
        fig.add_trace(go.Scatter(x=df['x'],y=df['SRS'],
                                 mode='lines',
                                 line=dict(color=rgba[3])
                                 ),row=row,col=col)
        # fig = px.line(df, x='x', y=['Sample','SAS','SKR','SRS'], title='Profile comparison')
        # fig.update_traces(line=dict(color=rgba[0],dash='dash'), selector=dict(mode='lines', name='Sample'))
        # fig.update_traces(line=dict(color=rgba[1]), selector=dict(mode='lines', name='SAS'))
        # fig.update_traces(line=dict(color=rgba[2]), selector=dict(mode='lines', name='SKR'))
        # fig.update_traces(line=dict(color=rgba[3]), selector=dict(mode='lines', name='SRS'))

    else:
        df = pd.DataFrame({
            'x':x_supreme,
            'Sample':profile_lists[3],
            'SAS':smooth_profiles[0],
            'SKR':smooth_profiles[1],
            'SRS':smooth_profiles[2],
        })

        fig.add_trace(go.Scatter(x=df['x'],y=df['Sample'],
                                 mode='lines',
                                 line=dict(color=rgba[0],dash='dash',width=2)
                                 ),row=row,col=col)
        fig.add_trace(go.Scatter(x=df['x'],y=df['SAS'],
                                 mode='lines',
                                 line=dict(color=rgba[1])
                                 ),row=row,col=col)
        fig.add_trace(go.Scatter(x=df['x'],y=df['SRS'],
                                 mode='lines',
                                 line=dict(color=rgba[3])
                                 ),row=row,col=col)
    fig.update_xaxes(linecolor='black',gridcolor='lightgrey', mirror=True,row=row,col=col)
    fig.update_yaxes(linecolor='black',gridcolor='lightgrey', mirror=True,row=row,col=col)


fig.update_xaxes(title_text= 'Radius [mm]',row=3,col=1)
fig.update_xaxes(title_text= 'Radius [mm]',row=3,col=2)
fig.update_xaxes(title_text= 'Radius [mm]',row=2,col=3)
fig.update_yaxes(title_text='Phase [rad]',row=1,col=1)
fig.update_yaxes(title_text='Phase [rad]',row=2,col=1)
fig.update_yaxes(title_text='Phase [rad]',row=3,col=1)

fig.update_layout(showlegend=showlegend,plot_bgcolor='white')
fig.update_layout(
                title=dict(text='Phase ladder profile comparison', font=dict(size=30), yref='paper'),
                title_x=0.5,
                font_family='Arial',
                font_color="black",
                title_font_family="Arial",
                title_font_color="black",
                )                                 
fig.update_layout(
font=dict(
    family="Arial",
    size=20,  # Set the font size here
    color="black"
)
)

# print(errors)
config = {
  'toImageButtonOptions': {
    'format': 'svg', # one of png, svg, jpeg, webp
    'filename': 'phase_error',
    'width': 1024,
    'height': 640,
    'scale' : 10
  }
}


# fig.show(config=config)

folder_name = r'C:\Users\tom_p\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\Graphs\Phase_mean_error.html'
NA_plot = [0.1,0.2,0.3,0.4,0.4,0.6,0.7,0.8]
AS_reg = [0.5827*i+0.0375 for i in NA_plot]
kr_reg = [0.4545*i+0.1129 for i in NA_plot]
rs_reg = [0.4902*i+0.0987 for i in NA_plot]
fig2 = go.Figure(data=go.Scatter(x=NA_plot,y=error_lists[0],
                 mode='markers',
                 marker_line_width=1,
                 marker=dict(color=rgba[1])))
fig2.add_trace(go.Scatter(x=NA_plot[0:4],y=error_lists[1],
                 mode='markers',
                 marker_line_width=1,
                 marker=dict(color=rgba[2])))
fig2.add_trace(go.Scatter(x=NA_plot,y=error_lists[2],
                 mode='markers',
                 marker_line_width=1,
                 marker=dict(color=rgba[3])))

fig2.add_trace(go.Scatter(x=NA_plot,y=AS_reg,
                 mode='lines',
                 line=dict(color=rgba[1],dash='dash')))
fig2.add_trace(go.Scatter(x=NA_plot,y=kr_reg,
                 mode='lines',
                 line=dict(color=rgba[2])))
fig2.add_trace(go.Scatter(x=NA_plot,y=rs_reg,
                 mode='lines',
                 line=dict(color=rgba[3])))


fig2.update_xaxes(title_text= 'NA',linecolor='black',gridcolor='lightgrey', mirror=True)
fig2.update_yaxes(title_text='Phase error [rad]',linecolor='black',gridcolor='lightgrey', mirror=True)



fig2.update_layout(showlegend=False,plot_bgcolor='white')
fig2.update_layout(
                title=dict(text='Phase mean error', font=dict(size=30), yref='paper'),
                title_x=0.5,
                font_family='Arial',
                font_color="black",
                title_font_family="Arial",
                title_font_color="black",
                )                                 
fig2.update_layout(
font=dict(
    family="Arial",
    size=20,  # Set the font size here
    color="black"
)
)
fig2.show(config=config)
# fig.write_html(folder_name)




