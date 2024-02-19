# August 2023
# Library for DLHM, In-line holography
# Authors: Tomas Velez, Maria J Lopera
# Kreuzer taken from Maria J Lopera (https://github.com/mloper23/DLHM-backend), angular spectrum taken from C Trujillo (https://github.com/catrujilla/pyDHM)
# Autofocus based on Maria J Lopera (https://github.com/mloper23/DLHM-backend)

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
from scipy.signal import convolve2d
import scipy.signal as scp
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import pandas as pd
import skimage as skm
import xarray as xr
import cv2 as cv2
import matplotlib.pyplot as plt

class focus:
    def __init__(self,mode=None ):
        if mode == 'amp' or mode == None:
            self.mode = 'amplitude'
        elif mode=='phase':
            self.mode = 'phase'

    def manual_focus(self, propagator_name, parameters, z1, z2, it):
        delta = (z2 - z1) / it
        M,N = np.shape(parameters[0])
        tensor = np.zeros((int(M/2),int(N/2),it))
        # tensor = np.zeros((M-1, N-1, it))
        propagation = reconstruct()
        zs = np.linspace(z1, z2, it)
        for i in range(0, it):
            params = [z1 + i * delta] + parameters
            hz = propagation.autocall(propagator_name, params)

            if self.mode =='amplitude':
                amp = np.abs(hz)**2
                if np.shape(amp)[0]>1024:
                    amp = cv2.resize(amp,(1023,1023))
                img = amp[int(M/4):int(M/2 + (M/4)),int(N/4):int(N/2 + (N/4))]
                # img = amp
            elif self.mode == 'phase':
                phase = np.angle(hz)
                if np.shape(phase)[0]>1024:
                    phase = cv2.resize(amp,(1023,1023))
                img = phase[int(M/4):int(M/2 + (M/4)),int(N/4):int(N/2 + (N/4))]
                # img = phase
            print(i+1)
            # tensor[:,:,i] = np.resize(np.abs(hz)**2,(int(M/2),int(N/2)))
            tensor[:,:,i] = img
        data =xr.DataArray(tensor,
                           dims=("Height","Width","z"),
                           coords={"z":np.round(zs*1000,decimals=3)})
        fig = px.imshow(data,animation_frame='z',color_continuous_scale='gray')
        fig.update_yaxes(scaleanchor='x', constrain='domain')
        fig.update_xaxes(constrain='domain')
        # fig.show()
        fig.write_html(r'images\manual_focus.html')
        return data    

    def autofocus(self,propagator_name,parameters,z1,z2,it):
        delta = (z2-z1)/it
        focus_metric = np.zeros((3,it))
        propagation = reconstruct()
        for i in range(0,it):
            params = [z1 + i*delta] + parameters
            hz = propagation.autocall(propagator_name,params)
            m1 = np.sum(np.abs(hz))
            m2 = np.abs(np.sum(np.log(1+np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(np.abs(np.abs(hz))))))))
            m4 = np.sqrt(np.std(np.abs(hz)**2)/(np.mean(np.abs(hz))))
            metric_local = np.transpose(np.array([m1, m2, m4]))
            focus_metric[:,i] = metric_local
        focus_metric = self.normrows(focus_metric)
        min = np.argmin(focus_metric)
        d = z1 + min * delta
        return d
    
    def normrows(self,array):
        rows = np.shape(array)[0]
        for row in range(rows):
            col = array[row,:]-np.amin(array[row,:])
            col = col/np.amax(col)
            array[row,:] = col
        return array

class reconstruct:

    def autocall(self,propagator,parameters):
        self.propagator = propagator
        self.param = parameters
        if callable(getattr(self, propagator, None)):
            # print(f"'{propagator}' is a valid propagator!")
            propagator = getattr(self,propagator,None)
            sol = propagator(*self.param)
            return sol
            # try:
                
            # except:
            #     raise ValueError(f"The parameters '{parameters}' do not correspond to '{propagator}' ")

        else:
            raise ValueError(f"'{propagator}'? That's not a valid propagator method.")
        
        pass

    def realistic_rec_DLHM(self, z, field, wavelength, L, w_c, pinhole, it_ph, bits):
        N, M = field.shape
        mag = L / z
        w_s = w_c / mag
        rad_ph = np.round(pinhole * N / w_s)
        th = np.linspace(0, 2 * np.pi, it_ph)
        x_unit = np.round(rad_ph / 2 * np.cos(np.deg2rad(th)))
        y_unit = np.round(rad_ph / 2 * np.sin(np.deg2rad(th)))

        NA = (w_c / 2) / (np.sqrt(w_c / 2) ** 2 + L ** 2)

        holo = np.zeros([N, M])
        ps = holo.copy()

        # sample_M = np.array(cv.resize(sample, [round(N * mag), round(M * mag)])).astype('float')
        x = np.linspace(-w_c / 2, w_c / 2, N)
        y = np.linspace(-w_c / 2, w_c / 2, M)
        u, v = np.meshgrid(x, y)
        r = np.sqrt(u ** 2 + v ** 2 + (L-z) ** 2)
        df = 1/w_s
        dfix = np.linspace(-N/2 * df, N/2 * df, M)
        dfiy = np.linspace(-M/2 * df, M/2 * df, M)
        fx, fy = np.meshgrid(dfix, dfiy)

        for i in range(0, it_ph):
            # sample_ = sample_M[round(N * mag / 2 - N / 2 - y_unit[i]):round(N * mag / 2 + N / 2 - y_unit[i]),
            #           round(M * mag / 2 - M / 2 - x_unit[i]):round(M * mag / 2 + M / 2 - x_unit[i])]
            sample_ = field
            Ui = self.AS(sample_, 2*np.pi/wavelength, fx, fy, L-z)
            ps_ = self.point_src(M, L, x_unit[i] * w_s / N, y_unit[i] * w_s /N, wavelength, w_c/N)
            ps = ps + ps_
            holo = holo + Ui
        
        Utot = holo.copy()
        amp = np.abs(Utot)
        phase = np.angle(Utot)
        holo = np.abs(holo) ** 2

        camMat = np.array([[N, 0, N/2], [0, M, M/2], [0, 0, 1]])
        dist = np.array([NA*0.1, 0, 0, 0])
        amp = cv2.undistort(amp, camMat, dist)
        phase = cv2.undistort(amp, camMat, dist)
        Utot = amp*np.exp(1j*phase)

        holo = holo * (1 - r/r.max())
        holo = self.norm_bits(holo, bits)

        

        return Utot

    def realisticDLHM(self, field, wavelength, L, z, w_c, pinhole, it_ph, bits):
        N, M = field.shape
        mag = L / z
        w_s = w_c / mag
        rad_ph = np.round(pinhole * N / w_s)
        th = np.linspace(0, 2 * np.pi, it_ph)
        x_unit = np.round(rad_ph / 2 * np.cos(np.deg2rad(th)))
        y_unit = np.round(rad_ph / 2 * np.sin(np.deg2rad(th)))

        NA = (w_c / 2) / (np.sqrt(w_c / 2) ** 2 + L ** 2)

        holo = np.zeros([N, M])
        ps = holo.copy()

        # sample_M = np.array(cv.resize(sample, [round(N * mag), round(M * mag)])).astype('float')
        x = np.linspace(-w_c / 2, w_c / 2, N)
        y = np.linspace(-w_c / 2, w_c / 2, M)
        u, v = np.meshgrid(x, y)
        r = np.sqrt(u ** 2 + v ** 2 + (L-z) ** 2)
        df = 1/w_s
        dfix = np.linspace(-N/2 * df, N/2 * df, M)
        dfiy = np.linspace(-M/2 * df, M/2 * df, M)
        fx, fy = np.meshgrid(dfix, dfiy)

        for i in range(0, it_ph):
            # sample_ = sample_M[round(N * mag / 2 - N / 2 - y_unit[i]):round(N * mag / 2 + N / 2 - y_unit[i]),
            #           round(M * mag / 2 - M / 2 - x_unit[i]):round(M * mag / 2 + M / 2 - x_unit[i])]
            sample_ = field
            Ui = self.AS(sample_, 2*np.pi/wavelength, fx, fy, L-z)
            ps_ = self.point_src(M, L, x_unit[i] * w_s / N, y_unit[i] * w_s /N, wavelength, w_c/N)
            ps = ps + ps_
            holo = holo + Ui
        
        
        holo = np.abs(holo) ** 2
        camMat = np.array([[N, 0, N/2], [0, M, M/2], [0, 0, 1]])
        max_dist = np.abs((L + np.abs(np.sqrt(w_c ** 2 / 2 + L ** 2) - L)) / z - L / z)
        dist = np.array([-max_dist, 0, 0, 0])
        holo = cv2.undistort(holo, camMat, dist)

        holo = holo * (1 - r/r.max())
        holo = self.norm_bits(holo, bits)

        ref = np.abs(1-r/r.max()) ** 2
        ref = self.norm_bits(ref, bits)

        return holo, ref

    def rayleigh1Free(self, z, field, wavelength, pixel_pitch_in, pixel_pitch_out, out_shape):
        '''
        Function to cumpute the Raleygh Sommerfeld 1 diffraction integral wothout approximations or the use of FFT,
        but allowing to change the output sampling (pixel pitch and shape).
        ### Inputs: 
        * field - complex field to be diffracted
        * z - propagation distance
        * wavelength - wavelength of the light used
        * pixel_pitch_in - Sampling pitches of the input field as a (2,) list
        * pixel_pitch_out - Sampling pitches of the output field as a (2,) list
        * Output_shape - Shape of the output field as an tuple of integers
        '''

        dx = pixel_pitch_in[0]  # Input Pixel Size X
        dy = pixel_pitch_in[1]  # Input Pixel Size Y
        ds = dx * dy
        dx_out = pixel_pitch_out[0]  # Output Pixel Size X
        dy_out = pixel_pitch_out[1]  # Output Pixel Size Y

        M, N = np.shape(field)
        (M2, N2) = out_shape
        k = (2 * np.pi) / wavelength  # Wave number of the ilumination source

        U0 = np.zeros(out_shape, dtype='complex_')
        U1 = field  # This will be the hologram plane

        x_inp_lim = dx * int(N / 2)
        y_inp_lim = dy * int(M / 2)

        x_cord = np.linspace(-x_inp_lim, x_inp_lim, num=N)
        y_cord = np.linspace(-y_inp_lim, y_inp_lim, num=M)

        [X_inp, Y_inp] = np.meshgrid(x_cord, y_cord, indexing='xy')

        x_out_lim = dx_out * int(N2 / 2)
        y_out_lim = dy_out * int(M2 / 2)

        x_cord_out = np.linspace(-x_out_lim, x_out_lim, num=N2)
        y_cord_out = np.linspace(-y_out_lim, y_out_lim, num=M2)

        # The first pair of loops ranges over the points in the output plane in order to determine r01
        for x_sample in range(out_shape[0]):
            x_fis_out = x_cord_out[x_sample]
            for y_sample in range(out_shape[1]):
                # start = time.time()
                y_fis_out = y_cord_out[y_sample]
                mr01 = np.sqrt(np.power(x_fis_out - X_inp, 2) + np.power(y_fis_out - Y_inp, 2) + (z) ** 2)
                Obliquity = (z) / mr01
                kernel = np.exp(1j * k * mr01) / mr01
                dif = (1j * k) + (1 / mr01)
                U0[y_sample, x_sample] = np.sum(U1 * dif * kernel * Obliquity * ds)
                # stop = time.time()
                # print('Tiempo de ejecución: ', 1000*(stop-start))
        U0 = -U0 / (2 * np.pi)
        Viewing_window = [-x_out_lim, x_out_lim, -y_out_lim, y_out_lim]
        return U0

    def convergentSAASM(self, z_micro, field, wavelength, pixel_pitch_in, pixel_pitch_out,L):
        '''
        Function to diffract a complex field using the angular spectrum approach with a Semi-Analytical spherical wavefront.
        This operator only works for convergent fields.
        For further reference review: https://opg.optica.org/josaa/abstract.cfm?uri=josaa-31-3-591 and https://doi.org/10.1117/12.2642760
        
        ### Inputs:
        * field - complex field to be diffracted
        * z - propagation distance
        * wavelength - wavelength of the light used
        * pixel_pitch_in - Sampling pitches of the input field as a (2,) list
        * pixel_pitch_out - Sampling pitches of the output field as a (2,) list
        '''
        z = L-z_micro
        # Starting cooridnates computation
        k_wl = 2 * np.pi / wavelength
        M, N = field.shape
        # Linear Coordinates
        x = np.arange(0, N, 1)  # array x
        fx = np.fft.fftshift(np.fft.fftfreq(N, pixel_pitch_in[0]))
        y = np.arange(0, M, 1)  # array y
        fy = np.fft.fftshift(np.fft.fftfreq(M, pixel_pitch_in[1]))
        # Grids
        X_in, Y_in = np.meshgrid((x - (N / 2)) * pixel_pitch_in[0], (y - (M / 2)) * pixel_pitch_in[1], indexing='xy')
        FX, FY = np.meshgrid(fx, fy, indexing='xy')
        KX = FX * 2 * np.pi
        KY = FY * 2 * np.pi
        MR_in = (X_in ** 2 + Y_in ** 2)
        MK = np.sqrt(KX ** 2 + KY ** 2)
        kmax = np.abs(np.amax(MK))

        ''' IN THIS STEP THE FIRST FOURIER TRANSFORM OF THE FIELD IS CALCULATED DOING A RESAMPLING USING THE
        FAST FOURIER TRASNSFORM AND A PADDING. THIS TRANSFORM HAS AS OUTPUT COORDINATE THE SCALED COORDINATE
        BETA, THAT IS NOT RELEVANT FOR THIS STEP BUT THE NEXT ONE'''
        # Fitting parameters for the parabolic fase
        k_interm = (k_wl / kmax)
        c = (2 / 3 * k_interm) + 2 / 3 * np.sqrt(k_interm ** 2 - 0.5) - 1 / 3 * np.sqrt(k_interm ** 2 - 1)
        d = np.sqrt(k_interm ** 2 - 1) - k_interm
        pp0 = pixel_pitch_in[0]

        # Initial interpolation for j=1
        max_grad_alpha = -kmax / (2 * d * z) * np.amax(MR_in)
        alpha = (np.exp(-1j * c * kmax * z) * kmax / (2j * d * z)) * np.exp((1j * kmax * MR_in) / (4 * d * z))

        # Interpolation of the input field Scipy

        N2 = int(N * (2 + max_grad_alpha * pp0 / np.pi))
        M2 = int(M * (2 + max_grad_alpha * pp0 / np.pi))
        # M2 = 4*M
        # N2 = 4*N

        pp1 = M * pixel_pitch_in[0] / M2
        x1 = np.arange(0, N2 - 1, 1)
        y1 = np.arange(0, M2 - 1, 1)
        X1, Y1 = np.meshgrid((x1 - (N2 / 2)) * pp1, (y1 - (M2 / 2)) * pp1, indexing='ij')
        fx1 = np.fft.fftshift(np.fft.fftfreq(N2, pp1))
        fy1 = np.fft.fftshift(np.fft.fftfreq(M2, pp1))
        FX1, FY1 = np.meshgrid(fx1, fy1, indexing='xy')
        # THIS LINEs ARE FOR TRIALS ONLY
        X1 = Y_in
        Y1 = X_in
        FX1 = FX
        FY1 = FY
        # _______________________________
        KX1 = FX1 * 2 * np.pi
        KY1 = FY1 * 2 * np.pi
        MK1 = np.sqrt(KX1 ** 2 + KY1 ** 2)
        kmax = np.abs(np.amax(MK1))

        xin = (x - (N / 2)) * pp0
        yin = (y - (M / 2)) * pp0
        inter = RegularGridInterpolator((xin, yin), field, bounds_error=False, fill_value=None)
        E_interpolated = inter((X1, Y1))
        # E_interpolated = field

        MR1 = (X1 ** 2 + Y1 ** 2)
        k_interm = (k_wl / kmax)
        c = (2 / 3 * k_interm) + 2 / 3 * np.sqrt(k_interm ** 2 - 0.5) - 1 / 3 * np.sqrt(k_interm ** 2 - 1)
        d = np.sqrt(k_interm ** 2 - 1) - k_interm

        alpha = np.exp(-1j * c * kmax * z) * kmax / (2j * d * z) * np.exp((1j * kmax * MR1) / (4 * d * z))
        E_interpolated = E_interpolated - np.amin(E_interpolated)
        E_interpolated = E_interpolated / np.amax(E_interpolated)
        EM1 = np.divide(E_interpolated, alpha)

        # Padding variables for j=2
        # max_grad_kernel = np.amax(Mbeta)

        # Computation of the j=1 step
        FE1 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(EM1)))

        '''IN THIS STEP THE SECOND FOURIER TRANSFORM IS CALCULATED. HERE THE COORDINATES BETA ARE RELEVANT
        SINCE THE ELEMENT-WISE PRODUCT OF THE FE1 WITH THE PROPAGATION KERNEL REQUIRES THE KERNEL'S 
        ARGUMENT TO BE THE MAGNITUDE OF BETA INSTEAD OF THE MAGNITUD OF RHO'''
        # Calculation of the oversampled kernel
        M0, N0 = np.shape(FE1)
        x2 = np.arange(0, N0, 1)
        y2 = np.arange(0, M0, 1)
        # If required, check the pixel size
        # X_out, Y_out = np.meshgrid((x2 - (N0 / 2))*pp1, (y2- (M0 / 2))*pp1, indexing='xy')#<----------------------erase this
        X_out, Y_out = np.meshgrid((x2 - (N0 / 2)) * pixel_pitch_out[0], (y2 - (M0 / 2)) * pixel_pitch_out[1],
                                indexing='xy')
        Mrho = np.sqrt(np.power(X_out, 2) + np.power(Y_out, 2))
        bX = -kmax * X_out / (2 * d * z)
        bY = -kmax * Y_out / (2 * d * z)
        Mbeta = np.sqrt(np.power(bX, 2) + np.power(bY, 2))
        kernel = np.exp(-1j * d * z * np.power(Mbeta, 2) / (kmax))
        # kernel = np.exp(-1j * kmax * np.power(Mrho,2)/(4 * d * z))
        EM2 = FE1 * kernel

        # Computation of the j=2 step
        FE2 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(EM2)))
        # half_size2 = [int(np.shape(FE2)[0]/2),int(np.shape(FE2)[1]/2)]
        # FE2 = FE2[half_size2[0]-int(M/2):half_size2[0]+int(M/2),half_size2[1]-int(N/2):half_size2[1]+int(N/2)]

        '''IN THIS STEP THE THIRD FOURIER TRANSFORM IS CALCULATED. HERE THE SUPERIOR ORDER TERMS (H) ARE CALCULATED
        TO FIND NUMERICALLY THE MAXIMUM GRADIENT OF ITS ARGUMENT, THEN, A PADDING OF FE2 IS DONE AND FINALLY H
        IS RESAMPLED IN TERMS OF FE2'''
        # Calculation of the superior order phases
        Mfin, Nfin = np.shape(FE2)
        # ----------------ERASE THIS-----------------
        # fx_out = np.fft.fftshift(np.fft.fftfreq(Nfin,pp1))
        # fy_out = np.fft.fftshift(np.fft.fftfreq(Mfin,pp1))
        # -------------------------------------------

        fx_out = np.fft.fftshift(np.fft.fftfreq(Nfin, pixel_pitch_out[0]))
        fy_out = np.fft.fftshift(np.fft.fftfreq(Mfin, pixel_pitch_out[1]))
        FX_out, FY_out = np.meshgrid(fx_out, fy_out, indexing='xy')
        KX_out = FX_out * 2 * np.pi
        KY_out = FY_out * 2 * np.pi
        MK_out = np.sqrt(KX_out ** 2 + KY_out ** 2)
        taylor_no_sup = (c * kmax + d * (MK_out ** 2) / kmax)
        etay = np.exp(1j * z * taylor_no_sup)
        spherical_ideal = np.sqrt(k_wl ** 2 - MK_out ** 2)
        esph = np.exp(1j * z * spherical_ideal)
        h = spherical_ideal - taylor_no_sup

        # Computation of the j=3 step
        phase_h = np.exp(1j * z * h)
        EM3 = FE2 * phase_h
        E_out = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(EM3)))
        # half_size3 = [int(np.shape(E_out)[0]/2),int(np.shape(E_out)[1]/2)]
        # E_out = E_out[half_size3[0]-int(M/2):half_size3[0]+int(M/2),half_size3[1]-int(N/2):half_size3[1]+int(N/2)]
        # E_out = E_out[half_size3[0]-int(5017/2):half_size3[0]+int(5017/2),half_size3[1]-int(5017/2):half_size3[1]+int(5017/2)]
        # print('Output pixel pitch: ', pixel_pitch_out[0] * 10 ** 6, 'um')
        return E_out

    def convergentSAASM_full(self, z, field, wavelength, L, w_c):
        zp = L-z
        Magn = L/z
        k_wvl = 2*np.pi/wavelength
        N,M = np.shape(field)
        pad = int(M/2)
        # E1i = np.abs(self.ifts(np.pad(self.fts(field),((pad,pad),(pad,pad)))))**2
        E1i = field
        N,M = np.shape(E1i)
        xc = np.linspace(-w_c/2,w_c/2,M)
        yc = np.linspace(-w_c/2,w_c/2,N)
        fxc = np.fft.fftshift(np.fft.fftfreq(M,w_c/M))
        fyc = np.fft.fftshift(np.fft.fftfreq(N,w_c/N))
        Xc,Yc = np.meshgrid(xc,yc,indexing='xy')
        FXC,FYC = np.meshgrid(fxc,fyc,indexing='xy')
        kmax = np.amax(2 * np.pi * np.sqrt(FXC**2 + FYC**2))
        k_int = k_wvl/kmax
        c = (2/3 *k_int)+(2/3 * np.sqrt(k_int-0.5))+(1/3 * np.sqrt(k_int**2-1))
        d = 1/3 * np.sqrt(k_int**2-1) - k_wvl/kmax
        gammaf = -kmax/(2*d*L)
        
        xcgamma = gammaf*xc
        ycgamma = gammaf*yc

        inter = RegularGridInterpolator((xc, yc), E1i, bounds_error=False, fill_value=None)
        Xint , Yint = np.meshgrid(xcgamma,ycgamma,indexing='xy')
        # E1i = inter((Xint,Yint))


        Xo = Xc/Magn
        Yo = Yc/Magn
        FXo,FYo = fxc = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(M,w_c/(Magn*M))), np.fft.fftshift(np.fft.fftfreq(N,w_c/(Magn*N))))
        
        
        alphaf = np.exp(-1j*c*kmax*L) * kmax /(2j*d*L) * np.exp(1j * (kmax*((Xint)**2+(Yint)**2)/(4*d*L)))
        
        E1m = gammaf **2 * np.divide(E1i,alphaf)
        E2i = self.fts(E1m)
        
        # psMagn = w_c*k_wvl/(np.sqrt(2) * np.pi * Magn*M) -0.1
        psMagn = 1
        x = np.linspace(-w_c/2,w_c/2,M)/(psMagn*Magn)
        y = np.linspace(-w_c/2,w_c/2,N)/(psMagn*Magn)
        X, Y = np.meshgrid(x,y,indexing='xy')
        FX, FY = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(M,w_c/(psMagn*Magn*M))), np.fft.fftshift(np.fft.fftfreq(N,w_c/(psMagn*Magn*N))))

        #_______________________________________
        X = Xo
        Y = Yo
        FX = FXo
        FY = FYo
        #_____________________________
        

        kernelf = np.exp(-1j * kmax * (X**2 + Y**2)/(4*d*L))
        E2m = np.multiply(E2i,kernelf)
        E3i = self.fts(E2m)
        
        r2 = (2*np.pi*FX)**2 + (2*np.pi*FY)**2
        # complex_show(E3i)
        sp = np.sqrt(k_wvl**2 - r2)
        avort = c*kmax  + d * (r2)/kmax
        h = sp - avort

        
        



        # kz = 0
        
        exp_h = np.exp(1j*(L+zp)*h)
        E3m = E3i * exp_h
        E4i = self.ifts(E3m)
        
        

        # Ef = self.ifts(self.fts(self.fts(np.divide(El,alphaf))*kernelf)*hf)
        # Ef = Ef[pad+1:2*pad,pad+1:2*pad]
        
        kerneld = np.exp(-1j*kmax*(X**2+Y**2)/(4*d*zp))
        E4m = E4i*kerneld
        E4o = self.fts(E4m)

        alphad = -np.exp(1j*c*kmax*zp)*kmax/(2j*d*zp) * np.exp(-1j*kmax*(Xo**2 + Yo**2)/(4*d*zp))
        Eo = alphad * E4o
        # Eo = alphad * self.fts(self.ifts(self.fts(self.fts(np.divide(E1i,alphaf))*kernelf)*hf*hd)*kerneld)
        complex_show(Eo)
        return Eo

    def convergentSAASM_redone(self, z, field, wavelength, L, w_c):
        zp = L-z
        Magn = L/z
        k_wvl = 2*np.pi/wavelength
        N,M = np.shape(field)

        pad = int((M/2))
        
        
        E1i = self.ifts(np.pad(self.fts(field),((pad,pad),(pad,pad))))

        # E1i = field
        N,M = np.shape(E1i)
        xc = np.linspace(-w_c/2,w_c/2,M)
        yc = np.linspace(-w_c/2,w_c/2,N)
        fxc = np.fft.fftshift(np.fft.fftfreq(M,w_c/M))
        fyc = np.fft.fftshift(np.fft.fftfreq(N,w_c/N))
        Xc,Yc = np.meshgrid(xc,yc,indexing='xy')
        FXC,FYC = np.meshgrid(fxc,fyc,indexing='xy')
        



        kmax = np.amax(2 * np.pi * np.sqrt(FXC**2 + FYC**2))

        k_int = k_wvl/kmax
        c = (2/3 *k_int)+(2/3 * np.sqrt(k_int**2 -0.5))-(1/3 * np.sqrt(k_int**2-1))
        d = np.sqrt(k_int**2-1) - k_int

        
        gammaf = 1

        x = np.linspace(-w_c/2,w_c/2,M)/(Magn)
        y = np.linspace(-w_c/2,w_c/2,N)/(Magn)
        X, Y = np.meshgrid(x,y,indexing='xy')
        FX, FY = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(M,w_c/(Magn*M))), np.fft.fftshift(np.fft.fftfreq(N,w_c/(Magn*N))))

        magL = 20
        # alphaf = np.exp(-1j*c*kmax*zp) * kmax /(2j*d*zp) * np.exp(1j * (kmax*((gammaf*Xc)**2+(gammaf*Yc)**2)/(4*d*zp)))
        alphaf = np.exp(-1j*c*kmax*zp*magL) * kmax /(2j*d*zp*magL) * np.exp(1j * (kmax*((gammaf*Xc)**2+(gammaf*Yc)**2)/(4*d*zp*magL)))
        
        E1m = np.divide(E1i,alphaf)
        E2i = self.fts(E1m)
        
        kernelf = np.exp(-1j * kmax * (X**2 + Y**2)/(4*d*zp))
        
        E2m = np.multiply(E2i,kernelf)
        E3i = self.fts(E2m)
        

        r2 = (2*np.pi*FX)**2 + (2*np.pi*FY)**2
        # complex_show(E3i)
        sp = np.sqrt(k_wvl**2 - r2)
        avort = c*kmax  + d * (r2)/kmax
        
        h = (sp - avort)
        # plt.figure()
        # plt.plot(sp[:,int(M/2)])
        # plt.plot(avort[:,int(M/2)])
        # plt.plot(h[:,int(M/2)])
        # plt.show()
        E_forwrd = self.ifts(E3i * np.exp(1j*zp*h))
        
        return E_forwrd

    def convergentSAASM_full_backup(self, z, field, wavelength, L, w_c):
            zp = L-z
            Magn = L/z
            k_wvl = 2*np.pi/wavelength
            N,M = np.shape(field)
            pad = int(M/2)
            E1i = self.ifts(np.pad(self.fts(field),((pad,pad),(pad,pad))))
            N,M = np.shape(E1i)
            xc = np.linspace(-w_c/2,w_c/2,M)
            yc = np.linspace(-w_c/2,w_c/2,N)
            fxc = np.fft.fftshift(np.fft.fftfreq(M,w_c/M))
            fyc = np.fft.fftshift(np.fft.fftfreq(N,w_c/N))
            Xc,Yc = np.meshgrid(xc,yc,indexing='xy')
            Xo = Xc/Magn
            Yo = Yc/Magn
            FXC,FYC = np.meshgrid(fxc,fyc,indexing='xy')
            FXo,FYo = fxc = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(M,w_c/(Magn*M))), np.fft.fftshift(np.fft.fftfreq(N,w_c/(Magn*N))))
            kmax = np.amax(2 * np.pi * np.sqrt(FXC**2 + FYC**2))
            k_int = k_wvl/kmax
            c = (2/3 *k_int)+(2/3 * np.sqrt(k_int-0.5))+(1/3 * np.sqrt(k_int-1))
            d = 1/3 * np.sqrt(k_int-1) - k_wvl/kmax
            # gammaf = -kmax/(2*d*L)
            gammaf = 1
            alphaf = np.exp(-1j*c*kmax*L) * kmax /(2j*d*L) * np.exp(1j * (kmax*((gammaf*Xc)**2+(gammaf*Yc)**2)/(4*d*L)))
            
            E1m = gammaf**2 * np.divide(E1i,alphaf)
            E2i = self.fts(E1m)
            
            psMagn = w_c*k_wvl/(np.sqrt(2) * np.pi * Magn*M) -0.1
            x = np.linspace(-w_c/2,w_c/2,M)/(psMagn*Magn)
            y = np.linspace(-w_c/2,w_c/2,N)/(psMagn*Magn)
            X, Y = np.meshgrid(x,y,indexing='xy')
            FX, FY = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(M,w_c/(psMagn*Magn*M))), np.fft.fftshift(np.fft.fftfreq(N,w_c/(psMagn*Magn*N))))

            #_______________________________________
            X = Xo
            Y = Yo
            FX = FXo
            FY = FYo
            #_____________________________
            

            kernelf = np.exp(-1j * kmax * (X**2 + Y**2)/(4*d*L))
            E2m = np.multiply(E2i,kernelf)
            E3i = self.fts(E2m)
            
            r2 = (2*np.pi*FX)**2 + (2*np.pi*FY)**2
            # complex_show(E3i)
            sp = np.sqrt(k_wvl**2 - r2)
            avort = c*kmax  + d * (r2)/kmax
            h = sp - avort

            
            



            # kz = 0
            
            exp_h = np.exp(1j*(L+zp)*h)
            E3m = E3i * exp_h
            E4i = self.ifts(E3m)
            
            

            # Ef = self.ifts(self.fts(self.fts(np.divide(El,alphaf))*kernelf)*hf)
            # Ef = Ef[pad+1:2*pad,pad+1:2*pad]
            
            kerneld = np.exp(-1j*kmax*(X**2+Y**2)/(4*d*zp))
            E4m = E4i*kerneld
            E4o = self.fts(E4m)

            alphad = -np.exp(1j*c*kmax*zp)*kmax/(2j*d*zp) * np.exp(-1j*kmax*(Xo**2 + Yo**2)/(4*d*zp))
            Eo = alphad * E4o
            # Eo = alphad * self.fts(self.ifts(self.fts(self.fts(np.divide(E1i,alphaf))*kernelf)*hf*hd)*kerneld)
            complex_show(Eo)
            return Eo

    def kreuzer_reconstruct(self, z, holo, lamvda, L, x, ref):
        # Definition of geometrical parameters, definition of contrast hologram
        c1 = 1
        c2 = 1
        [fi, co] = holo.shape
        holoContrast = holo - ref
        NA = np.arctan(x/(2*L))
        if NA>0.55:
            r = 0.8
            c1 = 1
            c2 = 4
            z = z*c2
            # padi = int((2*fi - fi/r)/2) 
            padi = int(fi/4)
            holoContrast = resize(holoContrast,1/r)
            holoContrast = np.pad(holoContrast,(padi,padi))

        
        L = L*c1
        x = x/c2
        
        
        
        

        # dx: real pixel size
        dx = x / fi
        

        [fi, co] = holoContrast.shape
        # deltaX: pixel size at reconstruction plane
        deltaX = z * dx / L

        # crit = (lamvda / fi) * (L/dx)
        # print(deltaX<=crit)
        # Cosenus filter creation
        FC = self.filtcosenoF(100, fi,0)
        # Reconstruct
        K = self.kreuzer3F(z, holoContrast, lamvda, dx, deltaX, L,FC)
        return K

    def kreuzer3F(self, z, field, wavelength, pixel_pitch_in, pixel_pitch_out, L, FC):
        dx = pixel_pitch_in
        dX = pixel_pitch_out
        # Squared pixels
        deltaY = dX
        # Matrix size
        [row, a] = field.shape
        # Parameters
        k = 2 * np.pi / wavelength
        W = dx * row
        #  Matrix coordinates
        delta = np.linspace(1, row, num=row)
        [X, Y] = np.meshgrid(delta, delta)
        # Hologram origin coordinates
        xo = -W / 2
        yo = -W / 2
        # Prepared hologram, coordinates origin
        xop = xo * L / np.sqrt(L ** 2 + xo ** 2)
        yop = yo * L / np.sqrt(L ** 2 + yo ** 2)
        # Pixel size for the prepared hologram (squared)
        deltaxp = xop / (-row / 2)
        deltayp = deltaxp
        # Coordinates origin for the reconstruction plane
        Yo = -dX * row / 2
        Xo = -dX * row / 2
        Xp = (dx * (X - row / 2) * L / (
            np.sqrt(L ** 2 + (dx ** 2) * (X - row / 2) ** 2 + (dx ** 2) * (Y - row / 2) ** 2)))
        Yp = (dx * (Y - row / 2) * L / (
            np.sqrt(L ** 2 + (dx ** 2) * (X - row / 2) ** 2 + (dx ** 2) * (Y - row / 2) ** 2)))
        # Preparation of the hologram
        CHp_m = self.prepairholoF(field, xop, yop, Xp, Yp)
        # Multiply prepared hologram with propagation phase
        Rp = np.sqrt((L ** 2) - (deltaxp * X + xop) ** 2 - (deltayp * Y + yop) ** 2)
        r = np.sqrt((dX ** 2) * ((X - row / 2) ** 2 + (Y - row / 2) ** 2) + z ** 2)
        CHp_m = CHp_m * ((L / Rp) ** 4) * np.exp(-0.5 * 1j * k * (r ** 2 - 2 * z * L) * Rp / (L ** 2))
        # Padding constant value
        pad = int(row / 2)
        # Padding on the cosine rowlter
        FC = np.pad(FC, (int(pad), int(pad)))
        # Convolution operation
        # First transform
        T1 = CHp_m * np.exp((1j * k / (2 * L)) * (
                2 * Xo * X * deltaxp + 2 * Yo * Y * deltayp + X ** 2 * deltaxp * dX + Y ** 2 * deltayp * deltaY))
        T1 = np.pad(T1, (int(pad), int(pad)))
        T1 = self.fts(T1 * FC)
        # Second transform
        T2 = np.exp(-1j * (k / (2 * L)) * ((X - row / 2) ** 2 * deltaxp * dX + (Y - row / 2) ** 2 * deltayp * deltaY))
        T2 = np.pad(T2, (int(pad), int(pad)))
        T2 = self.fts(T2 * FC)
        # Third transform
        K = self.ifts(T2 * T1)
        K = K[pad + 1:pad + row, pad + 1: pad + row]

        return K

    def angularSpectrum(self, z_micro, field, wavelength, pixel_pitch_in, L):
        '''
        # Function from pyDHM (https://github.com/catrujilla/pyDHM)
        # Function to diffract a complex field using the angular spectrum approach
        # Inputs:
        # field - complex field
        # z - propagation distance
        # wavelength - wavelength
        # dx/dy - sampling pitches
        '''
        Magn = L/z_micro
        z = L-z_micro
        M, N = np.shape(field)
        pixel_pitch_out = pixel_pitch_in/Magn
        crit = (N*pixel_pitch_in)**2 / (wavelength*3*(N-3))
        # if z<=crit:
            # print(f'For {z*1000} mm, the AS formalism is not valid')

        x = np.arange(0, N, 1)  # array x
        y = np.arange(0, M, 1)  # array y
        X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')
        # dfx = 1 / (pixel_pitch_in * M)
        # dfy = 1 / (pixel_pitch_in * N)
        dfx = 1 / (pixel_pitch_in * M)
        dfy = 1 / (pixel_pitch_in * N)

        field_spec = np.fft.fftshift(field)
        field_spec = np.fft.fft2(field_spec)
        field_spec = np.fft.fftshift(field_spec)
            
        phase = np.exp2(1j * z * np.pi * np.sqrt(np.power(1/wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))
        
        tmp = field_spec*phase
        
        out = np.fft.ifftshift(tmp)
        out = np.fft.ifft2(out)
        out = np.fft.ifftshift(out)

        return out

    def rayleigh_convolutional(self, z_micro, field, wavelength, L, width):
        
        N,_ = np.shape(field)
        N_init = N
        zp = int(N/2)
        sample = self.fts(field)
        sample = np.pad(sample,((zp,zp),(zp,zp)))
        sample = self.ifts(sample)
        N,_ = np.shape(sample)
        dxc = width/N
        width = dxc*N
        k_wvl = 2 * np.pi / wavelength
        # z = L - z_micro
        z = z_micro
        z_micro = L - z
        M = L/z_micro
        Ws = width/M
        xc = yc = np.linspace(-width/2,width/2,N)
        xs = ys = np.linspace(-Ws/2,Ws/2,N)
        
        X,Y = np.meshgrid(xs,ys,indexing='xy')
        X0,Y0 = np.meshgrid(xc,yc, indexing='xy')
        r = np.sqrt((X - X0)**2 + (Y - Y0)**2 + (z)**2)
        ks = (np.exp(1j * k_wvl * r)/r) * (1j * k_wvl + 1/r) * ((z)/r)
        reconstruction = self.ifts(self.fts(sample) * self.fts(ks))
        zp = int(zp/2)
        return reconstruction

    def AS(self, U0, k, fx, fy, z):
        E = np.exp(-1j * z * np.sqrt(k ** 2 - 4 * np.pi * (fx ** 2 + fy ** 2)))
        Uz = self.ifts(self.fts(U0) * E)
        return Uz

    def prepairholoF(self, CH_m, xop, yop, Xp, Yp):
        # User function to prepare the hologram using nearest neihgboor interpolation strategy
        [row, a] = CH_m.shape
        # New coordinates measured in units of the -2*xop/row pixel size
        Xcoord = (Xp - xop) / (-2 * xop / row)
        Ycoord = (Yp - yop) / (-2 * xop / row)
        # Find lowest integer
        iXcoord = np.floor(Xcoord)
        iYcoord = np.floor(Ycoord)
        # Assure there isn't null pixel positions
        iXcoord[iXcoord == 0] = 1
        iYcoord[iYcoord == 0] = 1

        # Assure there are no outrange values
        iXcoord[iXcoord == row] = row-2
        iYcoord[iYcoord == row] = row-2
        iXcoord[iXcoord == row-1] = row-2
        iYcoord[iYcoord == row-1] = row-2

        # Calculate the fractionating for interpolation
        x1frac = (iXcoord + 1.0) - Xcoord  # Upper value to integer
        x2frac = 1.0 - x1frac
        y1frac = (iYcoord + 1.0) - Ycoord  # Lower value to integer
        y2frac = 1.0 - y1frac
        x1y1 = x1frac * y1frac  # Corresponding pixel areas for each direction
        x1y2 = x1frac * y2frac
        x2y1 = x2frac * y1frac
        x2y2 = x2frac * y2frac
        # Pre allocate the prepared hologram
        CHp_m = np.zeros([row, row])
        # Prepare hologram (preparation - every pixel remapping)
        for it in range(0, row - 2):
            for jt in range(0, row - 2):
                CHp_m[int(iYcoord[it, jt]), int(iXcoord[it, jt])] = CHp_m[int(iYcoord[it, jt]), int(iXcoord[it, jt])] + (
                    x1y1[it, jt]) * CH_m[it, jt]
                CHp_m[int(iYcoord[it, jt]), int(iXcoord[it, jt]) + 1] = CHp_m[int(iYcoord[it, jt]), int(
                    iXcoord[it, jt]) + 1] + (x2y1[it, jt]) * CH_m[it, jt]
                CHp_m[int(iYcoord[it, jt]) + 1, int(iXcoord[it, jt])] = CHp_m[int(iYcoord[it, jt]) + 1, int(
                    iXcoord[it, jt])] + (x1y2[it, jt]) * CH_m[it, jt]
                CHp_m[int(iYcoord[it, jt]) + 1, int(iXcoord[it, jt]) + 1] = CHp_m[int(iYcoord[it, jt]) + 1, int(
                    iXcoord[it, jt]) + 1] + (x2y2[it, jt]) * CH_m[it, jt]

        return CHp_m
    
    def filtcosenoF(self, par, fi, num_fig):
        # Coordinates
        Xfc, Yfc = np.meshgrid(np.linspace(-fi / 2, fi / 2, fi), np.linspace(fi / 2, -fi / 2, fi))

        # Normalize coordinates [-π,π] and create horizontal and vertical filters
        FC1 = np.cos(Xfc * (np.pi / par) * (1 / Xfc.max())) ** 2
        FC2 = np.cos(Yfc * (np.pi / par) * (1 / Yfc.max())) ** 2

        # Intersection
        FC = (FC1 > 0) * (FC1) * (FC2 > 0) * (FC2)

        # Rescale
        FC = FC / FC.max()

        if num_fig != 0:
            fig = px.imshow(FC)
            fig.show()

        return FC

    def norm_bits(self, img, bits):
        norm = ((img - img.min()) / img.max()) * bits
        norm = np.round(norm, 0)
        return norm

    def ifts(self, A):
        return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(A)))

    def fts(self, A):
        return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(A)))
    
    def point_src(self, M, z, x0, y0, lamb, dx):
        N = M
        dy = dx

        m, n = np.meshgrid(np.linspace(-M/2,M/2,M), np.linspace(-N/2, N/2, N))

        k = 2 * np.pi / lamb

        r = np.sqrt(z ** 2 + (m * dx - x0) ** 2 + (n * dy - y0) ** 2)

        P = np.exp(1j * k * r) / r

        return P

class metrics:
    def measure_noise(self, I, rangex,rangey):
        '''
        Function to estimate the sigma noise factor in an image according to https://doi.org/10.1006/cviu.1996.0060 implemented
        in https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image
        
        ### Input:
        * I: Grayscale image as a numpy array 
        '''
        region = I[rangex[0]:rangex[1],rangey[0]:rangey[1]]
        sigma = np.std(region)
        return sigma
    
    def find_peaks(self, signal, threshold):
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i] < signal[i-1] and signal[i] < signal[i+1] and signal[i] < threshold:
                peaks.append(i)
        return peaks

    def measure_contrast(self,profile,min_coords,max_coords):
        # Input are: the profiel to evaluate the intensity, min coords and max coords
        # are tuples with the range to evaluate the intensity 
        min_value = np.mean(profile[min_coords[0]:min_coords[1]])
        max_value = np.mean(profile[max_coords[0]:max_coords[1]])
        return (max_value - min_value)/(max_value + min_value)


    def measure_resolution(self,I,profile_idx_0,min_coords,max_coords,kreuzer_in):

        measurements = 5
        I_array = np.copy(I)
        I = np.uint8(I*255)
        M,N = np.shape(I)
        contrast_mat = np.zeros((measurements,6))

        for i in range(measurements):
            
            profile_idx = profile_idx_0+i
            profile = I_array[:,profile_idx]
            

            for j in range(6):
                local_min = min_coords[j]
                local_max = max_coords[j]


                contrast = self.measure_contrast(profile,local_min,local_max)
                contrast_mat[i,j] = contrast

        
        # freqpx = [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
        freqpx = [1/12, 1/10, 1/8, 1/6, 1/4, 1/2]
        contrast = np.mean(contrast_mat,axis=0)
        std = np.std(contrast_mat,axis=0)

        return contrast, std
    
    def measure_distortion_improved(self,I,L):
        pixel_pitch = 2.93e-6 #Camera sampling
        z = L/4
        I = I[300:724,300:724]
        I = (I * 255).astype(np.uint8)
        comp = cv2.adaptiveThreshold(I, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 20)
        M,N = np.shape(comp)
        mid = np.sqrt(2 * int(N/2)**2)
        diagonal = np.diag(comp)
        first_cero = np.where(diagonal == 0)[0][0]

        
        x = y = pixel_pitch * (212-first_cero)
        delta_L = np.abs(np.sqrt(x**2 + y**2 +L**2)-L)
        k1 = delta_L/z

        # distortion_percent = distortion*100
        return k1,(212-first_cero)

    def measure_distortion(self,I):
        kernel = np.array([[0.88,0.26,0.04,0.26,0.88],
                           [0.26,0.0, 0.0, 0.0,0.27],
                           [0.05,0.0, 0.0, 0.0,0.05],
                           [0.26,0.0, 0.0, 0.0,0.27],
                           [0.88,0.26,0.043,0.26,0.88]])
        I = np.uint8(I*255)
        grayImage = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
        Mar,N = np.shape(I)
        grayImage = cv2.filter2D(grayImage, -1, kernel)
        image_copy = grayImage.copy()
        _, thr = cv2.threshold(grayImage,200,255, cv2.THRESH_BINARY_INV)
        # image_copy = thr.copy()
        thr = cv2.cvtColor(thr, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(image=thr, mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
        cntmax = 0
        centroids = []
        rmax = 0
        cXm = cYm = 0
        r2 = 0
        for cnt in contours:
            # Calculate moments
            M = cv2.moments(cnt)
            
            # Calculate centroid
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            r = cX**2 + cY**2
            if r>r2:
                if r>rmax:
                    cntmax = cnt
                    r2 = rmax
                    rmax = r
                    c2X = cXm
                    cXm = cX
                    c2Y = cYm
                    cYm = cY
                else:
                    c2X = cX
                    c2Y = cY
                    r2 = r
                
            centroids.append((cX, cY))
        center_max = (cXm,cYm)
        center_2 = (c2X,c2Y)
        ideal_relation = 0.3273657289002558
        d2 = np.sqrt((center_max[0] - center_2[0])**2 + (center_max[1] - center_2[1])**2)
        d1 = np.sqrt((center_max[0] - Mar/2)**2 + (center_max[1] - N/2)**2)
        distortion = np.abs((ideal_relation - (d2/d1))/ideal_relation)
        cv2.drawContours(image=image_copy, contours=cntmax, contourIdx=-1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(image_copy, center_max, radius=5, color=(0, 0, 255), thickness=-1)
        cv2.circle(image_copy, (c2X,c2Y), radius=5, color=(0, 255, 0), thickness=-1)         
        # see the results
        resized_image = cv2.resize(image_copy, (512,512), interpolation=cv2.INTER_AREA)
        cv2.imshow('None approximation', resized_image)
        cv2.waitKey(0)
        # cv2.imwrite(r"F:\OneDrive - Universidad EAFIT\Semestre X\TDG\Images\distort.jpg",image_copy)

        return distortion*100,center_max
    
    def measure_phase_sensitivity(self,I):
        diagonal = np.diag(I)
        prof = np.flip(diagonal)[110:914]
        mean = np.mean(prof[0:140])
        center_prof = prof - mean
        

        # sens = vals[0]
        return center_prof
    

def open_image(file_path):
    im = Image.open(file_path).convert('L')
    im = np.asarray(im) / 255
    return im

def save_image(I, file_path):
    I = I - np.amin(I)
    I = I / np.amax(I)
    I = Image.fromarray(np.uint8(I*255))
    path = file_path
    I.save(path)

def complex_show(U,negative=False):
    amplitude = np.abs(U)
    amplitude = amplitude-np.amin(amplitude)
    amplitude = amplitude*255/np.amax(amplitude)
    if negative==True:
        amplitude = 255-amplitude
    amplitude = skm.color.gray2rgb(amplitude)
    amplitude = go.Image(z=amplitude)
    
    phase = np.angle(U)
    phase = phase-np.amin(phase)
    phase = phase*255/np.amax(phase)
    phase = skm.color.gray2rgb(phase)
    phase = go.Image(z=phase)
    fig = make_subplots(rows=1,cols=2,subplot_titles=("Amplitude","Phase"))
    fig.add_trace(
        amplitude,row=1,col=1
    )
    fig.add_trace(
        phase,row=1,col=2
    )
    fig.write_html(r'images\complex_show.html')
    return fig

def norm(array):
        array = array-np.amin(array)
        array = array/np.amax(array)
        return array

def save_gif(xarray):
    # Load your xarray data (replace this with your actual xarray)
    
    # Initialize an empty list to store PIL Image objects
    image_list = []

    # Loop through the xarray to convert each 2D array to a PIL Image
    for i in range(len(xarray["z"])):
        img_array = norm(xarray.isel(z=i).values)
        img_array = (img_array * 255).astype(np.uint8)  # Convert to 8-bit pixel values
        img = Image.fromarray(img_array, "L")  # "L" means grayscale
        image_list.append(img)

    # Save as an animated GIF
    image_list[0].save("animated.gif",
                    save_all=True, append_images=image_list[1:], loop=0, duration=200)

def find_local_minima(arr, percentage_threshold,maxi,neighbourhood):
    arr = 1-arr
   
    peaks = scp.find_peaks_cwt(arr,widths=15)
    # plt.plot(peaks,'r*')
    if len(peaks) > 7:
        peaks = peaks[1:7]
    minima = []

    # for i in range(neighbourhood, len(arr)-neighbourhood):
    #     local_mean = np.mean(arr[i-neighbourhood:i+neighbourhood])
    #     if (arr[i] < arr[i-1]) and (arr[i] < arr[i+1]) and (arr[i] < (percentage_threshold/100)*maxi) and (arr[i]<local_mean):
    #         minima.append(i)

    return peaks

def resize(I,sr):
    M,N = np.shape(I)
    M = int(sr*M)
    N = int(sr*N)
    shape = (M,N)
    I = np.uint8(I*255)
    grayImage = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
    out = cv2.resize(grayImage, shape, interpolation=cv2.INTER_AREA)

    return norm(np.asarray(out)[:,:,0])











