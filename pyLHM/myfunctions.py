# August 2023
# Library for DLHM, In-line holography
# Authors: Tomas Velez, Maria J Lopera
# Kreuzer taken from Maria J Lopera (https://github.com/mloper23/DLHM-backend), angular spectrum taken from C Trujillo (https://github.com/catrujilla/pyDHM)
# Autofocus based on Maria J Lopera (https://github.com/mloper23/DLHM-backend)

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
from scipy.signal import convolve2d
import plotly.express as px
import cv2 as cv


def autofocus(propagator_name, parameters, z1, z2, it):
    delta = (z2 - z1) / it
    focus_metric = []
    for i in range(0, it):
        params = [z1 + i * delta] + parameters
        hz = reconstruct(propagator_name, params)
        focus_metric[i] = np.sum(np.abs(hz))
    min = np.argmin(focus_metric)
    d = z1 + min * delta
    return d


def realisticDLHM(sample, L, z, w_c, lamb, pinhole, it_ph, bits):
    N, M = sample.shape
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
        sample_ = sample
        Ui = AS(sample_, 2*np.pi/lamb, fx, fy, L-z)
        ps_ = point_src(M, L, x_unit[i] * w_s / N, y_unit[i] * w_s /N, lamb, w_c/N)
        ps = ps + ps_
        holo = holo + Ui
    # return holo, ref

    holo = np.abs(holo) ** 2
    camMat = np.array([[N, 0, N/2], [0, M, M/2], [0, 0, 1]])
    dist = np.array([-NA*0.1, 0, 0, 0])
    holo = cv.undistort(holo, camMat, dist)

    holo = holo * (1 - r/r.max())
    holo = norm_bits(holo, bits)

    ref = np.abs(1-r/r.max()) ** 2
    ref = norm_bits(ref, bits)

    return holo, ref


def norm_bits(img, bits):
    norm = ((img - img.min()) / img.max()) * bits
    norm = np.round(norm, 0)
    return norm


def AS(U0, k, fx, fy, z):
    E = np.exp(-1j * z * np.sqrt(k ** 2 - 4 * np.pi * (fx ** 2 + fy ** 2)))
    Uz = ifts(fts(U0) * E)
    return Uz


def ifts(A):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(A)))

def fts(A):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(A)))

def point_src(M, z, x0, y0, lamb, dx):
    N = M
    dy = dx

    m, n = np.meshgrid(np.linspace(-M/2,M/2,M), np.linspace(-N/2, N/2, N))

    k = 2 * np.pi / lamb

    r = np.sqrt(z ** 2 + (m * dx - x0) ** 2 + (n * dy - y0) ** 2)

    P = np.exp(1j * k * r) / r

    return P


def angularSpectrum(z, field, wavelength, pixel_pitch_in, pixel_pitch_out):
    '''
    # Function from pyDHM (https://github.com/catrujilla/pyDHM)

    # Function to diffract a complex field using the angular spectrum approach
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx/dy - sampling pitches
    '''
    M, N = np.shape(field)
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (pixel_pitch_in[0] * M)
    dfy = 1 / (pixel_pitch_in[1] * N)

    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)

    phase = np.exp2(
        1j * z * np.pi * np.sqrt(np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))

    tmp = field_spec * phase

    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)

    return out


def rayleigh1Free(z, field, wavelength, pixel_pitch_in, pixel_pitch_out, out_shape):
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
    return U0, Viewing_window


def convergentSAASM(z, field, wavelength, pixel_pitch_in, pixel_pitch_out):
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
    print('Output pixel pitch: ', pixel_pitch_out[0] * 10 ** 6, 'um')
    return E_out


def kreuzer3F(z, field, wavelength, pixel_pitch_in, pixel_pitch_out, L, FC):
    dx = pixel_pitch_in[0]
    dX = pixel_pitch_out[0]
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
    CHp_m = prepairholoF(field, xop, yop, Xp, Yp)
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
    T1 = ft(T1 * FC)
    # Second transform
    T2 = np.exp(-1j * (k / (2 * L)) * ((X - row / 2) ** 2 * deltaxp * dX + (Y - row / 2) ** 2 * deltayp * deltaY))
    T2 = np.pad(T2, (int(pad), int(pad)))
    T2 = ft(T2 * FC)
    # Third transform
    K = ift(T2 * T1)
    K = K[pad + 1:pad + row, pad + 1: pad + row]

    return K


def filtcosenoF(par, fi, num_fig):
    # Coordinates
    Xfc, Yfc = np.meshgrid(np.linspace(-fi / 2, fi / 2, fi), np.linspace(fi / 2, -fi / 2, fi))

    # Normalize coordinates [-π,π] and create horizontal and vertical filters
    FC1 = np.cos(Xfc * (np.pi / par)) * (1 / Xfc.max()) ** 2
    FC2 = np.cos(Yfc * (np.pi / par)) * (1 / Yfc.max()) ** 2

    # Intersection
    FC = (FC1 > 0) * (FC1) * (FC2 > 0) * (FC2)

    # Rescale
    FC = FC / FC.max()

    if num_fig != 0:
        fig = px.imshow(FC)
        fig.show()

    return FC


def prepairholoF(CH_m, xop, yop, Xp, Yp):
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


def open_image(file_path):
    im = Image.open(file_path).convert('L')
    im = np.asarray(im) / 255
    return im


def reconstruct(propagator_name, parameters):
    if (propagator_name in locals()) and (callable(locals()[propagator_name])):
        propagator = locals()[propagator_name]
        try:
            output_image = propagator(*parameters)
        except:
            print('Not the appropiate set of parameters')
            exit()
    else:
        print('Not a valid propagator')
        exit()

    return output_image


def measure_noise(I):
    '''
    Function to estimate the sigma noise factor in an image according to https://doi.org/10.1006/cviu.1996.0060 implemented
    in https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image

    
    ### Input:
    * I: Grayscale image as a numpy array 
    '''
    H, W = I.shape

    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))
    return sigma


def ft(u):
    ut = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(u)))
    return ut


def ift(u):
    uit = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(u)))
    return uit
