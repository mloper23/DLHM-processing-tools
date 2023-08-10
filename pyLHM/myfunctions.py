import numpy as np

def angularSpectrum(field, z, wavelength, dx, dy):
    '''
    # Function from pyDHM

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

    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)
    
    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)
        
    phase = np.exp2(1j * z * np.pi * np.sqrt(np.power(1/wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))
	
    tmp = field_spec*phase
    
    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)
	
    return out

def RS1_Free(Field_Input,z,wavelength,pixel_pitch_in,pixel_pitch_out,Output_shape):
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


    dx = pixel_pitch_in[0] #Input Pixel Size X
    dy = pixel_pitch_in[1] #Input Pixel Size Y
    ds = dx*dy
    dx_out = pixel_pitch_out[0] #Output Pixel Size X
    dy_out = pixel_pitch_out[1] #Output Pixel Size Y
    
    M,N = np.shape(Field_Input)
    (M2,N2) = Output_shape
    k = (2*np.pi)/wavelength # Wave number of the ilumination source
    

    U0 = np.zeros(Output_shape,dtype='complex_')
    U1 = Field_Input  #This will be the hologram plane 


    x_inp_lim = dx*int(N/2)
    y_inp_lim = dy*int(M/2)

    x_cord = np.linspace(-x_inp_lim , x_inp_lim , num = N)
    y_cord = np.linspace(-y_inp_lim , y_inp_lim , num = M)

    [X_inp,Y_inp] = np.meshgrid(x_cord,y_cord,indexing='xy')


    x_out_lim = dx_out*int(N2/2)
    y_out_lim = dy_out*int(M2/2)

    x_cord_out = np.linspace(-x_out_lim , x_out_lim , num = N2)
    y_cord_out = np.linspace(-y_out_lim , y_out_lim , num = M2)

    
    # The first pair of loops ranges over the points in the output plane in order to determine r01
    for x_sample in range(OutputShape[0]):
        x_fis_out = x_cord_out[x_sample]
        for y_sample in range(OutputShape[1]):
            # start = time.time()
            y_fis_out = y_cord_out[y_sample]
            mr01 = np.sqrt(np.power(x_fis_out-X_inp,2)+np.power(y_fis_out-Y_inp,2)+(z)**2)
            Obliquity = (z)/ mr01
            kernel = np.exp(1j * k * mr01)/mr01
            dif = (1j*k)+(1/mr01)
            U0[y_sample,x_sample] = np.sum(U1 * dif * kernel * Obliquity * ds)
            # stop = time.time()
            # print('Tiempo de ejecución: ', 1000*(stop-start))
    U0 = -U0/(2*np.pi)
    Viewing_window = [-x_out_lim,x_out_lim,-y_out_lim,y_out_lim]
    return U0,Viewing_window

def CONV_SAASM_V2(field, z, wavelength, pixel_pitch_in,pixel_pitch_out):
    '''
    Function to diffract a complex field using the angular spectrum approach with a Semi-Analytical spherical wavefront.
    This operator only works for convergent fields, for divergent fields see DIV_SAASM
    For further reference review: https://opg.optica.org/josaa/abstract.cfm?uri=josaa-31-3-591 and https://doi.org/10.1117/12.2642760

    
    ### Inputs:
    * field - complex field to be diffracted
    * z - propagation distance
    * wavelength - wavelength of the light used
    * pixel_pitch_in - Sampling pitches of the input field as a (2,) list
    * pixel_pitch_out - Sampling pitches of the output field as a (2,) list
    '''


    # Starting cooridnates computation
    k_wl = 2 * pi / wavelength
    M, N = field.shape
    #Linear Coordinates
    x = np.arange(0, N, 1)  # array x
    fx = np.fft.fftshift(np.fft.fftfreq(N,pixel_pitch_in[0]))
    y = np.arange(0, M, 1)  # array y
    fy = np.fft.fftshift(np.fft.fftfreq(M,pixel_pitch_in[1]))
    #Grids
    X_in, Y_in = np.meshgrid((x - (N / 2))*pixel_pitch_in[0], (y - (M / 2))*pixel_pitch_in[1], indexing='xy')
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    KX = FX * 2 * pi
    KY = FY * 2 * pi
    MR_in = (X_in**2 + Y_in**2)
    MK = np.sqrt(KX**2 + KY**2)
    kmax = np.amax(MK)

    # Fitting parameters for the parabolic fase
    k_interm = (k_wl/kmax)
    c = (2/3 * k_interm) + 2/3 * np.sqrt(k_interm**2 - 0.5)- 1/3 * np.sqrt(k_interm**2 -1)
    d = np.sqrt(k_interm**2 - 1) - k_interm
    pp0 = pixel_pitch_in[0]
    


    #Calculating the beta coordinates as output for the first fourier transform
    X_out, Y_out = np.meshgrid((x - (N / 2))*pixel_pitch_out[0], (y- (M / 2))*pixel_pitch_out[1], indexing='xy')
    bX = -kmax * X_out / (2*d*z)
    bY = -kmax * Y_out / (2*d*z)
    Mbeta = np.sqrt(np.power(bX,2)+np.power(bY,2))
    

    ''' IN THIS STEP THE FIRST FOURIER TRANSFORM OF THE FIELD IS CALCULATED DOING A RESAMPLING USING THE
    FAST FOURIER TRASNSFORM AND A PADDING. THIS TRANSFORM HAS AS OUTPUT COORDINATE THE SCALED COORDINATE
    BETA, THAT IS NOT RELEVANT FOR THIS STEP BUT THE NEXT ONE'''
    # Initial interpolation for j=1
    max_grad_alpha = -kmax/(2*d*z) * np.amax(MR_in)
    alpha = (np.exp(-1j* c * kmax * z)*kmax/(2j * d * z)) * np.exp((1j * kmax * MR_in)/(4*d*z))

    #Interpolation of the input field Scipy
    xin = (x - (N / 2))*pp0
    yin = (y - (M / 2))*pp0
    N2 = int(N*(2+max_grad_alpha*pp0/np.pi))
    M2 = int(M*(2+max_grad_alpha*pp0/np.pi))

    pp1 = M*pixel_pitch_in[0]/M2
    x1 = np.arange(0, N2-1, 1)
    y1 = np.arange(0, M2-1, 1)
    

    X1,Y1 = np.meshgrid((x1 - (N2 / 2))*pp1, (y1 - (M2 / 2))*pp1,indexing='ij')
    inter = RegularGridInterpolator((xin,yin),field,bounds_error=False, fill_value=None)
    E_interpolated = inter((X1,Y1))
    MR1 = (X1**2 + Y1**2)
    alpha = np.exp(-1j* c * kmax * z)*kmax/(2j * d * z) * np.exp((1j * kmax * MR1)/(4*d*z))
    E_interpolated = E_interpolated - np.amin(E_interpolated)
    E_interpolated = E_interpolated/np.amax(E_interpolated)
    EM1 = np.divide(E_interpolated,alpha)

    #Padding variables for j=2
    max_grad_kernel = np.amax(Mbeta)
    
    
    # Computation of the j=1 step
    FE1 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(EM1)))
    #Slicing of the input field in the inner region where the field is valid
    # half_size1 = [int(np.shape(FE1)[0]/2),int(np.shape(FE1)[1]/2)]
    # FE1 = FE1[half_size1[0]-int(M/2):half_size1[0]+int(M/2),half_size1[1]-int(N/2):half_size1[1]+int(N/2)]

    

    '''IN THIS STEP THE SECOND FOURIER TRANSFORM IS CALCULATED. HERE THE COORDINATES BETA ARE RELEVANT
    SINCE THE ELEMENT-WISE PRODUCT OF THE FE1 WITH THE PROPAGATION KERNEL REQUIRES THE KERNEL'S 
    ARGUMENT TO BE THE MAGNITUDE OF BETA INSTEAD OF THE MAGNITUD OF RHO'''
    # Calculation of the oversampled kernel
    M0,N0 = np.shape(FE1)
    x2 = np.arange(0,N0,1)
    y2 = np.arange(0,M0,1)
    # If required, check the pixel size
    X_out, Y_out = np.meshgrid((x2 - (N0 / 2))*pixel_pitch_out[0], (y2- (M0 / 2))*pixel_pitch_out[1], indexing='xy')
    Mrho = np.sqrt(np.power(X_out,2)+np.power(Y_out,2))
    bX = -kmax * X_out / (2*d*z)    
    bY = -kmax * Y_out / (2*d*z)
    print('C = ',str(-kmax/(2*d*z)))
    Mbeta = np.sqrt(np.power(bX,2)+np.power(bY,2))
    kernel = np.exp(-1j * d * z * np.power(Mbeta,2)/(kmax))
    # kernel = np.exp(-1j * kmax * np.power(Mrho,2)/(4 * d * z))
    EM2 = FE1*kernel

    # Computation of the j=2 step
    FE2 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(EM2)))
    half_size2 = [int(np.shape(FE2)[0]/2),int(np.shape(FE2)[1]/2)]
    # FE2 = FE2[half_size2[0]-int(M/2):half_size2[0]+int(M/2),half_size2[1]-int(N/2):half_size2[1]+int(N/2)]
    

    '''IN THIS STEP THE THIRD FOURIER TRANSFORM IS CALCULATED. HERE THE SUPERIOR ORDER TERMS (H) ARE CALCULATED
    TO FIND NUMERICALLY THE MAXIMUM GRADIENT OF ITS ARGUMENT, THEN, A PADDING OF FE2 IS DONE AND FINALLY H
    IS RESAMPLED IN TERMS OF FE2'''
    # Calculation of the superior order phases
    Mfin,Nfin = np.shape(FE2)
    fx_out = np.fft.fftshift(np.fft.fftfreq(Nfin,pixel_pitch_out[0]))
    fy_out = np.fft.fftshift(np.fft.fftfreq(Mfin,pixel_pitch_out[1]))
    FX_out, FY_out = np.meshgrid(fx_out, fy_out, indexing='xy')
    KX_out = FX_out * 2 * pi
    KY_out = FY_out * 2 * pi
    MK_out = np.sqrt(KX_out**2 + KY_out**2)
    taylor_no_sup = (c*kmax + d *(MK_out**2)/kmax)
    spherical_ideal = np.sqrt(k_wl**2 - MK_out**2)
    h = spherical_ideal - taylor_no_sup
    
    # Computation of the j=3 step
    E_out = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(FE2 * np.exp(1j * z * h))))
    half_size3 = [int(np.shape(E_out)[0]/2),int(np.shape(E_out)[1]/2)]
    # E_out = E_out[half_size3[0]-int(M/2):half_size3[0]+int(M/2),half_size3[1]-int(N/2):half_size3[1]+int(N/2)]
    # E_out = E_out[half_size3[0]-int(5017/2):half_size3[0]+int(5017/2),half_size3[1]-int(5017/2):half_size3[1]+int(5017/2)]
    print('Output pixel pitch: ',pp1* 10**6,'um')
    return E_out


