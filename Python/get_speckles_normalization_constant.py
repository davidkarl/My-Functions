#Get speckle gaussian beam normalization factor:

#For ROI=100, speckle_size=10 -> normalization constant for speckle INTENSITY = 190.06
#THEREFORE:  normalization_constant = 190.06 * (10/speckle_size)**2 * (N/100)**2

ROI = 100;
speckle_size = 10;

    
#indexing auxiliary variables
start = -1;
end = -1;
    
#Simulation N:
N = ROI + 10;
    
#Get gaussian beam:
phase_screen = exp(1j*100*randn(N,N));
x = arange(-fix(N/2),ceil(N/2),1);
[X,Y] = meshgrid(x,x);    
radius = N/speckle_size;
gaussian_beam = (X**2 + Y**2) < radius;
gaussian_beam = exp(-(X**2+Y**2)/radius**2);
normalization_constant = sqrt(190.06 * (10/speckle_size)**2 * (N/110)**2);
gaussian_beam = gaussian_beam / normalization_constant;
    
#Get tilt phases k-space:
delta_f1 = 1/(N);
f_x = x*delta_f1;
#Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
f_x = fftshift(f_x);
#Build k-space meshgrid:
[kx,ky] = meshgrid(f_x,f_x);         



gaussian_beam_after_phase = gaussian_beam * phase_screen;
speckles = abs(fft2(gaussian_beam_after_phase))**2;
speckles_mean = mean(speckles);
print(speckles_mean);
#figure(1)
#imshow((speckles));
#figure(2)
#imshow(fftshift(speckles));


#inversly proportional to (speckle_size)^2
#proportional to (N)^2


