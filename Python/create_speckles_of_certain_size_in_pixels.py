import numpy
from numpy import * 
from numpy.random import *
from numpy.fft import *
from matplotlib.pyplot import *

def create_speckles_of_certain_size_in_pixels( speckle_size_in_pixels=10, N=512, polarization=0,flag_gauss_circ=1):   
    
#    import numpy
#    from numpy import arange
#    from numpy.random import *

#    #Parameters:
#    speckle_size_in_pixels = 10;
#    N = 512;
#    polarization = 0;
#    flag_gauss_circ = 1;
   
    #Calculations:
    wf = (N/speckle_size_in_pixels);
    
    if flag_gauss_circ == 1:
        x = arange(-N/2,N/2,1);
        distance_between_the_two_beams_x = 0;
        distance_between_the_two_beams_y = 0;
        [X,Y] = meshgrid(x,x);
        beam_one = exp( - ((X-distance_between_the_two_beams_x/2)**2 + 
                           (Y-distance_between_the_two_beams_y/2)**2)/wf**2);
        beam_two = exp( - ((X+distance_between_the_two_beams_x/2)**2 + 
                           (Y+distance_between_the_two_beams_y/2)**2)/wf**2);
        total_beam = beam_one + beam_two;
        total_beam = total_beam / sqrt(sum(sum(abs(total_beam)**2)));
    else:
          x = arange(-N/2,N/2,1)*1;
          y = x;
          [X,Y] = meshgrid(x,y);
          c = 0;
          distance_between_the_two_beams_y = 0;
          beam_one=((X-distance_between_the_two_beams_y/2)**2+(Y-distance_between_the_two_beams_y/2)**2<=wf^2);
          beam_two=((X+distance_between_the_two_beams_y/2)**2+(Y+distance_between_the_two_beams_y/2)**2<=wf^2);
          total_beam = beam_one + beam_two;
         


    #Polarization:
    if (polarization>0 & polarization<1) : 
        beam_one = total_beam*exp(2*pi*1j*10*randn(N,N));
        beam_two = total_beam*exp(2*pi*1j*10*randn(N,N));
        speckle_pattern1 = fftshift(fft2(fftshift(beam_one)));
        speckle_pattern2 = fftshift(fft2(fftshift(beam_two)));
        speckle_pattern_total_intensity = (1-polarization)*abs(speckle_pattern1)**2 + \
                                            polarization*abs(speckle_pattern2)**2;
    else:
        total_beam = total_beam * exp(2*pi*1j*(10*randn(N,N)));
        speckle_pattern1 = fftshift(fft2(fftshift(total_beam)));
        speckle_pattern2 = numpy.empty_like(speckle_pattern1);
        speckle_pattern_total_intensity = abs(speckle_pattern1)**2;
    
    
    imshow(speckle_pattern_total_intensity);
    
    return speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2;
   
    
#create_speckles_of_certain_size_in_pixels()