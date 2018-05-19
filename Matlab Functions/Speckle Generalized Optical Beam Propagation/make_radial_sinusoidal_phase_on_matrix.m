function [sinus_phase] = make_radial_sinusoidal_phase_on_matrix(a,spatial_frequency_r_in_pixels)

N=size(a);
x=[-N/2:N/2-1];
[X,Y]=meshgrid(x);

sinus_phase = exp(2*pi*1i*sqrt(X.^2+Y.^2)/spatial_frequency_r_in_pixels);


