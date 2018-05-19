function [sinus_phase] = make_sinusoidal_phase_on_matrix(a,spatial_frequency_x_in_pixels,spatial_frequency_y_in_pixels)

N=size(a);
x=[-N/2:N/2-1];
[X,Y]=meshgrid(x);

sinus_phase = exp(2*pi*1i*(X/spatial_frequency_x_in_pixels+Y/spatial_frequency_y_in_pixels));


