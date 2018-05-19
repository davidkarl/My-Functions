function [sinus_phase] = make_radial_gaussian_phase_on_matrix(a,variance_in_pixels)

N=size(a);
x=[-N/2:N/2-1];
[X,Y]=meshgrid(x);

sinus_phase = exp(1i*(X.^2+Y.^2)/(2*variance_in_pixels));


