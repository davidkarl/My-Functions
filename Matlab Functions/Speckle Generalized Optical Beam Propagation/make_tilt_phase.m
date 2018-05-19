function [tilt_phase]=make_tilt_phase(lambda,alpha_angle_x,alpha_angle_y,X,Y)
%the factor 2 in the exponential comes due to the fact that in speckle
%photography the far field shift is the result of specular reflection type
%scattering for small tilt angles.
k=2*pi/lambda;
tilt_phase = exp(-1i*2*k*tan(alpha_angle_x)*X - 1i*2*k*tan(alpha_angle_y)*Y);
