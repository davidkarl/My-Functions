function [tilt_phase,tilt_frame]=make_tilt_phase_for_fft(N,alpha_angle_x,alpha_angle_y,tilt_phase_frame_size_x,tilt_phase_frame_size_y)
%the factor 2 in the exponential comes due to the fact that in speckle
%photography the far field shift is the result of specular reflection type
%scattering for small tilt angles.
x=[-N/2:1:N/2-1];
[X,Y]=meshgrid(x);

%alpha_angle_x and alpha_angle_y are stated in [radians]
tilt_frame = (abs(X)<=tilt_phase_frame_size_x/2 & abs(Y)<=tilt_phase_frame_size_y/2);
% tilt_phase = exp((1i*alpha_angle_x*X*max(N/2/tilt_phase_frame_size_x,1) + 1i*alpha_angle_y*Y*max(N/2/tilt_phase_frame_size_y,1)).*tilt_frame);
tilt_phase = exp(2*pi*(1/N)*(1i*alpha_angle_x*X + 1i*alpha_angle_y*Y).*tilt_frame);

