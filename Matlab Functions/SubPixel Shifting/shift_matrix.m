function [shifted_matrix]=shift_matrix(mat_in,spacing,shiftx,shifty)
%shiftx and shifty in [meters]!!!
% N=size(mat_in,1);
% delta_f = 1/(N*spacing);
% f_x=[-N/2:N/2-1]*delta_f;
% [kx,ky]=meshgrid(f_x);
% 
%  displacement_matrix=exp(1i*2*pi*ky*shifty+1i*2*pi*kx*shiftx);
%   
%  shifted_matrix=ifftshift(ifft2(ifftshift(mat_in)));
%  shifted_matrix=fftshift(fft2(fftshift(shifted_matrix.*displacement_matrix)));



% %Test Stuff:
% mat_in = abs(create_speckles_of_certain_size_in_pixels(20,512,1,0));
% spacing = 1;
% shiftx = 0.5;
% shifty = 0;

%Build k-space axis vecs:
N=size(mat_in,2);
M=size(mat_in,1);
delta_f1 = 1/(N*spacing);
f_x=[-fix(N/2):ceil(N/2)-1]*delta_f1;
delta_f2 = 1/(M*spacing);
f_y=[-fix(M/2):ceil(M/2)-1]*delta_f2;

% %Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
f_x = fftshift(f_x);
f_y = fftshift(f_y);

%Build k-space meshgrid:
[kx,ky]=meshgrid(f_x,f_y); 

%Build displacement matrix:
displacement_matrix = exp(-(1i*2*pi*ky*shifty+1i*2*pi*kx*shiftx));

%Get shifted matrix:
shifted_matrix = fft2(mat_in);
shifted_matrix = ifft2(shifted_matrix.*displacement_matrix);






