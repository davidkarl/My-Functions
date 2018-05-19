function [lens,PSF,x_focal,y_focal] = get_aberrated_lens(mode_number_array,mode_coefficient_array,lambda,N,lens_spacing,lens_diameter,focal_length)

% mode_number_array contains the mode numbers for the corresponding mode_coefficient_array coefficients

% for instance mode_number_array=[1,3,5] and mode_coefficient_array=[0.2,0.11,1.3] means i want my phase
% aberrations to be:   W(x,y)=sum( mode_coefficient_array(i) * zernike(mode_number_array(i) )

W=zeros(N,N);
for k=1:length(mode_number_array)
   W=W+mode_coefficient_array(k)*zernike(mode_number_array(k),N,lens_spacing,lens_diameter);
end
% imagesc(abs(W));
 
 
x=[-N/2:N/2-1]*lens_spacing;
[X_lens,Y_lens]=meshgrid(x);
[lens_clean]=lens_phase_mask(lambda,lens_diameter,focal_length,1,X_lens,Y_lens);
% imagesc(unwrap(angle(lens)));

lens=lens_clean.*exp(-1i*2*pi*W);
% imagesc(unwrap(angle(lens)));

%get PSF on focal plane:
size_factor = 100;
diffraction_limited_diameter = focal_length * lambda / lens_diameter; 
final_image_size = diffraction_limited_diameter * size_factor;
image_spacing = final_image_size/N;
[PSF,x_focal,y_focal]=angular_spectrum_propagation(lens,lambda,lens_spacing,image_spacing,focal_length);
% [PSF_clean,x_focal,y_focal]=angular_spectrum_propagation(lens_clean,lambda,lens_spacing,image_spacing,focal_length);

%NORMALIZE PSF(?????):
PSF=PSF/sqrt(sum(sum(abs(PSF).^2)));

% figure(1)
% plot_beam(PSF_clean,x_focal);
% figure(2)
% plot_beam(PSF,x_focal);





