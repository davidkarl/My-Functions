function [orientation, unwrapped_phase, amplitude] = riesz_transform_for_image_subband(image_subband)

[subband_height, subband_width] = size(image_subband);

%approx riesz, from Riesz Pyramids for Fast Phase-Based Video Magnification
dxkernel = zeros(size(image_subband));
dxkernel(1, 2) = -0.5;
dxkernel(1,subband_width) = 0.5;

dykernel = zeros(size(image_subband));
dykernel(2, 1) = -0.5;
dykernel(subband_height, 1) = 0.5; 

R1 = ifft2(fft2(image_subband) .* fft2(dxkernel));
R2 = ifft2(fft2(image_subband) .* fft2(dykernel));
 
%Does this type of initialization help with speed? check!!!:
orientation = zeros(subband_height, subband_width);
phase = zeros(subband_height, subband_width);

orientation = (atan2(-R2, R1));
phase = atan2( sqrt(R1.^2+R2.^2), image_subband );
unwrapped_phase = unwrap(phase);
amplitude = sqrt(image_subband.^2 + R1.^2 + R2.^2);

end  