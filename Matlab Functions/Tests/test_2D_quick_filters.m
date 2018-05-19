%test 2D quick filters:

% Initialize filter.
filter_size = 100;
filter_order = 4;
filter_parameter = 200;
%set up grid (including dummy check-up variables):
[X,Y] = meshgrid(1:filter_size); 
filter_center = ceil(filter_size/2) + (1-mod(filter_size,2));
filter_end = floor((filter_size-1)/2);
filter_start = -ceil((filter_size-1)/2);
filter_length = filter_end-filter_start+1;
%get cutoffs:
low_cutoff = 10;
high_cutoff = 20;
%set up filter grid:
distance_from_center  = sqrt((X-filter_center).^2 + (Y-filter_center).^2);
max_distance = max(distance_from_center(:));
min_distance = min(distance_from_center(:));
% filter_low_pass = 1./(1 + (distance_from_center/low_cutoff).^(2*filter_order));
% filter_high_pass = 1 - 1./(1 + (distance_from_center/high_cutoff).^(2*filter_order));
filter_low_pass_real_space = 1/besseli(0,pi*filter_parameter) * besseli(0,pi*filter_parameter*sqrt(1-(distance_from_center/filter_size).^2));
filter_bandpass = filter_low_pass .* filter_high_pass;
filter_bandstop = 1 - filter_bandpass;
subplot(2,1,1)
mesh(filter_low_pass_real_space);
subplot(2,1,2)
imagesc(abs(ift2(filter_low_pass_real_space,1)));
colorbar;  


subplot(2,1,1) 
mesh(filter_low_pass_real_space);
subplot(2,1,2)
imagesc(abs(ft2(filter_low_pass_real_space,1)));
title('Filter Image')
filtered_image = ifftshift(filtered_image);
filtered_image = ifft2(filtered_image,2*nx-1,2*ny-1);
filtered_image = real(filtered_image(1:nx,1:ny));
filtered_image = uint8(filtered_image);

subplot(2,2,4)
imshow(filtered_image,[])
title('Filtered Image')



