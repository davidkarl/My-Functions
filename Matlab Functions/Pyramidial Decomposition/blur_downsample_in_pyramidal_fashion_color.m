function out = blur_downsample_in_pyramidal_fashion_color(mat_in, number_of_levels, filt)
% 3-color version of blurDn.

%------------------------------------------------------------
% OPTIONAL ARGS:

if (exist('nlevs') ~= 1) 
  number_of_levels = 1;
end

if (exist('filt') ~= 1) 
  filt = 'binom5';
end

%------------------------------------------------------------

tmp = blur_downsample_in_pyramidal_fashion(mat_in(:,:,1), number_of_levels, filt);
out = zeros(size(tmp,1), size(tmp,2), size(mat_in,3));
out(:,:,1) = tmp;
for color_counter = 2:size(mat_in,3)
  out(:,:,color_counter) = ...
      blur_downsample_in_pyramidal_fashion(mat_in(:,:,color_counter), number_of_levels, filt);
end
