function out = upsample_inserting_zeros_blur_pyramid_color(mat_in, number_of_levels, filt)
% 3-color version upBlur

%------------------------------------------------------------
% OPTIONAL ARGS:

if ~exist('number_of_levels','var')
    number_of_levels = 1;
end

if ~exist('filt','var')
    filt = 'binom5';
end

%------------------------------------------------------------

tmp = upsample_inserting_zeros_blur_pyramid(mat_in(:,:,1), number_of_levels, filt);
out = zeros(size(tmp,1), size(tmp,2), size(mat_in,3));
out(:,:,1) = tmp;
for color_counter = 2:size(mat_in,3)
    out(:,:,color_counter) = ...
                    upsample_inserting_zeros_blur_pyramid(mat_in(:,:,color_counter), number_of_levels, filt);
end
