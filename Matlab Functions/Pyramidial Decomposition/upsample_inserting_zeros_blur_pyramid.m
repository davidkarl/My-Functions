function [mat_filtered] = upsample_inserting_zeros_blur_pyramid(mat_in, number_of_levels, filter_mat_or_name)
% RES = upBlur(IM, LEVELS, FILT)
%
% Upsample and blur an image.  The blurring is done with filter
% kernel specified by FILT (default = 'binom5'), which can be a string
% (to be passed to namedFilter), a vector (applied separably as a 1D
% convolution kernel in X and Y), or a matrix (applied as a 2D
% convolution kernel).  The downsampling is always by 2 in each
% direction.
%
% The procedure is applied recursively LEVELS times (default=1).



%------------------------------------------------------------
%OPTIONAL ARGS:

if ~exist('number_of_levels','var')
  number_of_levels = 1;
end

if ~exist('filter_mat_or_name','var')
  filter_mat_or_name = 'binom5';
end

%------------------------------------------------------------

%Get appropriate filter acocrding to filter name string:
if ischar(filter_mat_or_name)
    filter_mat_or_name = namedFilter(filter_mat_or_name);
end

%If function hasn't reached base level continue to next level:
if number_of_levels > 1
  mat_in = upsample_inserting_zeros_blur_pyramid(mat_in,number_of_levels-1,filter_mat_or_name);
end


if number_of_levels >= 1
    
    %If input mat is 1D assume filter is also 1D just filter it:
    if any(size(mat_in)==1)
        
        if size(mat_in,1)==1
            filter_mat_or_name = filter_mat_or_name';
        end
        mat_filtered = upsample_inserting_zeros_convolve(mat_in,filter_mat_or_name,'reflect1',(size(mat_in)~=1)+1);
    
    elseif any(size(filter_mat_or_name)==1)
    %if input is 2D but filter is 1D convolve along each dimension:
        filter_mat_or_name = filter_mat_or_name(:);
        mat_filtered = upsample_inserting_zeros_convolve(mat_in,filter_mat_or_name,'reflect1',[2 1]);
        mat_filtered = upsample_inserting_zeros_convolve(mat_filtered,filter_mat_or_name','reflect1',[1 2]);
    
    else
    %if input is 2D and filter is 2D just convolve the two:
        mat_filtered = upsample_inserting_zeros_convolve(mat_in,filter_mat_or_name,'reflect1',[2 2]);
    end
    
else
    %If input number_of_levels doesn't make sense (<1) then just return original matrix:
    mat_filtered = mat_in;
end
