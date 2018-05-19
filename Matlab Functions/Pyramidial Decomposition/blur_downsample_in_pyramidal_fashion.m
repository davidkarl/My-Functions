function res = blur_downsample_in_pyramidal_fashion(mat_in, number_of_levels, filter_mat)
% RES = blurDn(IM, LEVELS, FILT)
%
% Blur and downsample an image.  The blurring is done with filter
% kernel specified by FILT (default = 'binom5'), which can be a string
% (to be passed to namedFilter), a vector (applied separably as a 1D
% convolution kernel in X and Y), or a matrix (applied as a 2D
% convolution kernel).  The downsampling is always by 2 in each
% direction.
%
% The procedure is applied recursively LEVELS times (default=1).

%------------------------------------------------------------
% OPTIONAL ARGS:

if ~exist('number_of_levels','var')
    number_of_levels = 1;
end

if ~exist('filter_mat','var')
    filter_mat = 'binom5';
end

%------------------------------------------------------------

if ischar(filter_mat)
    filter_mat = get_filter_by_name(filter_mat);
end

filter_mat = filter_mat/sum(filter_mat(:));

if number_of_levels > 1
    mat_in = blur_downsample_in_pyramidal_fashion(mat_in,number_of_levels-1,filter_mat);
end

if number_of_levels >= 1
    
    if any(size(mat_in)==1)
        %if mat_in is 1D check filter is too:
        if ~any(size(filter_mat)==1)
            error('Cant apply 2D filter to 1D signal');
        end
        
        %match filter vec orientation to mat in:
        if (size(mat_in,2)==1)
            filter_mat = filter_mat(:);
        else
            filter_mat = filter_mat(:)';
        end
        
        %convolve and downsample:
        res = corr2_downsample(mat_in,filter_mat,'reflect1',(size(mat_in)~=1)+1);
    
    elseif any(size(filter_mat)==1)
        %mat_in is 2D but filter is 1D then operate on each dimension seperately:
        filter_mat = filter_mat(:);
        res = corr2_downsample(mat_in,filter_mat,'reflect1',[2 1]);
        res = corr2_downsample(res,filter_mat','reflect1',[1 2]);
    
    else
        %both mat_in and filter are 2D:
        res = corr2_downsample(mat_in,filter_mat,'reflect1',[2 2]);
    end
else
    res = mat_in;
end
