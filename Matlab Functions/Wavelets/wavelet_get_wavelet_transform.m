function y = wavelet_get_wavelet_transform(vec_in, minimum_scale, transform_direction, options)

% perform_wavelet_transform - wrapper to several wavelet transform.
%
%   y = perform_wavelet_transform(x, Jmin, dir, options);
%
%   For the orthogonal transform, this function makes use of wavelab.
%   For the translation invariant, it tries to uses a lifted transform (for
%   5-3 and 7-9 transform), otherwise it tries to use the Rice Wavelet
%   Toolbox (periodic boundary conditions).
%
%   'x' is either a 1D or a 2D array.
%   'Jmin' is the minimum scale (i.e. the coarse channel is of size 2^Jmin
%       in 1D).
%   'dir' is +1 for fwd transform and -1 for bwd.
%   'options.wavelet_vm' is the number of Vanishing moment (both for primal and dual).
%   'options.wavelet_type' can be
%       'daubechies', 'symmlet', 'battle', 'biorthogonal'.
%   Set options.ti=1 for translation invariant transform (better for denoising).
%
%   Typical use :
%       M = <load your image here>;
%       Jmin = 4;
%       options.wavelet_type = 'biorthogonal_swapped';
%       options.wavelet_vm = 4;
%       MW = perform_wavelet_transform(M, Jmin, +1, options);
%       Mt = <perform some modification on MW>
%       M = perform_wavelet_transform(Mt, Jmin, -1, options);
%
%   'y' is an array of the same size as 'x'. This means that for the 2D
%   we are stuck to the wavelab coding style, i.e. the result
%   of each transform is an array organized using Mallat's ordering
%   (whereas Matlab official toolbox use a 1D ordering for the 2D transform).
%
%   Here the transform automaticaly select symmetric boundary condition
%   if you use a symmetric filter. If your filter is not symmetric
%   (e.g. Dauechies filters) then as the output must have same length
%   as the input, the boundary condition are automatically set to periodic.
%
%   For options.ti=0, y has the same size as x.
%   For options.ti=1, y is a cell array with y{i} having same size as x.
%
%   Copyright (c) 2008 Gabriel Peyre

%
%   You do not need Wavelab to use this function (the Wavelab .m file are
%   included in this script). However, for faster execution time, you
%   should install the mex file within the Wavelab distribution.
%       http://www-stat.stanford.edu/~wavelab/

if nargin<3
    transform_direction = 1;
end
if nargin<2
    minimum_scale = 3;
end

options.null = 0;
wavelet_type = getoptions(options, 'wavelet_type', 'biorthogonal');
number_of_vanishing_moments = getoptions(options, 'wavelet_vm', 4);
flag_translation_invariant = getoptions(options, 'ti', 0);
flag_use_mex = getoptions(options, 'use_mex', 1);

if iscell(vec_in)
    number_of_dimensions = get_actual_number_of_dimensions(vec_in{1});
else
    number_of_dimensions = get_actual_number_of_dimensions(vec_in);
end

%for color images:
if number_of_dimensions==3 && size(vec_in,3)<=4
    y = vec_in;
    for channel_counter = 1:size(vec_in,3)
        y(:,:,channel_counter) = wavelet_get_wavelet_transform(vec_in(:,:,channel_counter), minimum_scale, transform_direction, options);
    end
    return;
end


%generate filters:
dual_quadrature_mirror_filter = [];
switch lower(wavelet_type)
    case 'daubechies'
        quadrature_mirror_filter = wavelet_get_orthonormal_QMF_filter('Daubechies',number_of_vanishing_moments*2);  % in Wavelab, 2nd argument is VM*2 for Daubechies... no comment ...
    case 'haar'
        quadrature_mirror_filter = wavelet_get_orthonormal_QMF_filter('Haar');  % in Wavelab, 2nd argument is VM*2 for Daubechies... no comment ...
    case 'symmlet'
        quadrature_mirror_filter = wavelet_get_orthonormal_QMF_filter('Symmlet',number_of_vanishing_moments);
    case 'battle'
        quadrature_mirror_filter = wavelet_get_orthonormal_QMF_filter('Battle',number_of_vanishing_moments-1);
        dual_quadrature_mirror_filter = quadrature_mirror_filter; % we need dual filter
    case 'biorthogonal'
        [quadrature_mirror_filter,dual_quadrature_mirror_filter] = wavelet_get_biorthonormal_QMF_filter_pair( 'CDF', [number_of_vanishing_moments,number_of_vanishing_moments] );
    case 'biorthogonal_swapped'
        [dual_quadrature_mirror_filter,quadrature_mirror_filter] = wavelet_get_biorthonormal_QMF_filter_pair( 'CDF', [number_of_vanishing_moments,number_of_vanishing_moments] );
    otherwise
        error('Unknown transform.');
end

% translation invariant transform:
if flag_translation_invariant==1
    %%%% Mex LIW implementation %%%%
    if flag_use_mex && exist('cwpt2_interface') && number_of_dimensions==2 && ...
            ( strcmp(wavelet_type, 'biorthogonal') || strcmp(wavelet_type, 'biorthogonal_swapped') )
        y = interface_liw(vec_in, minimum_scale, options);
        return;
    end
    %%%% Mex RWT implementation %%%%
    if flag_use_mex && exist('mirdwt') && exist('mrdwt')
        y = interface_rwt(vec_in, minimum_scale, quadrature_mirror_filter, dual_quadrature_mirror_filter);
        return;
    end
    %%%% Lifting implementation %%%%
    if strcmp(wavelet_type, 'biorthogonal') || strcmp(wavelet_type, 'biorthogonal_swapped')
        y = wavelet_interface_lifting(vec_in, minimum_scale, number_of_vanishing_moments);
        return;
    end    
    %%%% Wavelab implementation %%%%%
    if isempty(dual_quadrature_mirror_filter)
        y = interface_wavelab_ti(vec_in, minimum_scale, quadrature_mirror_filter);
        return;
    end
    error('This TI transform is not implemented.');
end

% perform transform
if ~exist('dqmf') || isempty(dual_quadrature_mirror_filter)
    %%% ORTHOGONAL %%%
    if number_of_dimensions==1
        if transform_direction==1
            y = wavelet_1D_transform_orthogonal(vec_in,minimum_scale,quadrature_mirror_filter);
        else
            y = wavelet_1D_inverse_transform_orthogonal(vec_in,minimum_scale,quadrature_mirror_filter);
        end
    elseif number_of_dimensions==2
        if transform_direction==1
            y = wavelet_2D_transform_orthogonal(vec_in,minimum_scale,quadrature_mirror_filter);
        else
            y = wavelet_2D_inverse_transform_orthogonal(vec_in,minimum_scale,quadrature_mirror_filter);
        end
    end
else
    %%% BI-ORTHOGONAL %%%
    if number_of_dimensions==1
        if transform_direction==1
            y = wavelet_1D_transform_symmetric_extension_biorthogonal(vec_in,minimum_scale,quadrature_mirror_filter,dual_quadrature_mirror_filter);
        else
            y = wavelet_1D_inverse_transform_symmetric_extenstion_biorthogonal(vec_in,minimum_scale,quadrature_mirror_filter,dual_quadrature_mirror_filter);
        end
    elseif number_of_dimensions==2
        if transform_direction==1
            y = wavelet_2D_transform_symmetric_extention_biorthogonal(vec_in,minimum_scale,quadrature_mirror_filter,dual_quadrature_mirror_filter);
        else
            y = wavelet_2D_inverse_transform_symmetric_extension_biorthogonal(vec_in,minimum_scale,quadrature_mirror_filter,dual_quadrature_mirror_filter);
        end
    end
end








%RWT interface:
function y = interface_rwt(x, minimum_scale, QMF, dual_QMF)

if isempty(dual_QMF)
    dual_QMF = QMF;
end
if iscell(x)
    ndim = get_actual_number_of_dimensions(x{1});
else
    ndim = get_actual_number_of_dimensions(x);
end

%%% USING RWT %%%
if ~iscell(x)
    signal_length = length(x);
    max_scale = log2(signal_length)-1;
    %%% FORWARD TRANSFORM %%%
    L = max_scale-minimum_scale+1;
    [yl,yh,L] = mrdwt(x, QMF, L);
    % turn into cell array
    if ndim==1
        y = cat(2,yl,yh);
        y = transform_matrix_into_cell_array(y);
    else
        %% 2D %%
        for j=max_scale:-1:minimum_scale
            for q=1:3
                s = 3*(max_scale-j)+q-1;
                M = yh(:,s*signal_length+1:(s+1)*signal_length);
                y{ 3*(j-minimum_scale)+q } = M;
            end
        end
        y{ 3*(max_scale-minimum_scale)+4 } = yl;
    end
    % reverse the order of the frequencies
    y = { y{end-1:-1:1} y{end} };
else %if X is a cell:
    signal_length = length(x{1});
    max_scale = log2(signal_length)-1;
    L = max_scale-minimum_scale+1;
    % reverse the order of the frequencies
    x = { x{end-1:-1:1} x{end} };    
    % turn into array
    if ndim==1
        %% 1D %%
        x = transform_matrix_into_cell_array(x);
        yl = x(:,1);
        yh = x(:,2:end);
    else
        %% 2D %%
        if L ~= (length(x)-1)/3
            warning('Jmin is not correct.');
            L = (length(x)-1)/3;
        end
        yl = x{ 3*(max_scale-minimum_scale)+4 };
        yh = zeros( signal_length,3*L*signal_length );
        for j=max_scale:-1:minimum_scale
            for q=1:3
                s = 3*(max_scale-j)+q-1;
                yh(:,s*signal_length+1:(s+1)*signal_length) = x{ 3*(j-minimum_scale)+q };
            end
        end
    end
    
    %%% BACKWARD TRANSFORM %%%
    [y,L] = mirdwt(yl,yh,dual_QMF,L);
end



% LetIt wave interface
function y = interface_liw(x, minimum_scale, options)

% either 'quad' or 'tri'
decomp_type = getoptions(options, 'decomp_type', 'quad');
wavelet_types = '7-9';
% wavelet_types = 'spline';

if ~iscell(x)
    dirs = 'forward';
    max_scale = log2(size(x,1)) - minimum_scale;
    M = x;
else
    dirs = 'inverse';
    if strcmp(decomp_type, 'tri')
        x = { x{1:end-3}, x{end}, x{end-2:end-1} };
    else
        x = { x{1:end-4}, x{end}, x{end-3:end-1} };
    end
    max_scale = 0;
    M = zeros(size(x{1},1), size(x{1},2), max_scale);
    for i=1:length(x)
        M(:,:,i) = x{i};
    end
end

y = cwpt2_interface(M, dirs, wavelet_types, decomp_type, max_scale);

if ~iscell(x)
    M = y; y = {};
    for i=1:size(M,3)
        y{i} = M(:,:,i);
    end
    % put low frequency at the end
    if strcmp(decomp_type, 'tri')
        y = { y{1:end-3}, y{end-1:end}, y{end-2} };
    else
        y = { y{1:end-4}, y{end-2:end}, y{end-3} };
    end
end



function y = wavelet_interface_wavelab_ti(x, Jmin, qmf)
% wavelab TI interface

if iscell(x)
    dir = -1; 
    ndim = get_actual_number_of_dimensions(x{1});
else
    dir = 1; 
    ndim = get_actual_number_of_dimensions(x);
end

if ndim==1
    if dir==1
        y = wavelet_convert_translation_invariant_to_stationary_transform( wavelet_translation_invariant_forward_transform(x,Jmin,qmf) );
        %transform into cell array:
        y = transform_matrix_into_cell_array(y);
    else
        x = transform_matrix_into_cell_array(x);
        y = wavelet_invert_translation_invariant_wavelet_transform( ...
                                    wavelet_convert_stationary_to_translation_invariant_transform(x),qmf);
        y = y(:);
    end
elseif ndim==2
    if dir==1
        y = wavelet_2D_translation_invariant_forward_transform(x,Jmin,qmf);
        n = size(x,1);
        y = reshape( y, [n size(y,1)/n n] );
        y = permute(y, [1 3 2]);
        % transform into cell array
        y = transform_matrix_into_cell_array(y);
    else
        x = transform_matrix_into_cell_array(x);
        x = permute(x, [1 3 2]);
        x = reshape( x, [size(x,1)*size(x,2) size(x,3)] );
        y = wavelet_2D_invert_translation_invariant_wavelet_transform(x,Jmin,qmf);
    end
end



function y = wavelet_interface_lifting(vec_in, Jmin, number_of_vanishing_moments)
% lifting interface

if number_of_vanishing_moments==2
    options.filter = 'linear';
else
    options.filter = '9-7';
end
options.ti = 1;

direction = 1;
if iscell(vec_in)
    direction = -1;
end

if direction<0
    vec_in = { vec_in{end} vec_in{end-1:-1:1} };
    vec_in = transform_matrix_into_cell_array(vec_in);
    if size(vec_in,3)==1
        % 1D
        vec_in = reshape(vec_in, [size(vec_in,1) 1 size(vec_in,2)]);
    end
end
y = perform_lifting_transform(vec_in, Jmin, direction, options);
if direction>0
    y = transform_matrix_into_cell_array(squeeze(y));
    % reverse frequencies
    y = { y{end:-1:2} y{1} };
end



