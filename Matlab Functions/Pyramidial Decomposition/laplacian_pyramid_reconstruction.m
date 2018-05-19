function x = laplacian_pyramid_reconstruction(coarse_image_half_size, ...
                                              detail_image_full_size, ...
                                              filter1, ...
                                              filter2, ...
                                              decomposition_mode, ...
                                              extention_mode)
% LPDEC   Pyramid Reconstruction
%
%	x = lprec(c, d, h, g, opt,mode)
%
% Input:
%   c:      coarse image at half size
%   d:      detail image at full size
%   h, g:   two one or two-dimesional  filter, depend on opt
%   opt :   Parameter define the mode of decomposition: 
%           0 : default, reconstructed by Do and Vetterli method.
%               See 'Framing Pyramids'
%           1 : reconstructed LP by the conventional (Burt-Andelson ) method
%               Not a tight frame reconstruction. See EUSIPCO 06 'On Aliasing ....'
%               h and g are 1-D filters
%           2 : no aliasing method, the lowpass filter h is nyquist 2  
%               g is the highpass filter, 0.25*h(w)^2+g(w)^2 = 1
%               h and g are 2-D filters
%
%   mod :   Optional : 'sym' and 'per' specify the extension mode of the
%           low pass band
%
% Output:
%   x:      reconstructed image
%
% See also:	LPDEC, PDFBREC
%

if ~exist('decomposition_mode')
    decomposition_mode = 0;
end

% mode = 'per';
if ~exist('extention_mode','var')
    extention_mode = 'per';
end

if decomposition_mode < 2 % h , g is 1-D filter  ----------------------------------------
    if decomposition_mode==1 % opt = 1 LP Burt-Andelson
        x_detail = zeros(size(coarse_image_half_size));
    else % opt = 0 LP Framing Pyramid
        % First, filter and downsample the detail image
        x_detail = filter_2D_seperable_filtering_with_extension_handling(detail_image_full_size, ...
                                                                         filter1, ...
                                                                         filter1, ...
                                                                         extention_mode);
        x_detail = x_detail(1:2:end, 1:2:end);
    end

    %Subtract from the coarse image, and then upsample and filter (LAPLACIAN):
    x_coarse_minus_detail = coarse_image_half_size - x_detail;
    x_coarse_minus_detail = diagonal_upsampling(x_coarse_minus_detail, [2, 2]);

    %Even size filter needs to be adjusted to obtain perfect reconstruction with zero shift
    adjust = mod(length(filter2) + 1, 2);
    
    x_coarse_minus_detail = filter_2D_seperable_filtering_with_extension_handling(x_coarse_minus_detail, ...
                                                                                  filter2, ...
                                                                                  filter2, ...
                                                                                  extention_mode, ...
                                                                                  adjust * [1, 1]);
    
    %Final combination:
    x = x_coarse_minus_detail + detail_image_full_size;

else % h , g is 2-D filter  ----------------------------------------
    % filtered
    x_u = kron(coarse_image_half_size, [1 0 ; 0 0]);
    x = filter_2D_with_edge_handling(detail_image_full_size, filter2,'sym') + ...
        filter_2D_with_edge_handling(x_u, 2*filter1,'sym');
end
