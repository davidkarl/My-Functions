function [c, d] = laplacian_pyramid_decomposition(mat_in, ...
                                                  filter1, ...
                                                  filter2, ...
                                                  decomposition_mode, ...
                                                  extention_mode)
% LPDEC   Pyramid Decomposition
%
%	[c, d] = lpdec(x, h, g, opt, mode)
%
% Input:
%   x:      input image
%   h, g:   two one or two-dimesional filters, depend on opt.
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
%   c:      coarse image at half size
%   d:      detail image at full size
%
% See also:	LPREC, PDFBDEC

% Lowpass filter and downsample
if ~exist('extention_mode','var')
    extention_mode = 'per';
end

if ~exist('decomposition_mode')
    decomposition_mode = 0;
end

if decomposition_mode < 2 % h , g is 1-D filter  ----------------------------------------
    xlo = filter_2D_seperable_filtering_with_extension_handling(mat_in, filter1, filter1, extention_mode);
    c = xlo(1:2:end, 1:2:end);

    % Compute the residual (bandpass) image by upsample, filter, and subtract
    % Even size filter needs to be adjusted to obtain perfect reconstruction
    adjust = mod(length(filter2) + 1, 2);

    xlo = zeros(size(mat_in));
    xlo(1:2:end, 1:2:end) = c;
    d = mat_in - filter_2D_seperable_filtering_with_extension_handling(xlo, filter2, filter2, extention_mode, adjust * [1, 1]);
else % h , g is 2-D filter  ----------------------------------------
    % filtered
    d = filter_2D_with_edge_handling(mat_in, filter2,'sym');
    x_l = filter_2D_with_edge_handling(mat_in, 2*filter1,'sym');

    % decimation
    c = x_l(1:2:end,1:2:end);

end


