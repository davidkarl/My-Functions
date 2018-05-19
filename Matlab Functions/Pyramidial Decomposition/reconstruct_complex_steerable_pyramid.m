function res = reconstruct_complex_steerable_pyramid(pyramid, index_matrix, levels, bands, transition_width)
% RES = reconSCFpyr(PYR, INDICES, LEVS, BANDS, TWIDTH)
%
% The inverse of buildSCFpyr: Reconstruct image from its complex steerable pyramid representation,
% in the Fourier domain.
%
% The image is reconstructed by forcing the complex subbands to be analytic
% (zero on half of the 2D Fourier plane, as they are supossed to be unless
% they have being modified), and reconstructing from the real part of those
% analytic subbands. That is equivalent to compute the Hilbert transforms of
% the imaginary parts of the subbands, average them with their real
% counterparts, and then reconstructing from the resulting real subbands.
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.
%
% LEVS (optional) should be a list of levels to include, or the string
% 'all' (default).  0 corresonds to the residual highpass subband.
% 1 corresponds to the finest oriented scale.  The lowpass band
% corresponds to number spyrHt(INDICES)+1.
%
% BANDS (optional) should be a list of bands to include, or the string
% 'all' (default).  1 = vertical, rest proceeding anti-clockwise.
%
% TWIDTH is the width of the transition region of the radial lowpass
% function, in octaves (default = 1, which gives a raised cosine for
% the bandpass filters).

% Javier Portilla, 7/04, basing on Eero Simoncelli's Matlab Pyrtools code
% and our common code on texture synthesis (textureSynthesis.m).

%%------------------------------------------------------------
% DEFAULTS:

if ~exist('levs','var')
    levels = 'all';
end

if ~exist('bands','var')
    bands = 'all';
end

if ~exist('twidth','var')
    transition_width = 1;
elseif (transition_width <= 0)
    fprintf(1,'Warning: TWIDTH must be positive.  Setting to 1.\n');
    transition_width = 1;
end

%%------------------------------------------------------------


pind = index_matrix;
Nsc = log2(pind(1,1)/pind(end,1));
Nor = (size(pind,1)-2)/Nsc;

for nsc = 1:Nsc
    firstBnum = (nsc-1)*Nor+2;
    
    % Re-create analytic subbands
    dims = pind(firstBnum,:);
    ctr = ceil((dims+0.5)/2);
    ang = make_2D_angles_meshgrid(dims, 0, ctr);
    ang(ctr(1),ctr(2)) = -pi/2;
    for nor = 1:Nor,
        nband = (nsc-1)*Nor+nor+1;
        ind = get_pyramid_subband_indices(pind,nband);
        ch = get_pyramid_subband(pyramid, pind, nband);
        ang0 = pi*(nor-1)/Nor;
        xang = mod(ang-ang0+pi, 2*pi) - pi;
        amask = 2*(abs(xang) < pi/2) + (abs(xang) == pi/2);
        amask(ctr(1),ctr(2)) = 1;
        amask(:,1) = 1;
        amask(1,:) = 1;
        amask = fftshift(amask);
        ch = ifft2(amask.*fft2(ch));    % "Analytic" version
        %f = 1.000008;  % With this factor the reconstruction SNR goes up around 6 dB!
        f = 1;
        ch = f*0.5*real(ch); % real part
        pyramid(ind) = ch;
    end     % nor
end         % nsc

res = reconstruct_steerable_pyramid_fourier(pyramid, index_matrix, levels, bands, transition_width);

