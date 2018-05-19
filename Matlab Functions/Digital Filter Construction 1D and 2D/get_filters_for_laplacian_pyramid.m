function [h, g] = get_filters_for_laplacian_pyramid(filter_name)	
% PFILTERS    Generate filters for the Laplacian pyramid
%
%	[h, g] = pfilters(fname)
%
% Input:
%   fname:  Name of the filters, including the famous '9-7' filters
%           and all other available from WFILTERS in Wavelet toolbox
%           Other choices of fname    
%               'Burt' : original Burt
%               'meyer' : two low-pass filter strictly band limited for reduce
%                   aliasing in the normal PDFB (see pdfb)
%               'nalias' : one low and high pass filter satisfy PR for
%                   undecimated FB, the low is strictly lowpass for structure
%                   similar to steerble pyramid
%
% Output:
%   h, g:   1D filters (lowpass for analysis and synthesis, respectively)
%           for seperable pyramid
%           2D filters whent fname = 'nalias'
%
% Note : Based on function of the same name in contourlet toolbox
%


switch filter_name
    case {'9-7', '9/7'}
        h = [.037828455506995 -.023849465019380 -.11062440441842 ...
            .37740285561265];
        h = [h, .85269867900940, fliplr(h)];

        g = [-.064538882628938 -.040689417609558 .41809227322221];
        g = [g, .78848561640566, fliplr(g)];

    case {'5-3', '5/3'}
        h = [-1, 2, 6, 2, -1] / (4 * sqrt(2));
        g = [1, 2, 1] / (2 * sqrt(2));

    case {'burt'}
        h = [0.6, 0.25, -0.05];
        h = sqrt(2) * [h(end:-1:2), h];

        g = [17/28, 73/280, -3/56, -3/280];
        g = sqrt(2) * [g(end:-1:2), g];
    case {'meyer'}
        % h = firpm(20,[0, 0.3 0.5 1],[1 ,1 ,0 ,0]);
        % Calculate the DFT of the diamond meyer fb
        % ------------------------------
        N = 128;
        % cutoff =    1.22*pi;
        cutoff = pi;
        rev = (8*pi^2/3)/cutoff;
        xa = 0:rev/(N):rev;
        % Compute support of Fourier transform of phi.
        int1 = find((xa < 2*pi/3));
        int2 = find((xa >= 2*pi/3) & (xa < 4*pi/3));

        % Compute Fourier transform of phi.
        phihat = zeros(1,N+1);
        phihat(int1) = ones(size(int1));
        phihat(int2) = cos(pi/2*meyeraux(3/2/pi*xa(int2)-1));

        phihat = [phihat,phihat((end-1):-1:2)];
        h = ifftshift(ifft(phihat));

        % [h,g] = wfilters('dmey','l');
        h = fit_matrix_dimensions_to_certain_size(h,[1,42]);
        h = h(2:end);
        g = h(end:-1:1); %g = fitmat(g,[1,42]);
        h = sqrt(2)*h./sum(h);
        g = sqrt(2)*g./sum(g);
    case {'nalias'}
        % create the lowpass no aliasing filter and highpass filter ---------------
        N = 128;
        % cutoff =    2*pi;
        cutoff =  1*pi;
        rev = (8*pi^2/3)/cutoff;
        xa = 0:rev/(N):rev;
        % Compute support of Fourier transform of phi.
        int1 = find((xa < 2*pi/3));
        int2 = find((xa >= 2*pi/3) & (xa < 4*pi/3));

        % Compute Fourier transform of phi.
        phihat = zeros(1,N+1);
        phihat(int1) = ones(size(int1));
        phihat(int2) = cos(pi/2*meyeraux(3/2/pi*xa(int2)-1));

        phihat = [phihat,phihat((end-1):-1:2)];
        psihat = (1-phihat).^(1/2);
        h = ifftshift(ifft(phihat));
        g = ifftshift(ifft(psihat));

        % [h,g] = wfilters('dmey','l');
        h = fitmat(h,[1,62]);
        h = h(2:end);
        h = h./sum(h);
        h = h'*h;
        g = fftshift(ifft2( (1 - abs(fft2(h)).^2).^(1/2) ) );
        g = fit_matrix_dimensions_to_certain_size(real(g),61); 
    otherwise
        %mathworks wavelet filters:
        [h, g] = wfilters(filter_name, 'l');
end

