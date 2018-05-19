function y = PDTDFB_decomposition(mat_in, ...
    number_of_directional_filters_levels_at_each_scale_vec, ...
    laplacian_lowpass_filter_name, ...
    diamond_or_fan_filter_name, ...
    wavelet_filter_name, ...
    flag_residual_high_pass_band)
% PDTDFBDEC   Pyramidal Dual Tree Directional Filter Bank Decomposition
%
%	y = pdtdfbdec(x, nlevs, [lpfname], [dfname], [wfname], [res])
%
% Input:
%   x:      Input image
%   nlevs:  vector of numbers of directional filter bank decomposition levels
%           at each pyramidal level (from coarse to fine scale).
%           0 : Laplacian pyramid level
%           1 : Wavelet decomposition
%           n : 2^n directional PDTDFB decomposition
%   lpfname : [optional] The name of the laplacian lowpass filter that is used,
%           default is 'nalias'
%           if lpfname is 'nalias' then the lpftype = 2
%   dfname  :[optional] The name of the diamond or fan filter that is used,
%           'meyer': default (very big filters - very slow)
%           'pkva''pkva6', 'pkva8', 'pkva12' (Contourlet toolbox, using ladder structure)
%
%   wfname: [optional] The name of the wavelet filter that is used, default is '9-7'
%   res :   [optional] Residual high passband, default 1
%           0 is no, 1 is yes
%   mode :
%
% Output:
%   y:      a cell vector of length length(nlevs) + 1, where except y{1} is
%           the lowpass subband, each cell corresponds to one pyramidal
%           level and is a cell vector that contains bandpass directional
%           subbands from the DFB at that level.
%
% Call function: LPDEC,
% See also:	PDTDFBREC, PDFBDEC, TDFBDEC,
%
% Note : PDTDFB data structure y{resolution}{1}{1-2^n} : primal branch
%                              y{resolution}{2}{1-2^n} : dual branch

% running time for 1024 image size [3]: 31 sec
%                                  [5]: 33 sec
%                             [4 7 7 5]  : 41 sec
%

if ~exist('flag_residual_high_pass_band','var')
    flag_residual_high_pass_band = 0 ; % default implementation have a highpass residual band
end

if ~exist('laplacian_lowpass_filter_name','var')
    laplacian_lowpass_filter_name = 'nalias' ; % default implementation by meyer type FB
end

if ~exist('diamond_or_fan_filter_name','var')
    diamond_or_fan_filter_name = 'meyer' ; % default implementation by meyer type FB
    % Note: meyer diamond is different from meyer type low pass
end

if ~exist('wavelet_filter_name','var')
    wavelet_filter_name = '9-7' ; % default implementation by meyer type FB
end

if ~exist('mode','var')
    mode = 'pdtdfb' ; % default implementation is PDTDFB decomposition
end


if length(number_of_directional_filters_levels_at_each_scale_vec) == 0
    
    y = {mat_in};
    
else %----------------------------------------------------------------
    
    if flag_residual_high_pass_band % remove the highpass component to residual image
        N = 128;
        cutoff = 2*pi;
        %cutoff =    pi;
        rev = (8*pi^2/3)/cutoff;
        xa = 0 : rev/(N) : rev;
        %Compute support of Fourier transform of phi.
        int1 = find((xa < 2*pi/3));
        int2 = find((xa >= 2*pi/3) & (xa < 4*pi/3));
        
        %Compute Fourier transform of phi:
        phihat = zeros(1,N+1);
        phihat(int1) = ones(size(int1));
        phihat(int2) = cos(pi/2*meyeraux(3/2/pi*xa(int2)-1));
        
        phihat = [phihat,phihat((end-1):-1:2)];
        psihat = (1-phihat).^(1/2);
        h = ifftshift(ifft(phihat));
        g = ifftshift(ifft(psihat));
        
        % [h,g] = wfilters('dmey','l');
        h = fit_matrix_dimensions_to_certain_size(h,[1,32]);
        h = h(2:end);
        h = h./sum(h);
        H = h'*h;
        G = fftshift(ifft2( (1-abs(fft2(H)).^2).^(1/2) ) );
        G = fit_matrix_dimensions_to_certain_size(real(G),31);
        x_res = filter_2D_with_edge_handling(mat_in,G,'sym');
        mat_in = filter_2D_with_edge_handling(mat_in,H,'sym');
    end
    
    %determine the kind of laplacian pyramid decomposition based on
    %lpfname decide type of pyramidal decomposition
    switch lower(laplacian_lowpass_filter_name)
        case ('special')
            laplacian_pyramid_type = 3; %
            disp('Frequency implemenation for the first two level')
        case ('nalias')
            laplacian_pyramid_type = 2; % noaliasing type pyramid
            disp('Noalias frame pyramid')
        case ('fp')
            laplacian_pyramid_type = 0; % Burt and Aldeson pyramid
            disp('framming pyramid')
        otherwise
            laplacian_pyramid_type = 1; % Burt and Aldeson pyramid
            disp('Burt and Aldeson pyramid')
    end
    
    switch number_of_directional_filters_levels_at_each_scale_vec(end)
        case {0}
            %Get the pyramidal filters:
            [h,g] = get_filters_for_laplacian_pyramid(laplacian_lowpass_filter_name);
            
            %keep the laplacian highpass:
            [xlo, xhi] = laplacian_pyramid_decomposition(mat_in,h,g,laplacian_pyramid_type);
            xhi_dir = xhi;
        case {1}
            %Get the pyramidal filters:
            [h,g] = get_filters_for_laplacian_pyramid(laplacian_lowpass_filter_name);
            %wavelet decomposittion
            [h, g] = get_filters_for_laplacian_pyramid(wavelet_filter_name);
            
            [xlo, xLH, xHL, xHH] = wavelet_filterbank_2D_decomposition(mat_in, h, g);
            xhi_dir = {xLH, xHL, xHH};
        otherwise
            if (laplacian_pyramid_type~=3)
                % laplacian pyramid
                % Get the pyramidal filters
                [h,g] = get_filters_for_laplacian_pyramid(laplacian_lowpass_filter_name);
                [xlo, xhi] = laplacian_pyramid_decomposition(mat_in,h,g,laplacian_pyramid_type);
                
                % DFB on the bandpass image
                switch diamond_or_fan_filter_name        % Decide the method based on the filter name
                    case {'pkva6', 'pkva8', 'pkva12', 'pkva'}
                        % Use the ladder structure (whihc is much more efficient)
                        % disp('ladder')
                        % dual tree dfb
                        tmp = DFB_decomposition_time_domain_ladder_structure(xhi, number_of_directional_filters_levels_at_each_scale_vec(end),'primal', diamond_or_fan_filter_name);
                        xhi_dir{1} = tmp;
                        tmp = DFB_decomposition_time_domain_ladder_structure(xhi, number_of_directional_filters_levels_at_each_scale_vec(end),'dual', diamond_or_fan_filter_name);
                        xhi_dir{2} = tmp;
                    otherwise
                        % General case
                        % dual tree dfb
                        tmp = DFB_decomposition_time_domain(xhi, number_of_directional_filters_levels_at_each_scale_vec(end),'primal', diamond_or_fan_filter_name);
                        xhi_dir{1} = tmp;
                        tmp = DFB_decomposition_time_domain(xhi, number_of_directional_filters_levels_at_each_scale_vec(end),'dual', diamond_or_fan_filter_name);
                        xhi_dir{2} = tmp;
                end
                
            else
                % frequency implementation
                
                alpha = 0.15;
                
                s = size(mat_in);
                
                % create the grid and transform to circle grid
                S1 = -1.5*pi:pi/(s(1)/2):1.5*pi;
                S2 = -1.5*pi:pi/(s(2)/2):1.5*pi;
                [x1, x2] = meshgrid(S2,S1);
                r = [0.4 0.5 1-alpha, 1+ alpha];
                
                [x1, x2] = tran_sf(x1,x2); %????? no such function
                
                rd = sqrt(x1.*x1+x2.*x2);
                theta = angle(x1+sqrt(-1)*x2);
                
                % Low pass window
                sz = size(rd);
                cen = rd((sz(1)+1)/2,:);
                cen = abs(cen);
                fl = get_1D_meyer_function(cen,pi*[-2 -1 r(1:2)]);
                FL =  fl'*fl;
                
                % high pass window
                ang = pi/4*[-alpha alpha];
                ang = [-pi/4+ang(1:2), 3*pi/4+ang(1:2)];
                
                f3 = fun_meyer_curv(rd, theta, pi*r, ang, 's'); %??????? no such function
                f3 = periodize_2D(f3,s, 's');
                
                % take out the center and square root
                FL = sqrt(fftshift(FL(s(1)/4+1:s(1)/4+s(1),s(2)/4+1:s(2)/4+s(2))));
                f3 = sqrt(fftshift(f3(s(1)/4+1:s(1)/4+s(1),s(2)/4+1:s(2)/4+s(2))));
                
                % actual transform by FFT
                imf = fft2(mat_in);
                sz = s/2;
                dm = [2 2];
                
                imf2 = (1/prod(dm))*periodize_2D(imf.*FL, sz);
                
                xlo = ifft2(imf2(1:sz(1),1:sz(2)));
                xhi = ifft2(imf.*f3);
                
                %now doing normal DFB on both branches
                tmp = DFB_decomposition_time_domain_ladder_structure(real(xhi), number_of_directional_filters_levels_at_each_scale_vec(end),'primal', diamond_or_fan_filter_name);
                xhi_dir{1} = tmp;
                tmp = DFB_decomposition_time_domain_ladder_structure(imag(xhi), number_of_directional_filters_levels_at_each_scale_vec(end),'primal', diamond_or_fan_filter_name);
                xhi_dir{2} = tmp;
            end
            
    end
    
    %Recursive call on the low band:
    res2 = 0; % from the second level , there is no residual band
    ylo = PDTDFB_decomposition(xlo, number_of_directional_filters_levels_at_each_scale_vec(1:end-1), laplacian_lowpass_filter_name, diamond_or_fan_filter_name, wavelet_filter_name, res2);
    
    %Add bandpass directional subbands to the final output:
    y = {ylo{:}, xhi_dir};
    
    if flag_residual_high_pass_band % add back the residual band to the end of xhi_dir
        y{end+1} = x_res;
    end
end

