function y2 = PDTDFB_decomposition_FFT(mat_in, ...
                                       number_of_directional_filters_levels_at_each_scale_vec, ...
                                       F_window, ...
                                       F_alpha_parameter, ...
                                       flag_residual)
% PDTDFBDEC_F   Pyramidal Dual Tree Directional Filter Bank Decomposition
% using FFT at the multresolution FB and the first two level of dual DFB
% tree
%
%	y2 = pdtdfbdec_f(im, nlevs, F2, alpha, res)
%
% Input:
%   x:      Input image
%   nlevs:  vector of numbers of directional filter bank decomposition levels
%           at each pyramidal level (from coarse to fine scale).
%           0 : Laplacian pyramid level
%           1 : Wavelet decomposition
%           n : 2^n directional PDTDFB decomposition
%   F2:     [optional] Precomputed F window
%   alpha:  [optional] Parameter for F2 window, incase F2 is not pre-computed
%   res :   [optional] Boolean value, specify weather existing residual
%           band
% 
% Output:
%   y:      a cell vector of length length(nlevs) + 1, where except y{1} is
%           the lowpass subband, each cell corresponds to one pyramidal
%           level and is a cell vector that contains bandpass directional
%           subbands from the DFB at that level.
%
% See also:	PDTDFBREC_F, PDTDFBDEC
%
% Note : PDTDFB data structure y{resolution}{1}{1-2^n} : primal branch
%                              y{resolution}{2}{1-2^n} : dual branch

% running time for 1024 image size [3]:
%                                  [5]:  sec
%                             [4 7 7 5]  : sec
%

if ~exist('flag_residual','var')
    flag_residual = 0 ; % default implementation
end

mat_in_size = size(mat_in);
L_number_of_levels = length(number_of_directional_filters_levels_at_each_scale_vec);
x = [0 0; ...
    1 1; ...
    0 1; ...
    1 0];

if ~exist('F_alpha_parameter','var')
    F_alpha_parameter = 0.3;
end

if ~exist('F_window','var')
    F_window = get_PDTDFB_frequency_windows(mat_in_size, F_alpha_parameter, L_number_of_levels);
    disp('Precalculated window function will run much faster')
end

%residual band:
mat_in_fft = fft2(mat_in);

if flag_residual
    %residual band:
    mat_in_fft_filtered = mat_in_fft .* F_window{L_number_of_levels+1};
    y2{L_number_of_levels+2} = ifft2(mat_in_fft_filtered);
end

for current_level_counter = L_number_of_levels+1 : -1 : 2
    
    F = F_window{current_level_counter-1};
    
    number_of_directional_filter_current = number_of_directional_filters_levels_at_each_scale_vec(current_level_counter-1);
    
    s = (1/2^(L_number_of_levels+1-current_level_counter)) * mat_in_size;
    [sy, sx] = meshgrid(0:1/s(2):(1-1/s(2)),0:1/s(1):(1-1/s(1)));
    sx = 2*pi*sx;
    sy = 2*pi*sy;
    
    %low pass band:
    mat_in_fft_filtered = mat_in_fft .* F{1};
    imf3 = periodize_2D(mat_in_fft_filtered,s/2);
    mat_in = 0.25*ifft2(imf3);
    %filter 2:
    mat_in_fft_filtered = mat_in_fft.*F{2};
    imf3 = periodize_2D(mat_in_fft_filtered,s/2);
    im1 = 0.25*ifft2(imf3);
    y2{current_level_counter}{1}{1} = real(im1);
    y2{current_level_counter}{2}{1} = imag(im1);
    %filter 3:
    mat_in_fft_filtered = exp(1j*(sx+sy)).*mat_in_fft.*F{3};
    imf3 = periodize_2D(mat_in_fft_filtered,s/2);
    im1 = 0.25*ifft2(imf3);
    y2{current_level_counter}{1}{2} = real(im1);
    y2{current_level_counter}{2}{2} = imag(im1);
    %filter 4:
    mat_in_fft_filtered = exp(1j*(sy)).*mat_in_fft.*F{4};
    imf3 = periodize_2D(mat_in_fft_filtered,s/2);
    im1 = 0.25*ifft2(imf3);
    y2{current_level_counter}{1}{3} = real(im1);
    y2{current_level_counter}{2}{3} = imag(im1);
    %filter 5:
    mat_in_fft_filtered = exp(1j*(sx)).*mat_in_fft.*F{5};
    imf3 = periodize_2D(mat_in_fft_filtered,s/2);
    im1 = 0.25*ifft2(imf3);
    y2{current_level_counter}{1}{4} = real(im1);
    y2{current_level_counter}{2}{4} = imag(im1);
    
    %transform to FFT domain for next level processing
    mat_in_fft = fft2(mat_in);
    
    % --------------------------------------------------------------------
    if (number_of_directional_filter_current>2)
        % Ladder filter
        filter_name = 'pkva';
        if ischar(filter_name)
            f = get_filter_for_ladder_structure_network(filter_name);
        end
        y = y2{current_level_counter}{1};
        % Now expand the rest of the tree
        %(1). primal branch
        for l = 3:number_of_directional_filter_current
            % Allocate space for the new subband outputs
            y_old = y;
            y = cell(1, 2^l);
            
            % The first half channels use R1 and R2
            for k = 1:2^(l-2)
                i = mod(k-1, 2) + 1;
                [y{2*k}, y{2*k-1}] = ...
                    FB_decomposition_ladder_structure(y_old{k}, f, 'p', i, 'per');
            end
            
            % The second half channels use R3 and R4
            for k = 2^(l-2)+1 : 2^(l-1)
                i = mod(k-1, 2) + 3;
                [y{2*k}, y{2*k-1}] = ...
                    FB_decomposition_ladder_structure(y_old{k}, f, 'p', i, 'per');
            end
            
            % circlular shift to make the subband has minimum delay
            for inl = 1 : 4 : 2^(l-1)
                y{inl} = circshift(y{inl}, [0 1]);
            end
            for inl = 2^(l-1)+1 : 4 : 2^(l)
                y{inl} = circshift(y{inl}, [1 0]);
            end
            
            for l2 = l:-1:4
                for inl = 1:2:2^(l-2);
                    csh = cshift(l2,abs(2^(l-2)-inl) );
                    y{inl} = circshift(y{inl}, [0 csh]);
                end
                for inl = 2^(l-2)+1:2:2^(l-1);
                    csh = cshift(l2,abs(2^(l-2)-inl) );
                    y{inl} = circshift(y{inl}, [0 -csh]);
                end
                for inl = 2^(l-1)+1:2:3*2^(l-2);
                    csh = cshift(l2,abs(3*2^(l-2)-inl));
                    y{inl} = circshift(y{inl}, [csh 0]);
                end
                for inl = 3*2^(l-2)+1:2:2^(l);
                    csh = cshift(l2,abs(3*2^(l-2)-inl));
                    y{inl} = circshift(y{inl}, [-csh 0]);
                end
            end
        end %end of primal branch loop
        
        %Backsampling:
        y = back_sample_subbands(y);
        %Flip the order of the second half channels:
        y(2^(number_of_directional_filter_current-1)+1:end) = ...
                                                fliplr(y(2^(number_of_directional_filter_current-1)+1:end));
        y2{current_level_counter}{1} = y;
        
        %(2). dual branch
        y = y2{current_level_counter}{2};
        for l = 3:number_of_directional_filter_current
            % Allocate space for the new subband outputs
            y_old = y;
            y = cell(1, 2^l);
            
            % The first half channels use R1 and R2
            for k = 1:2^(l-2)
                i = mod(k-1, 2) + 1;
                [y{2*k}, y{2*k-1}] = ...
                        FB_decomposition_ladder_structure(y_old{k}, f, 'p', i, 'per');
            end
            
            % The second half channels use R3 and R4
            for k = 2^(l-2)+1:2^(l-1)
                i = mod(k-1, 2) + 3;
                [y{2*k}, y{2*k-1}] = ...
                        FB_decomposition_ladder_structure(y_old{k}, f, 'p', i, 'per');
            end
            
            % circlular shift to make the subband has minimum delay
            for inl = 1:4:2^(l-1)
                y{inl} = circshift(y{inl}, [0 1]);
            end
            for inl = 2^(l-1)+1:4:2^(l)
                y{inl} = circshift(y{inl}, [1 0]);
            end
            
            for l2 = l:-1:4
                for inl = 1:2:2^(l-2);
                    csh = cshift(l2,abs(2^(l-2)-inl) );
                    y{inl} = circshift(y{inl}, [0 csh]);
                end
                for inl = 2^(l-2)+1:2:2^(l-1);
                    csh = cshift(l2,abs(2^(l-2)-inl) );
                    y{inl} = circshift(y{inl}, [0 -csh]);
                end
                for inl = 2^(l-1)+1:2:3*2^(l-2);
                    csh = cshift(l2,abs(3*2^(l-2)-inl));
                    y{inl} = circshift(y{inl}, [csh 0]);
                end
                for inl = 3*2^(l-2)+1:2:2^(l);
                    csh = cshift(l2,abs(3*2^(l-2)-inl));
                    y{inl} = circshift(y{inl}, [-csh 0]);
                end
            end
        end
        
        % Backsampling
        y = back_sample_subbands(y);
        % Flip the order of the second half channels
        y(2^(number_of_directional_filter_current-1)+1:end) = fliplr(y(2^(number_of_directional_filter_current-1)+1:end));
        y2{current_level_counter}{2} = y;
        
    end
    
end

y2{1} = mat_in;

%---------------------------------
function csh = cshift(l2, re)
if l2 == 4
    csh = 1;
else
    % if rem < 4
    %    csh = 0;
    % else
    %    csh = 2;
    % end
    if l2 == 5
        tmp = floor(re/4);
        csh = 2^(l2-4)*tmp;
    else
        csh = 0;
    end
end


