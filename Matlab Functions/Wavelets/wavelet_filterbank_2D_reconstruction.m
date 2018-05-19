function x = wavelet_filterbank_2D_reconstruction(x_LL, x_LH, x_HL, x_HH, lowpass_analysis_filter, lowpass_synthesis_filter)
% WFB2REC   2-D Wavelet Filter Bank Decomposition
%
%       x = wfb2rec(x_LL, x_LH, x_HL, x_HH, h, g)
%
% Input:
%   x_LL, x_LH, x_HL, x_HH:   Four 2-D wavelet subbands
%   h, g:   lowpass analysis and synthesis wavelet filters
%
% Output:
%   x:      reconstructed image

% Make sure filter in a row vector
lowpass_analysis_filter = lowpass_analysis_filter(:)';
lowpass_synthesis_filter = lowpass_synthesis_filter(:)';
lowpass_synthesis_filter_length = length(lowpass_synthesis_filter);
lowpass_synthesis_filter_invalid_samples = floor((lowpass_synthesis_filter_length - 1) / 2);

%Highpass synthesis filter: G1(z) = -z H0(-z)
highpass_synthesis_filter_length = length(lowpass_analysis_filter);
highpass_synthesis_filter_center = floor((highpass_synthesis_filter_length + 1) / 2);
highpass_synthesis_filter_samples_vec = [1:highpass_synthesis_filter_length] - highpass_synthesis_filter_center;
highpass_synthesis_filter = (-1) * lowpass_analysis_filter .* (-1) .^ (highpass_synthesis_filter_samples_vec);
highpass_synthesis_filter_invalid_samples = highpass_synthesis_filter_length - (highpass_synthesis_filter_center + 1);

%Get the output image size:
[height, width] = size(x_LL);
x_upsampled_temp = zeros(height * 2, width);

%Column-wise filtering:
x_upsampled_temp(1:2:end, :) = x_LL;
x_L = rowfiltering(x_upsampled_temp', lowpass_synthesis_filter, lowpass_synthesis_filter_invalid_samples)';
x_upsampled_temp(1:2:end, :) = x_LH;
x_L = x_L + rowfiltering(x_upsampled_temp', highpass_synthesis_filter, highpass_synthesis_filter_invalid_samples)';

x_upsampled_temp(1:2:end, :) = x_HL;
x_H = rowfiltering(x_upsampled_temp', lowpass_synthesis_filter, lowpass_synthesis_filter_invalid_samples)';
x_upsampled_temp(1:2:end, :) = x_HH;
x_H = x_H + rowfiltering(x_upsampled_temp', highpass_synthesis_filter, highpass_synthesis_filter_invalid_samples)';

%Row-wise filtering:
x_upsampled_temp = zeros(2*height, 2*width);
x_upsampled_temp(:, 1:2:end) = x_L;
x = rowfiltering(x_upsampled_temp, lowpass_synthesis_filter, lowpass_synthesis_filter_invalid_samples);
x_upsampled_temp(:, 1:2:end) = x_H;
x = x + rowfiltering(x_upsampled_temp, highpass_synthesis_filter, highpass_synthesis_filter_invalid_samples);


% Internal function: Row-wise filtering with border handling 
function y = rowfiltering(x, f, ext1)
ext2 = length(f) - ext1 - 1;
x = [x(:, end-ext1+1:end) x x(:, 1:ext2)];
y = conv2(x, f, 'valid');