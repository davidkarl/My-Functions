function [x_LL, x_LH, x_HL, x_HH] = wavelet_filterbank_2D_decomposition(mat_in, lowpass_analysis_filter, lowpass_synthesis_filter)
% WFB2DEC   2-D Wavelet Filter Bank Decomposition
%
%       y = wfb2dec(x, h, g)
%
% Input:
%   x:      input image
%   h, g:   lowpass analysis and synthesis wavelet filters
%
% Output:
%   x_LL, x_LH, x_HL, x_HH:   Four 2-D wavelet subbands

%Make sure filter in a row vector:
lowpass_analysis_filter = lowpass_analysis_filter(:)';
lowpass_synthesis_filter = lowpass_synthesis_filter(:)';

%Get invalid number of samples:
lowpass_analysis_filter_length = length(lowpass_analysis_filter);
number_of_invalid_samples_lowpass_analysis = floor(lowpass_analysis_filter_length / 2);

%Highpass analysis filter: H1(z) = -z^(-1) G0(-z)
highpass_analysis_filter_length = length(lowpass_synthesis_filter);
highpass_analysis_filter_center = floor((highpass_analysis_filter_length + 1) / 2); 

%Shift the center of the filter by 1 if its length is even:
if mod(highpass_analysis_filter_length, 2) == 0
    highpass_analysis_filter_center = highpass_analysis_filter_center + 1;
end

%Get highpass analysis filter:
centered_highpass_analysis_filter_samples_vec = [1:highpass_analysis_filter_length] - highpass_analysis_filter_center;
highpass_analysis_filter = - lowpass_synthesis_filter .* (-1).^(centered_highpass_analysis_filter_samples_vec);
number_of_invalid_samples_highpass_analysis = highpass_analysis_filter_length - highpass_analysis_filter_center + 1;

%Row-wise filtering:
x_L = rowfiltering(mat_in, lowpass_analysis_filter, number_of_invalid_samples_lowpass_analysis);
x_L = x_L(:, 1:2:end);

x_H = rowfiltering(mat_in, highpass_analysis_filter, number_of_invalid_samples_highpass_analysis);
x_H = x_H(:, 1:2:end);

%Column-wise filtering:
x_LL = rowfiltering(x_L', lowpass_analysis_filter, number_of_invalid_samples_lowpass_analysis)';
x_LL = x_LL(1:2:end, :);

x_LH = rowfiltering(x_L', highpass_analysis_filter, number_of_invalid_samples_highpass_analysis)';
x_LH = x_LH(1:2:end, :);

x_HL = rowfiltering(x_H', lowpass_analysis_filter, number_of_invalid_samples_lowpass_analysis)';
x_HL = x_HL(1:2:end, :);

x_HH = rowfiltering(x_H', highpass_analysis_filter, number_of_invalid_samples_highpass_analysis)';
x_HH = x_HH(1:2:end, :);


% Internal function: Row-wise filtering with border handling 
function y = rowfiltering(x, f, ext1)
ext2 = length(f) - ext1 - 1;
x = [x(:, end-ext1+1:end) , x , x(:, 1:ext2)];
y = conv2(x, f, 'valid');