function [noise_masking_threshold] = audio_get_masking_threshold( clean_speech_power_spectrum_estimate, FFT_size, Fs, nbits)
%Johnston perceptual model initialisation

%Initialize helper parameters:  
freq_vec = [0:Fs/FFT_size:Fs/2]';
half_lsb = (1/(2^nbits-1))^2/FFT_size;
threshold = half_lsb;

%Get perceptual/psychoacoustic bark frequecy scale edges:
bark_frequency_scale_edges = [0;100;200;300;400;510;630;770;920;1080;1270;...
        1480;1720;2000;2320;2700;3150;3700;4400;5300;6400;7700;...
        9500;12000;15500;Inf];

%Maximum Bark frequency:
maximum_relevant_bark_frequency_index = max(find(bark_frequency_scale_edges < freq_vec(end)));

% Normalised (to 0 dB) threshold of hearing values (Fletcher, 1929) 
% as used  by Johnston.  First and last thresholds are corresponding 
% critical band endpoint values, elsewhere means of interpolated 
% critical band endpoint threshold values are used.
abs_thr = 10.^([38;31;22;18.5;15.5;13;11;9.5;8.75;7.25;4.75;2.75;...
        1.5;0.5;0;0;0;0;2;7;12;15.5;18;24;29]./10);
ABSOLUTE_THRESH = threshold .* abs_thr(1:maximum_relevant_bark_frequency_index);

%Calculation of tone-masking-noise offset ratio in dB:
offset_ratio_dB = 9 + (1:maximum_relevant_bark_frequency_index)';

%Initialisation of matrices for bark/linear frequency conversion
number_of_frequency_bins = length(freq_vec);
linear_to_bark = zeros(maximum_relevant_bark_frequency_index,number_of_frequency_bins);
i = 1;
for j = 1:number_of_frequency_bins
    while ~((freq_vec(j) >= bark_frequency_scale_edges(i)) && ...
            (freq_vec(j) < bark_frequency_scale_edges(i+1))),
        i = i+1;
    end
    linear_to_bark(i,j) = 1;
end

%Calculation of spreading function (Schroeder et al., 82):
spreading_function = zeros(maximum_relevant_bark_frequency_index);
summ = 0.474:maximum_relevant_bark_frequency_index;
spread = 10.^((15.81+7.5.*summ-17.5.*sqrt(1+summ.^2))./10);
for i = 1:maximum_relevant_bark_frequency_index
    for j = 1:maximum_relevant_bark_frequency_index
        spreading_function(i,j) = spread(abs(j-i)+1);
    end
end

%Calculation of excitation pattern function:
excitation_pattern = spreading_function * linear_to_bark;

%Calculation of DC gain due to spreading function:
DC_gain = spreading_function * ones(maximum_relevant_bark_frequency_index,1);

%Sx = X.* conj(X);
C = excitation_pattern *  clean_speech_power_spectrum_estimate;

%Calculation of spectral flatness measure SFM_dB:
[number_of_frequency_bins, number_of_frames] = size(clean_speech_power_spectrum_estimate);
k = 1/number_of_frequency_bins;
SFM_dB = 10.*log10((prod(clean_speech_power_spectrum_estimate).^k)./(k.*sum(clean_speech_power_spectrum_estimate)+eps)+ eps);

%Calculation of tonality coefficient and masked threshold offset:
alpha = min(1,SFM_dB./-60);
O_dB = offset_ratio_dB(:,ones(1,number_of_frames)) .* alpha(ones(length(offset_ratio_dB),1),:) + 5.5;

%Threshold calculation and renormalisation, accounting for absolute thresholds:
T = C./10.^(O_dB./10);
T = T./DC_gain(:,ones(1,number_of_frames));
T = max( T, ABSOLUTE_THRESH(:, ones(1, number_of_frames)));

%Reconversion to linear frequency scale 
noise_masking_threshold = linear_to_bark' * T;