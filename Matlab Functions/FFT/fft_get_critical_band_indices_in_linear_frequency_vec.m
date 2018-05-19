function [critical_bands_indices_in_linear_frequency_vec] = fft_get_critical_band_indices_in_linear_frequency_vec(Fs,dft_length,nbits,frame_overlap)
freq_val = (0:Fs/dft_length:Fs/2)';
freq=freq_val;
crit_band_ends = [0;100;200;300;400;510;630;770;920;1080;1270;1480;1720;2000;2320;2700;3150;3700;4400;5300;6400;7700;9500;12000;15500;Inf];
imax = max(find(crit_band_ends < freq(end)));
num_bins = length(freq);
LIN_TO_BARK = zeros(imax,num_bins);
i = 1;
for j = 1:num_bins
    while ~((freq(j) >= crit_band_ends(i)) & (freq(j) < crit_band_ends(i+1))),i = i+1;end
    LIN_TO_BARK(i,j) = 1;
end
% Calculation of critical band frequency indices--i.e., which bins are in which critical band for i = 1:imax
for i=1:imax,
    critical_bands_indices_in_linear_frequency_vec{i} = find(LIN_TO_BARK(i,:));
end