%test sound with noise in differences:
[y,Fs] = read_wav_file_full_options('feynman.wav',10,30,44100,0);
[y_downsampled,Fs2] = read_wav_file_full_options('feynman.wav',10,30,44100/4,0);
noise = randn(length(y)-1,1)*0.1;
noise_downsampled = noise(1:4:end);

%add noise to high sampled version:
y_diff = diff(y);
y_diff_noisy = y_diff + noise;
y_diff_noisy_cumsum = cumsum(y_diff_noisy);

%add noise to low sampled version:
y_downsampled_diff = diff(y_downsampled);
y_downsampled_diff_noisy = y_downsampled_diff + noise_downsampled;
y_downsampled_diff_noisy_cumsum = cumsum(y_downsampled_diff_noisy);

%filter high sampled version to equate apples with apples:
bla = get_filter_1D('kaiser',10,5000,Fs,44100/2/4,1,'lowpass');
bla2 = get_filter_1D('kaiser',10,5000,Fs,44100/2/4,1,'lowpass');
y_diff_noisy_cumsum = filter(bla.Numerator,1,y_diff_noisy_cumsum);
y_noisy_filtered = filter(bla.Numerator,1,y(1:end-1)+noise);

bla1 = y(6001:end) - y_diff_noisy_cumsum(6000:end);
std1 = std(bla1);
bla2 = y_downsampled(6001:end) - y_downsampled_diff_noisy_cumsum(6000:end);
std2 = std(bla2);
1;
% soundsc(y_diff_noisy_cumsum,Fs);
% soundsc(y_downsampled_diff_noisy_cumsum,Fs2);
% soundsc(y_noisy_filtered,Fs);
% soundsc(y_downsampled(1:end-1) + noise_downsampled, Fs2);
%clear sound 


