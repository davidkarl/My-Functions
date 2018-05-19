%test different filters and windows:
rng default;
samples_per_frame = 256;
filter_order = 50;
Fs = 44100;
t_vec = my_linspace(0,1/Fs,samples_per_frame);
f_vec_small = linspace(-Fs/2,Fs/2,filter_order+1);
f_vec_large = linspace(-Fs/2,Fs/2,samples_per_frame);
input_signal = ecg(samples_per_frame)'+0.25*randn(samples_per_frame,1); % noisy waveform
filter_with_small_window = get_filter_1D('hann',20,filter_order,Fs,5000,8000,'low');
filter_with_large_window = get_filter_1D('hann',20,samples_per_frame-1,Fs,5000,8000,'low');
current_filter_small_window_group_delay = round(mean(grpdelay(filter_with_small_window)));
current_filter_large_window_group_delay = round(mean(grpdelay(filter_with_large_window)));
small_filter_window = filter_with_small_window.Numerator;
large_filter_window = filter_with_large_window.Numerator;
small_filter_window_small_fft = (fft(filter_with_small_window.Numerator));
small_filter_window_large_fft = (fft(filter_with_small_window.Numerator,samples_per_frame));
large_filter_window_large_fft = (fft(filter_with_large_window.Numerator));
color_list = hsv(10);
color_counter = 1;

% %plot original frequency response of filter:
% figure; 
% plot(filter_window);
% title('current filter time domain window');

% %plot filter frequency response:
% figure;
% hold on;
% plot(f_vec_small(end/2+1:end),20*log10(abs(filter_window_fft_small(end/2+1:end))),'b');
% plot(f_vec_large(end/2+1:end),20*log10(abs(filter_window_fft_large(end/2+1:end))),'g');
% plot(f_vec_large(end/2+1:end),20*log10(abs(large_filter_window_fft_large(end/2+1:end))),'r');
% legend('small window large fft','small window large fft','large window large fft');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot original, fftfilt with small window, fftfilt with large window:
%plot original signal for reference:
color_list = hsv(5);
color_counter = 1;
legend_string={};

figure('units','normalized','outerposition',[0 0 1 1]);
hold on;
plot(1:length(input_signal),input_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'original signal';
color_counter = color_counter+1;

%plot fftfilt with small window:
filtered_signal = fftfilt(small_filter_window,input_signal);
scatter(1:length(input_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:),'MarkerEdgeColor',color_list(color_counter,:));
legend_string{color_counter} = 'fftfilt (matlab function) filtering small window without padding';
color_counter = color_counter+1;

%plot fftfilt with large window:
filtered_signal = fftfilt(large_filter_window,input_signal);
scatter(1:length(input_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:),'MarkerEdgeColor',color_list(color_counter,:));
legend_string{color_counter} = 'fftfilt (matlab function) large window without padding';
color_counter = color_counter+1;

%plot regular filter small window:
filtered_signal = filter(filter_with_small_window,input_signal);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter small window without padding';
color_counter = color_counter+1;

%plot regular filter large window:
filtered_signal = filter(filter_with_large_window,input_signal);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter large window without padding';
color_counter = color_counter+1;

legend(legend_string);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot original signal, regular filter with small window, regular filter
%with large window, fft with small window, fft with large window:
color_list = hsv(5);
color_counter = 1;
legend_string={};

%plot original signal for reference:
figure('units','normalized','outerposition',[0 0 1 1]);
hold on;
plot(1:length(input_signal),input_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'original signal';
color_counter = color_counter+1;

%plot regular filter:
filtered_signal = filter(filter_with_small_window,input_signal);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter small window without padding';
color_counter = color_counter+1;

%regular filter large window with backward padding';
filtered_signal = filter(filter_with_large_window,input_signal);
plot(1:length(input_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter large window without padding';
color_counter = color_counter+1;

%plot direct filter in fft domain with small window:
filtered_signal = ifft(fft(input_signal).*fft(small_filter_window',samples_per_frame));
scatter(1:length(filtered_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:));
legend_string{color_counter} = 'direct fft using small window fft';
color_counter = color_counter+1;

%plot direct filter in fft domain with large window:
filtered_signal = ifft(fft(input_signal).*fft(large_filter_window'));
scatter(1:length(filtered_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:));
legend_string{color_counter} = 'direct fft using large window fft';
color_counter = color_counter+1;

legend(legend_string);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot regular filter with small window, regular filter with large window,
%regular filter with small window back padding, regular filter with large
%window back padding, regular filter with small window forward padding,
%regular filter with large window forward padding:
color_list = hsv(7);
color_counter = 1;
legend_string={};

%plot original signal for reference:
figure('units','normalized','outerposition',[0 0 1 1]);
hold on;
plot(1:length(input_signal),input_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'original signal';
color_counter = color_counter+1;

%plot regular filter with small window and backward padding:
filtered_signal = filter(filter_with_small_window,[zeros(current_filter_small_window_group_delay,1);input_signal]);
filtered_signal = filtered_signal(current_filter_small_window_group_delay+1:end);
scatter(1:length(filtered_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter small window with backward padding';
color_counter = color_counter+1;

%plot regular filter with small window and forward padding:
filtered_signal = filter(filter_with_small_window,[input_signal;zeros(current_filter_small_window_group_delay,1)]);
filtered_signal = filtered_signal(current_filter_small_window_group_delay+1:end);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter small window with forward padding';
color_counter = color_counter+1;

%plot regular filter with large window and backward padding:
filtered_signal = filter(filter_with_large_window,[zeros(current_filter_large_window_group_delay,1);input_signal]);
filtered_signal = filtered_signal(current_filter_large_window_group_delay+1:end);
scatter(1:length(filtered_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter large window with backward padding';
color_counter = color_counter+1;

%plot regular filter with large window and forward padding:
filtered_signal = filter(filter_with_large_window,[input_signal;zeros(current_filter_large_window_group_delay,1)]);
filtered_signal = filtered_signal(current_filter_large_window_group_delay+1:end);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter large window forward padding';

legend(legend_string);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot filtfilt with small window, filtfilt with large window, filter fft
%with small window, filter fft with large window:
color_list = hsv(5);
color_counter = 1;
legend_string={};

%plot original signal for reference:
figure('units','normalized','outerposition',[0 0 1 1]);
hold on;
plot(1:length(input_signal),input_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'original signal';
color_counter = color_counter+1;

%plot filtfilt with small filter window:
tic
filtered_signal = filtfilt(small_filter_window,1,input_signal);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'filtfilt with small window';
color_counter = color_counter+1;
toc

%plot regular filter with small window and forward padding:
tic
filtered_signal = filter(filter_with_small_window,[input_signal;zeros(current_filter_small_window_group_delay,1)]);
filtered_signal = filtered_signal(current_filter_small_window_group_delay+1:end);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter small window with forward padding';
color_counter = color_counter+1;
toc

%plot regular filter with large window and forward padding:
tic
filtered_signal = filter(filter_with_large_window,[input_signal;zeros(current_filter_large_window_group_delay,1)]);
filtered_signal = filtered_signal(current_filter_large_window_group_delay+1:end);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter large window with forward padding';
toc

legend(legend_string);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot filtfilt with small window, filter fft
%with small window two ways, filter fft with large window two ways:
color_list = hsv(7);
color_counter = 1;
legend_string={};

%plot original signal for reference:
figure('units','normalized','outerposition',[0 0 1 1]);
hold on;
plot(1:length(input_signal),input_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'original signal';
color_counter = color_counter+1;

%plot filtfilt with small filter window:
tic
filtered_signal = filtfilt(small_filter_window,1,input_signal);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'filtfilt with small window';
color_counter = color_counter+1;
toc

%plot regular filter with small window and forward padding:
tic
filtered_signal_forward_filter = filter(filter_with_small_window,[input_signal;zeros(current_filter_small_window_group_delay,1)]);
filtered_signal_forward_filter = filtered_signal_forward_filter(current_filter_small_window_group_delay+1:end);
filtered_flipped_signal_forward_filter = filter(filter_with_small_window,[flip(input_signal);zeros(current_filter_small_window_group_delay,1)]);
filtered_flipped_signal_forward_filter = filtered_flipped_signal_forward_filter(current_filter_small_window_group_delay+1:end);
filtered_flipped_signal_forward_filter = flip(filtered_flipped_signal_forward_filter);
filtered_signal = (filtered_signal_forward_filter+filtered_flipped_signal_forward_filter)/2;
scatter(1:length(filtered_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter small window: (forward padding + flipped_forwardpadding_flipped)/2';
color_counter = color_counter+1;
toc

%plot regular filter with small window and forward padding:
tic
filtered_signal_forward_filter = filter(filter_with_large_window,[input_signal;zeros(current_filter_large_window_group_delay,1)]);
filtered_signal_forward_filter = filtered_signal_forward_filter(current_filter_large_window_group_delay+1:end);
filtered_signal_backward_filter = filter(filter_with_large_window,[flip(input_signal);zeros(current_filter_large_window_group_delay,1)]);
filtered_signal_backward_filter = filtered_signal_backward_filter(current_filter_large_window_group_delay+1:end);
filtered_signal_backward_filter = flip(filtered_signal_backward_filter);
filtered_signal = (filtered_signal_forward_filter+filtered_signal_backward_filter)/2;
scatter(1:length(filtered_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter large window: (forward padding + flipped_forwardpadding_flipped)/2';
color_counter = color_counter+1;
toc

legend(legend_string);
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot filtfilt with small window, filter fft
%with small window two ways, filter fft with large window two ways:
color_list = hsv(5);
color_counter = 1;
legend_string={};

input_signal_small_window_forward_padding = [input_signal;zeros(current_filter_small_window_group_delay,1)];
input_signal_large_window_forward_padding = [input_signal;zeros(current_filter_large_window_group_delay,1)];
input_signal_small_window_forward_padding_flipped = [flip(input_signal);zeros(current_filter_small_window_group_delay,1)];
input_signal_large_window_forward_padding_flipped = [flip(input_signal);zeros(current_filter_large_window_group_delay,1)];
small_window_padded_signal_length = length(input_signal_small_window_forward_padding);
large_window_padded_signal_length = length(input_signal_large_window_forward_padding);

%plot original signal for reference:
figure('units','normalized','outerposition',[0 0 1 1]);
hold on;
plot(1:length(input_signal),input_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'original signal';
color_counter = color_counter+1;

%plot filtfilt with small filter window:
tic
filtered_signal = filtfilt(small_filter_window,1,input_signal);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'filtfilt with small window';
color_counter = color_counter+1;
toc

%plot direct filter in fft domain with small window and forward padding both ways:
tic
filtered_signal1 = ifft(fft(input_signal_small_window_forward_padding).*fft(small_filter_window',small_window_padded_signal_length));
filtered_signal2 = (ifft(fft(input_signal_small_window_forward_padding_flipped).*fft(small_filter_window',small_window_padded_signal_length)));
filtered_signal = (filtered_signal1(current_filter_small_window_group_delay+1:end) + flip(filtered_signal2(current_filter_small_window_group_delay+1:end)))/2;
scatter(1:length(filtered_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:));
legend_string{color_counter} = 'direct fft small window: (forward padding + flipped_forwardpadding_flipped)/2';
color_counter = color_counter+1;
toc

%plot regular filter with small window and forward padding both ways:
tic
filtered_signal_forward_filter = filter(filter_with_small_window,[input_signal;zeros(current_filter_small_window_group_delay,1)]);
filtered_signal_forward_filter = filtered_signal_forward_filter(current_filter_small_window_group_delay+1:end);
filtered_signal_backward_filter = filter(filter_with_small_window,[flip(input_signal);zeros(current_filter_small_window_group_delay,1)]);
filtered_signal_backward_filter = filtered_signal_backward_filter(current_filter_small_window_group_delay+1:end);
filtered_signal_backward_filter = flip(filtered_signal_backward_filter);
filtered_signal = (filtered_signal_forward_filter+filtered_signal_backward_filter)/2;
scatter(1:length(filtered_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter small window: (forward padding + flipped_forwardpadding_flipped)/2';
color_counter = color_counter+1;
toc

%plot regular filter with large window and forward padding both ways:
tic
filtered_signal_forward_filter = filter(filter_with_large_window,[input_signal;zeros(current_filter_large_window_group_delay,1)]);
filtered_signal_forward_filter = filtered_signal_forward_filter(current_filter_large_window_group_delay+1:end);
filtered_signal_backward_filter = filter(filter_with_large_window,[flip(input_signal);zeros(current_filter_large_window_group_delay,1)]);
filtered_signal_backward_filter = filtered_signal_backward_filter(current_filter_large_window_group_delay+1:end);
filtered_signal_backward_filter = flip(filtered_signal_backward_filter);
filtered_signal = (filtered_signal_forward_filter+filtered_signal_backward_filter)/2;
scatter(1:length(filtered_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter large window: (forward padding + flipped_forwardpadding_flipped)/2';
color_counter = color_counter+1;
toc

legend(legend_string);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot filtfilt, regular filter small window with forward padding, direct
%fft filtering with larger N and small window:
color_list = hsv(5);
color_counter = 1;
legend_string={};

%plot original signal for reference:
figure('units','normalized','outerposition',[0 0 1 1]);
hold on;
plot(1:length(input_signal),input_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'original signal';
color_counter = color_counter+1;

%plot filtfilt with small filter window:
tic
filtered_signal = filtfilt(small_filter_window,1,input_signal);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'filtfilt with small window';
color_counter = color_counter+1;
toc

%plot direct fft filtering with larger N small window:
next_power_2_length = 2*length(input_signal);
original_length = length(input_signal);
tic
filtered_signal = ifft(fft(input_signal,next_power_2_length).*fft(small_filter_window',next_power_2_length));
filtered_signal = filtered_signal(ceil(length(small_filter_window)/2):original_length+ceil(length(small_filter_window)/2)-1);
scatter(1:length(filtered_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:));
legend_string{color_counter} = 'direct fft using small window fft';
color_counter = color_counter+1;
toc

%plot regular filter with small window and forward padding:
tic
filtered_signal = filter(filter_with_small_window,[input_signal;zeros(current_filter_small_window_group_delay,1)]);
filtered_signal = filtered_signal(current_filter_small_window_group_delay+1:end);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'regular filter small window with forward padding';
color_counter = color_counter+1;
toc

legend(legend_string);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot cumsum+small_window_filter , integrator_filter+small_window_filter:
color_list = hsv(7);
color_counter = 1;
legend_string={};

%plot original signal for reference:
figure('units','normalized','outerposition',[0 0 1 1]);
hold on;
plot(1:length(input_signal),cumsum(input_signal),'color',color_list(color_counter,:));
legend_string{color_counter} = 'cumsum';
color_counter = color_counter+1;

%plot signal after cumsum and low pass filtering:
% integral_signal = cumsum(input_signal);
% integral_signal = filter(filter_with_small_window,integral_signal);
integral_signal = filter(filter_with_small_window,input_signal);
integral_signal = cumsum(integral_signal);
plot(1:length(integral_signal),integral_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'cumsum + small window lowpass filtering';
color_counter=color_counter+1;

%plot signal after integrator+low pass filters:
integrator_filter = dfilt.df1(1,[1,-0.999999999999999]);
total_filter = dfilt.cascade(integrator_filter,filter_with_small_window);
filtered_signal = filter(total_filter,input_signal);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'integrator + small window lowpass filtering';
color_counter=color_counter+1;

%plot signal after filtering with forward padding and cumsum:
filtered_signal = filter(filter_with_small_window,[input_signal;zeros(current_filter_small_window_group_delay,1)]);
filtered_signal = filtered_signal(current_filter_small_window_group_delay+1:end);
filtered_signal = cumsum(filtered_signal);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'small window filter with forward padding + cumsum';
color_counter=color_counter+1;

%plot signal after direct fft with small window filtering with large N and cumsum:
next_power_2_length = 2*length(input_signal);
original_length = length(input_signal);
filtered_signal = real(ifft(fft(small_filter_window,next_power_2_length).*fft(input_signal',next_power_2_length)));
filtered_signal = filtered_signal(ceil(length(small_filter_window)/2):original_length+ceil(length(small_filter_window)/2)-1);
filtered_signal = cumsum(filtered_signal);
scatter(1:length(filtered_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:));
legend_string{color_counter} = 'direct fft with small window & large N + cumsum';
color_counter=color_counter+1;

%plot signal after filtering with forward padding and cumsum:
filtered_signal = filter(filter_with_large_window,[input_signal;zeros(current_filter_large_window_group_delay,1)]);
filtered_signal = filtered_signal(current_filter_large_window_group_delay+1:end);
filtered_signal = cumsum(filtered_signal);
plot(1:length(filtered_signal),filtered_signal,'color',color_list(color_counter,:));
legend_string{color_counter} = 'large window filter with forward padding + cumsum';
color_counter=color_counter+1;

%plot signal after direct fft filtering with large N and cumsum:
next_power_2_length = 2*length(input_signal);
original_length = length(input_signal);
filtered_signal = real(ifft(fft(large_filter_window,next_power_2_length).*fft(input_signal',next_power_2_length)));
filtered_signal = filtered_signal(current_filter_large_window_group_delay+1:current_filter_large_window_group_delay+original_length-1);
filtered_signal = cumsum(filtered_signal);
scatter(1:length(filtered_signal),filtered_signal,'MarkerFaceColor',color_list(color_counter,:));
legend_string{color_counter} = 'direct fft with large window & large N + cumsum';
color_counter=color_counter+1;
legend(legend_string);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot small window lowpass + high pass regular, direct fft, and filtfilt
color_list = hsv(7);
color_counter = 1;
legend_string={};




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot small window filter with window size larger then input signal regular
%and direct fft and filtfilt:
color_list = hsv(7);
color_counter = 1;
legend_string={};



1;


 





