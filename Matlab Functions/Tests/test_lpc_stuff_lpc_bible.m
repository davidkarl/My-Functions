%test lpc stuff lpc bible
clear all;
clc;


%Initialize autio parameters:
LPC_model_order = 12;
offset_frame_number=20;
FFT_size=512;
frame_size_in_seconds=20*10^-3;
frame_overlap_size_in_seconds=5*10^-3;
frame_total_size_in_seconds = frame_size_in_seconds+2*frame_overlap_size_in_seconds;

%read input signal:
[input_signal,Fs]=wavread('shirt_2mm_ver_200m_audioFM final demodulated audio150-3000[Hz]');
input_signal=input_signal(:,1);

%initialize frame sample sizes:
samples_per_frame = round(frame_total_size_in_seconds*Fs);
overlapping_samples_per_frame = round(frame_overlap_size_in_seconds*Fs);
non_overlapping_samples_per_frame = samples_per_frame - overlapping_samples_per_frame;

%cut input signal and make it a multiple of samples_per_frame:
input_signal = input_signal(1:length(input_signal)-mod(length(input_signal),samples_per_frame));
input_signal_min_and_max = [min(input_signal),max(input_signal)];
input_signal_dimensions = size(input_signal);
total_number_of_frames = round(length(input_signal)/non_overlapping_samples_per_frame);
framed_input_signal_dimensions = [samples_per_frame, total_number_of_frames];
minmaxyframe = [min(input_signal),max(input_signal)];

%initialize framing window:
hamming_window = hamming(samples_per_frame);

%frame input signal:
framed_input_signal = buffer(input_signal,samples_per_frame,overlapping_samples_per_frame);

%window frame input signal using a hamming window:
framed_and_windowed_input_signal = bsxfun(@times,framed_input_signal,hamming_window);
framed_and_windowed_narrowband_input_signal = framed_and_windowed_input_signal;

%calculate one sided autocorrelation for each frame:
autocorrelation_signal_narrowband = zeros(2*samples_per_frame-1,total_number_of_frames);
autocorrelation_lags = zeros(2*samples_per_frame-1,total_number_of_frames);
for i=1:total_number_of_frames
   [autocorrelation_signal_narrowband(:,i),autocorrelation_lags(:,i)] = xcorr(framed_and_windowed_narrowband_input_signal(:,i)); 
end
autocorrelation_signal_narrowband = autocorrelation_signal_narrowband(find(autocorrelation_lags(:,1)==0):end,:);

%estimate the LPC model model parameters for each frame (g is the variance, so we later use sqrt(g)):
[a,g] = levinson(autocorrelation_signal_narrowband,LPC_model_order);

%calculate frequency response of sqrt(g)/a filter:
[H,F] = freqz(sqrt(g(offset_frame_number)),a(offset_frame_number,:),FFT_size,Fs);

%calculate LSF (Line Spectral Frequencies) representation of LPC coefficients a:
lsf = poly2lsf(a(offset_frame_number,:));

%calculate frequency response of 1/lsf filter:
[H1,F1]=freqz(1,lsf,FFT_size,Fs);

%calculate the error signal (y(n)=g*x(n)-sum(a(k)y(n-k)) -> error_signal = x(n):
error_signal_narrowband = filter(a(offset_frame_number,:),sqrt(g(offset_frame_number)),framed_and_windowed_narrowband_input_signal);

%reconstruct the original signal from x(n) using the filter sqrt(g)/a: 
final_narrowband_signal_reconstructed = filter(sqrt(g(offset_frame_number)),a(offset_frame_number,:),error_signal_narrowband);

 
figure;
subplot(2,1,1)
%show framed input signal's frame number "offset_frame_number" and error signal:
hold on;
plot([1:samples_per_frame],framed_and_windowed_input_signal(:,offset_frame_number),'g');
plot(1:samples_per_frame,real(error_signal_narrowband(:,offset_frame_number)),'b');
plotyframewindow = plot([1:5:samples_per_frame],framed_and_windowed_input_signal(1:5:end,offset_frame_number),'go');
plotyframe = plot([1:samples_per_frame],framed_input_signal(:,offset_frame_number),'r');
ploterror = plot(1:5:samples_per_frame,real(error_signal_narrowband(1:5:end,offset_frame_number)),'bx');
title('input signal and error signal frame 20');
xlabel('samples[n]');
ylabel('amplitude');
grid
xlim([1,samples_per_frame]);
legend([plotyframe plotyframewindow ploterror],'inputsignal','inputsignal*hamming','errorsignal');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
subplot(2,1,2)
%show frame fft and filter H (create by sqrt(g)/a & lpc analysis) frequency response:
hold on;
minmaxdBscale = [min(20*log10(2*abs(H)/FFT_size)),max(20*log10(2*abs(H)/FFT_size))];
plotfft = plot([0:FFT_size-1]*Fs/(FFT_size-1),20*log10(2*abs(fft(framed_and_windowed_input_signal(:,offset_frame_number),FFT_size))/FFT_size));
plot(F,20*log10(2*abs(H)/FFT_size),'r');
plotlpc = plot(F(1:10:end),20*log10(2*abs(H(1:10:end))/FFT_size),'rx');
stem((lsf/pi)*Fs/2,-200+20*log10(ones(1,length(lsf))),'m');
title('fft of input signal and frequency response of lpc frame 20');
xlabel('frequency[Hz]');
ylabel('amplitude[dB]');
legend([plotfft plotlpc],'fft','lpc order 12');
grid
ylim([minmaxdBscale]);
xlim([0,Fs/2]);


%show framed and windowed input signal:
figure;
plot([0:samples_per_frame-1]*1/Fs+(frame_total_size_in_seconds-frame_overlap_size_in_seconds)*(offset_frame_number-1),framed_and_windowed_input_signal(:,offset_frame_number));
title('input signal frame 20');
xlim([0,samples_per_frame-1]*1/Fs+(frame_total_size_in_seconds-frame_overlap_size_in_seconds)*(offset_frame_number-1));
xlabel('time[s]');
ylabel('amplitude');
ylim([minmaxyframe]);
grid;

%show lpc coefficients and lpc filter frequency response:
figure;
subplot(2,1,1)
stem([0:LPC_model_order],a(offset_frame_number,:));
title('LPC coefficients LPC order 12');
xlabel('coefficients[n]');
ylabel('amplitude');
subplot(2,1,2)
plot(F,20*log10(2*abs(H)/FFT_size));
title('LPC frequency response frame 20');
xlim([0,Fs/2]);
xlabel('frequency [Hz]');
ylabel('amplitude [dB]');
grid

%show final error signal:
figure
plot([0:samples_per_frame-1]*1/Fs + (frame_total_size_in_seconds-frame_overlap_size_in_seconds)*(offset_frame_number-1),error_signal_narrowband(:,offset_frame_number));
title('error signal frame 20');
xlabel('time[s]');
ylabel('amplitdue');
ylim([minmaxyframe]);
grid

%show final reconstructed signal:
figure;
plot([0:samples_per_frame-1]*1/Fs+(frame_total_size_in_seconds-frame_overlap_size_in_seconds)*(offset_frame_number-1),final_narrowband_signal_reconstructed(:,offset_frame_number));
xlim([0,samples_per_frame-1]*1/Fs+(frame_total_size_in_seconds-frame_overlap_size_in_seconds)*(offset_frame_number-1));
xlabel('time[s]');
ylabel('amplitude');
ylim([minmaxyframe]);
grid;


%LSF (line spectral frequencies):
input_data = framed_and_windowed_input_signal(:,offset_frame_number);
[LPC_coefficients,sigma] = lpc(input_data,LPC_model_order);
[H2,F2] = freqz(sigma,LPC_coefficients,512,Fs);
LSF_coefficients = poly2lsf(LPC_coefficients);
Y = H2( round( LSF_coefficients*(Fs/2)/pi/FFT_size ) );
LSF_points = [exp(1i*pi*LSF_coefficients);exp(-1i*pi*LSF_coefficients)];
[K,V] = tf2latc(1,LPC_coefficients); %transfer function 2 lattice filter conversion
[numerator,denominator] = latc2tf(K,V);
[z,p,k] = zpkdata(tf(numerator,denominator));
p = cell2mat(p);
theta=0:0.01:2*pi;
x = cos(theta);
y = sin(theta);

figure;
hold on;
plot(F2,20*log10(abs(H2)));
stem(LSF_coefficients*(Fs/2)/pi,-200+abs(Y),'r');
xlabel('frequency[Hz]');
ylabel('amplitude[dB]');
axis([0,4000,-200,-100]);

figure;
hold on;
plot(LSF_points,'ro');
plot(p,'b^');
plot(x,y,'k'); 
axis equal;
xlabel('real axis');
ylabel('imaginary axis');
axis([-1.1,1.1,-1.1,1.1]);






