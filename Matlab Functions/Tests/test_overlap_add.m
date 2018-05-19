%test overlap add:
filter_length = 31;         % FIR filter length in taps
F_cutoff = 600;       % lowpass cutoff frequency in Hz
Fs = 4000;      % sampling rate in Hz

total_signal_length = 2048*200;     % signal length in samples
period = round(filter_length/3); % signal period in samples

% M = filter_length;                  % nominal window length
samples_per_frame = 2048;
Nfft = 2^(ceil(log2(samples_per_frame+filter_length-1))); % FFT Length
M = Nfft-filter_length+1;            % efficient window length
R = M;                  % hop size for rectangular window
Nframes = 1+floor((total_signal_length-M)/R);  % no. complete frames

epsilon = .0001;     % avoids 0 / 0
nfilt = (-(filter_length-1)/2:(filter_length-1)/2) + epsilon;
hideal = sin(2*pi*F_cutoff*nfilt/Fs) ./ (pi*nfilt);
w = hamming(filter_length); % FIR filter design by window method
h = w' .* hideal; % window the ideal impulse response

hzp = [h zeros(1,Nfft-filter_length)];  % zero-pad h to FFT size
H = fft(hzp);               % filter frequency response

y = zeros(1,total_signal_length + Nfft); % allocate output+'ringing' vector
for m = 0:(Nframes-1)
    index = m*R+1:min(m*R+M,total_signal_length); % indices for the mth frame
    xm = sig(index);  % windowed mth frame (rectangular window)
    xmzp = [xm zeros(1,Nfft-length(xm))]; % zero pad the signal
    Xm = fft(xmzp);
    Ym = Xm .* H;               % freq domain multiplication
    ym = real(ifft(Ym));         % inverse transform
    outindex = m*R+1:(m*R+Nfft);
    y(outindex) = y(outindex) + ym; % overlap add
end

frames2vec








