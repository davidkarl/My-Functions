signal = make_correlated_noise([100,1],-0.25);
filter1 = get_filter_1D('kaiser',8,20,5000,400,1500,'bandpass');
filter1 = filter1.numerator(:);

signal_length = length(signal);
filter_length = length(filter1); 

filter1 = filter1';
signal = signal';


%(1). Periodic convolution:
if filter_length <= signal_length
    signal_padded = [signal((signal_length+1-filter_length):signal_length) , signal];
else
    z = zeros(1,filter_length);
    for i=1:filter_length,
        imod = 1 + rem(filter_length*signal_length -filter_length + i-1 , signal_length);
        z(i) = signal(imod);
    end
    signal_padded = [z , signal];
end
ypadded = filter(filter1,1,signal_padded);
filtered_signal1 = ypadded((filter_length+1):(signal_length+filter_length));


%(2). Periodic convolution reverse filter:
if filter_length < signal_length,
    xpadded = [signal , signal(1:filter_length)];
else
    z = zeros(1,filter_length);
    for i=1:filter_length,
        imod = 1 + rem(i-1,signal_length);
        z(i) = signal(imod);
    end
    xpadded = [signal z];
end
filter_in_reversed = reverse_vec(filter1);
ypadded = filter(filter_in_reversed,1,xpadded);
filtered_signal2 = ypadded(filter_length:(signal_length+filter_length-1));


%(3). Periodic convolution symmetric filter:
symmetric_filter = filter1;
if filter_length <= signal_length,
    xpadded = [signal((signal_length+1-filter_length):signal_length) , signal];
else
    z = zeros(1,filter_length);
    for i=1:filter_length,
        imod = 1 + rem(filter_length*signal_length -filter_length + i-1,signal_length);
        z(i) = signal(imod);
    end
    xpadded = [z signal];
end
ypadded = filter(symmetric_filter,1,xpadded);
filtered_signal3 = ypadded((filter_length+1):(signal_length+filter_length));
shift = (filter_length+1)/2;
shift = 1 + rem(shift-1, signal_length);
filtered_signal3 = [filtered_signal3(shift:signal_length) filtered_signal3(1:(shift-1))];


%(4). Periodic convolution reverse symmetric filter:
symmetric_filter = filter1;
if filter_length < signal_length,
    xpadded = [signal , signal(1:filter_length)];
else
    z = zeros(1,filter_length);
    for i=1:filter_length,
        imod = 1 + rem(i-1,signal_length);
        z(i) = signal(imod);
    end
    xpadded = [signal z];
end
filter_in_reversed = reverse_vec(symmetric_filter);
ypadded = filter(filter_in_reversed,1,xpadded);
if filter_length < signal_length,
    filtered_signal4 = [ypadded((signal_length+1):(signal_length+filter_length)) ypadded((filter_length+1):(signal_length))];
else
    for i=1:signal_length,
        imod = 1 + rem(filter_length+i-1,signal_length);
        filtered_signal4(imod) = ypadded(filter_length+i);
    end
end
shift = (filter_length-1)/ 2 ;
shift = 1 + rem(shift-1, signal_length);
filtered_signal4 = [filtered_signal4((1+shift):signal_length) filtered_signal4(1:(shift))] ;


%(5). conv without end effects:
signal = signal(:);
filter1 = filter1(:);
flag_use_fft = 1;
if flag_use_fft==1
    filtered_signal5 = conv_fft(signal,filter1)/sum(filter1);
else
    filtered_signal5 = conv(signal,filter1,'same')/sum(filter1);
end
for k=1:floor(length(filter1)/2)   
    filtered_signal5(k) = sum(signal(k:length(filter1)-1) .* filter1(k+1:end)) / sum(filter1(k+1:end));
    filtered_signal5(end-k+1) = sum(signal(end-length(filter1)+1+k : end) .* filter1(1:end-k)) / sum(filter1(1:end-k));
end


%(6). conv fft:
Ly = length(signal)+length(filter1)-1;  % 
Ly2 = pow2(nextpow2(Ly));    % Find smallest power of 2 that is > Ly
X = fft(signal, Ly2);		   % Fast Fourier transform
H = fft(filter1, Ly2);	           % Fast Fourier transform
Y = X.*H;        	           % 
filtered_signal6 = real(ifft(Y, Ly2));      % Inverse fast Fourier transform
filtered_signal6 = filtered_signal6(floor(length(filter1)/2)+1:length(signal)+floor(length(filter1)/2));               % Take just the first N elements



%Plot filtered signals:
subplot(4,2,1);
plot(signal);
legend('original signal');
subplot(4,2,2);
plot(filtered_signal1);
legend('periodic convolution');
subplot(4,2,3);
plot(filtered_signal2);
legend('periodic convolution reverse filter');
subplot(4,2,4);
plot(filtered_signal3);
legend('periodic convolution symmetric filter');
subplot(4,2,5);
plot(filtered_signal4);
legend('periodic convolution reverse symmetric filter');
subplot(4,2,6);
plot(filtered_signal5);
legend('conv without end effects');
subplot(4,2,7);
plot(filtered_signal6);
legend('conv fft');
















