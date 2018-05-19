function y = smooth_vector_over_N_neighbors(input_signal,N)
input_signal = ones(100,1);
N=3;

input_signal_length = length(input_signal);
averaging_window_total = hanning(2*N+1);
averaging_window_left_part = averaging_window_total(1:N+1);
averaging_window_right_part = averaging_window_total(N+2:end);

%Smooth input signal first by averaging input_signal elements with values to the right (flipud) of them:
y1=filter(flipud(averaging_window_left_part),[1],input_signal);

%Smooth input signal second by average input_signal elements with values to the left (flipud) of them:
x2 = zeros(input_signal_length,1);
x2(1:input_signal_length-N) = input_signal(N+1:input_signal_length);
y2 = filter(flipud(averaging_window_right_part),[1],x2);

%Combine both parts:
y = (y1+y2)/norm(averaging_window_total,2); 