function [quantized_signal] = quantize_signal(signal,N_bits,flag_normalize)
%quantize a signal assumed to be -1<signal<1 
%OR- normalize it before quantization

maximum_signal = max(abs(signal));
if flag_normalize==1
   quantized_signal = signal/maximum_signal;
   maximum_signal = 1;
end

quanta = 2^(N_bits)-1;
quantized_signal = int64(quantized_signal*quanta);
quantized_signal = double(quantized_signal) / quanta;
quantized_signal = quantized_signal * maximum_signal;



