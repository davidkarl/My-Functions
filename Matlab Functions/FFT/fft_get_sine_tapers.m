function [sine_tapers] = fft_get_sine_tapers(number_of_sine_tapers,individual_sine_taper_length)

sine_tapers = zeros(number_of_sine_tapers, individual_sine_taper_length);
for sine_taper_counter = 1:number_of_sine_tapers
    sine_tapers(sine_taper_counter,:) = sqrt(2/(individual_sine_taper_length+1)) * sin(pi*sine_taper_counter*[1:individual_sine_taper_length]'/(individual_sine_taper_length+1));
end 

