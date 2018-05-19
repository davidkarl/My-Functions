function [frequency_response_amplitude,frequency_complex_response] = ...
    get_AR_parameters_frequency_response(AR_parameters, number_of_frequencies, flag_plot)
% draws a frequency response of an AR system

%All pole numerator/denominator initialization:
num = [1 , zeros(1,length(AR_parameters)-1)] ;
den = [1 , -AR_parameters] ;

%get (normalized) frequencies vec:
frequencies_vec = linspace(0.01,1,number_of_frequencies) ;
frequency_complex_response = zeros(1,number_of_frequencies);

%loop over the different normalized frequencies and get frequency response:
for k=1:number_of_frequencies,
  z=exp(-1i*frequencies_vec(k));
  frequency_complex_response(k)=polyval(num,z)/polyval(den,z);
end

%get frequency response amplitude:
frequency_response_amplitude = abs(frequency_complex_response);

%plot response if wanted:
if flag_plot==1
    plot(frequencies_vec,frequency_response_amplitude);
end

