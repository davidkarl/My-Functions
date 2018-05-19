function [auto_correlation_sequence] = get_auto_correlation_using_direct_calculation(input_signal,number_of_samples_to_compute)
% Computes the first n coefficients of the autocorellation
% function of the signal y
% c(1) is the correllation at lag 0, c(2) at lag 1 etc.
% The correlation is biased, as Mehra claims it gives better results
% for the identification

% corrn

N=length(input_signal) ;
auto_correlation_sequence=zeros(number_of_samples_to_compute,1) ;

for i=1:number_of_samples_to_compute,
  auto_correlation_sequence(i)=sum(input_signal(1:N-i+1).*input_signal(i:N))/N ;
end ;