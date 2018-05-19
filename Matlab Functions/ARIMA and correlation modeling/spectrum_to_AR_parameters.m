function [AR_parameters,g_AR_gain,auto_correlation_sequence]=spectrum_to_AR_parameters(power_spectrum,AR_model_order)
% Given a symetrical (full-length) power spectrum s, this function
% estimates the a and b parameters of an appropriate white noise driven 
% AR model. c is the autocorrelation calculated as a by product
% Usage: [a b]=s2ae(s,n) ;

%get current power specturm length:
N=length(power_spectrum) ;

%Compute the autocorrelation, omit dividing by N for simplicity:
[auto_correlation_sequence] = power_spectrum_to_autocorrelation(power_spectrum,N/2);

% various approaches can be taken to estimate a
% AR_parameters=c2ach(auto_correlation,n) ;
% AR_parameters=th2poly(ar(auto_correlation,n,'burg')) ;
% AR_parameters=-AR_parameters(2:n+1) ;

%Calculate the AR parameters:
[AR_parameters,forward_prediction_error,reflection_coefficient] = ...
    get_AR_parameters_burg_method(auto_correlation_sequence,AR_model_order);

%AR_parameters = tstlsw(auto_correlation,n)' ;
[amplitude_spectrum] = AR_parameters_to_amplitude_spectrum(AR_parameters,N);

g_AR_gain=sqrt(abs(sum(power_spectrum)/sum(amplitude_spectrum.^2))) ;

% b=getnoise(c,a)
% b=sqrt(abs(b))  ;
auto_correlation_sequence=auto_correlation_sequence(1:AR_model_order) ;