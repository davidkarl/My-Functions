function [amplitude_spectrum] = AR_parameters_to_amplitude_spectrum(a,FFT_size)
% Calculates amplitude spectrum of the length N for a given AR system
% The result should be the same as if an fft were taken from the output 
% of the system when fed with white noise of unit dispersion
% Usage: g=a2s(a,N) 

AR_model_order=length(a) ;
%denr=[1 -a] ;
%denr=denr(n+1:-1:1) ;
%f=linspace(0,2*pi,N) ;
%g=zeros(1,N) ;
%for k=1:N,
%  z=exp(-i*f(k)) ;
%  g(k)=abs(1/polyval(denr,z)) ;
%end ;

%a faster method is usually (depending on n) to use FFT
amplitude_spectrum = 1./max( abs(fft([1,-a,zeros(1,FFT_size-AR_model_order-1)])) , 1e-30*ones(1,FFT_size) );

%Scaling is necessary so that the output matches the fft command
amplitude_spectrum = amplitude_spectrum*sqrt(FFT_size) ;
