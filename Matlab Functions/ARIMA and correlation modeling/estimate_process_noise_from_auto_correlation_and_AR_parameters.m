function [q,r] = estimate_process_noise_from_auto_correlation_and_AR_parameters(auto_correlation_sequence,AR_parameters)
% Given autocorrelation coefficients c and AR model coefficients a
% this function attempts to estimate the measurement noise r and the 
% process noise q energies
% Usage: [q,r]=getnoise(c,a)

auto_correlation_sequence=auto_correlation_sequence(:) ;
m=length(auto_correlation_sequence) ;
cr=a2c(AR_parameters,m) ; % generate reference autocorrelation coefs.
M= ([ 1 zeros(1,m-1) ; cr' ])' ;
qr=M \ auto_correlation_sequence ;
q=qr(2) ; r=qr(1) ;