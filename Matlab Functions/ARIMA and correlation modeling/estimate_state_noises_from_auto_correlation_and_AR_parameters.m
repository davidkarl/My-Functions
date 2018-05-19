function [process_noise_estimate,measurement_noise_estimate] = ...
    estimate_state_noises_from_auto_correlation_and_AR_parameters(auto_correlation_input_sequence,AR_parameters)
% Given autocorrelation coefficients c and AR model coefficients a
% this function attempts to estimate the measurement noise r and the 
% process noise q energies
% Usage: [q,r]=getnoise(c,a)

auto_correlation_input_sequence = auto_correlation_input_sequence(:) ;
auto_correlation_length = length(auto_correlation_input_sequence) ;

%Get reference auto correlation coefficicents from AR parameters:
auto_correlation_from_AR_parameters = AR_parameters_to_auto_correlation(AR_parameters,auto_correlation_length);%

%Get design matrix for lsq operation:
M_design_matrix = ([ [1,zeros(1,auto_correlation_length-1)] ; auto_correlation_from_AR_parameters' ])' ;
% M_design_matrix = [ [1;zeros(1,auto_correlation_length-1)] , auto_correlation_from_AR_parameters(:) ];

%Get noises estimates:
qr = M_design_matrix \ auto_correlation_input_sequence ;

%Assign output variables:
process_noise_estimate = qr(2); 
measurement_noise_estimate = qr(1);