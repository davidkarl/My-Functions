function [AR_parameters] = get_ARX_parameters_only_AR(input_signal,AR_model)
% Identifies ARX model of the n-th order using data y
% uses LSM - one time identification
% returns vector of n a coefficients
% idarx

input_signal_length = length(input_signal);

z = zeros(input_signal_length-AR_model,AR_model);

b = input_signal(AR_model+1:input_signal_length);

for i = 1:input_signal_length-AR_model
    for j=1:AR_model
        z(i,j) = input_signal(i-j+AR_model);
    end
    b(i) = input_signal(i+AR_model);
end

% a=(z\b')'
% size(z), size(b)

AR_parameters = (z\(b'))' ;

