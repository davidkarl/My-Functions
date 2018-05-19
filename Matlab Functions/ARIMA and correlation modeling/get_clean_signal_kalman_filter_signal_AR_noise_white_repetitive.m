function [output_signal] = get_clean_signal_kalman_filter_signal_AR_noise_white_repetitive(...
                                                    input_signal,R_noise_covariance,ARX_model_order)
% This function takes the corrupted signal z=signal+noise, and noise
% covariance. Noise is assumed to be white, Gaussian, with covariance r.
% Estimate y of the signal using repetitive bidirectional nonstationary 
% Kalman filter.  n is the model order.
% Usage: y=arkalmr(z,r,[n]) ;

if nargin<3 
    ARX_model_order = 4; 
end


max_number_of_repetitions = 50;
min_number_of_repetitions = 2;
iterations_counter = 0; 
output_signal = input_signal;
total_error_previous = 0;
input_signal_length = length(input_signal);
lng = min(input_signal_length,256);

while iterations_counter < max_number_of_repetitions
    % cs=corrs(y,lng) ;
    % cs=cs(1:lng/2) ;
    % as=burg1(y,n) ;
    % cr=a2c(as,lng/2) ;
    % bs=sqrt(cr(2:lng/2)\cs(2:lng/2)) ;
    % g=1./abs(fft([1 -as zeros(1,lng-n-1)])) ; % spectrum of the AR system
    % g=(g.^2)*lng ;
    % s=(abs(fft(y))).^2 ;
    % g=g-ones(1,lng)*sum(g)/lng ;
    % s=s-ones(1,lng)*sum(s)/lng ; % detrend
    % bs=sqrt(s/g)
    % bs3=sqrt(sum(s.*g)/sum(g.*g)) ;

    %get current estimate AR parameters:
    [AR_parameters_signal, g_signal] = get_ARX_parameters(output_signal,ARX_model_order);
    
    %track previous signal estimate:
    output_signal_previous = output_signal;
        
    %use kalman filter again to estimate clean signal:
    [output_signal, total_error] = get_clean_signal_kalman_filter_with_signal_AR_and_noise_white(...
                                        input_signal,AR_parameters_signal,g_signal,R_noise_covariance);
    
    %if total error has risen then finish the process with the previous estimate as the final:
    if iterations_counter > min_number_of_repetitions && total_error > total_error_previous 
        iterations_counter = max_number_of_repetitions;
    end
    total_error_previous = total_error;
    iterations_counter = iterations_counter+1;
end

output_signal = output_signal_previous;



