%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Example: System Identification                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                               %
%  In this example we have a typical system identification scenario. We want    %
% to estimate the filter coefficients of an unknown system given by Wo. In      %
% order to accomplish this task we use an adaptive filter with the same         %
% number of coefficients, N, as the unkown system. The procedure is:            %
% 1)  Excitate both filters (the unknown and the adaptive) with the signal      %
%   x. In this case, x is chosen according to the 4-QAM constellation.          %
%   The variance of x is normalized to 1.                                       %
% 2)  Generate the desired signal, d = Wo' x + n, which is the output of the    %
%   unknown system considering some disturbance (noise) in the model. The       %
%   noise power is given by sigma_n2.                                           %
% 3)  Choose an adaptive filtering algorithm to govern the rules of coefficient %
%   updating.                                                                   %
%                                                                               %
%     Adaptive Algorithm used here: LMS                                         %
%                                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%   Definitions:
number_of_ensemble_realizations = 100; %number of realizations within the ensemble
N_number_of_iterations = 500; %number of iterations
Wo = [0.32+0.21*1i,-0.3+0.7*1i,0.5-0.8*1i,0.2+0.5*1i].'; %unknown system
noise_variance = 0.04; %noise power
number_of_coefficients = 4; %number of coefficients of the adaptive filter
mu_convergence_factor = 0.1; %convergence factor (step)  (0 < mu < 1)


%Initializing & Allocating memory:
filter_coefficients_over_time = ones(number_of_coefficients , N_number_of_iterations+1 , number_of_ensemble_realizations);   % coefficient vector for each iteration and realization; w(0) = [1 1 1 1].'
MSE_over_time = zeros(N_number_of_iterations,number_of_ensemble_realizations);% MSE for each realization
MSEmin_over_time = zeros(N_number_of_iterations,number_of_ensemble_realizations);% MSE_min for each realization


%Computing:
for l=1:number_of_ensemble_realizations
    
    %Initialize input signal to be used by real system weights W0 and desired signal:
    auxiliary_current_input_signal_last_few_samples = zeros(number_of_coefficients,1); %input at a certain iteration (tapped delay line)
    desired_signal = zeros(1,N_number_of_iterations);
    
    %Create actual current input signal and noise vec:
    input_signal = (sign(randn(N_number_of_iterations,1)) + 1i*sign(randn(N_number_of_iterations,1)))./sqrt(2);
    sigma_x2 = var(input_signal);
    noise_vec = sqrt(noise_variance/2)*(randn(N_number_of_iterations,1)+1i*randn(N_number_of_iterations,1));
    
    %build desired signal using decided upon system transfer function or W0:
    for k=1:N_number_of_iterations
        auxiliary_current_input_signal_last_few_samples = [input_signal(k,1) ; auxiliary_current_input_signal_last_few_samples(1:(number_of_coefficients-1),1)];
        desired_signal(k) = (Wo'*auxiliary_current_input_signal_last_few_samples(:,1))+noise_vec(k);
    end
    
    %Use adaptive filter to get current system:
    initial_coefficients = filter_coefficients_over_time(:,1,l);
    filter_order = number_of_coefficients-1;
    [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time(:,:,l)] = ...
    adaptive_filter_LMS(desired_signal,input_signal,filter_order,initial_coefficients,mu_convergence_factor);
    
    %get 
    MSE_over_time(:,l)    =   MSE_over_time(:,l)+(abs(filter_error_over_time(:,1))).^2;
    MSEmin_over_time(:,l) =   MSEmin_over_time(:,l)+(abs(noise_vec(:))).^2;

end


%   Averaging:
W_av = sum(filter_coefficients_over_time,3)/number_of_ensemble_realizations;
MSE_av = sum(MSE_over_time,2)/number_of_ensemble_realizations;
MSEmin_av = sum(MSEmin_over_time,2)/number_of_ensemble_realizations;


%   Plotting:
figure,
plot(1:N_number_of_iterations,10*log10(MSE_av),'-k');
title('Learning Curve for MSE');
xlabel('Number of iterations, k'); ylabel('MSE [dB]');

figure,
plot(1:N_number_of_iterations,10*log10(MSEmin_av),'-k');
title('Learning Curve for MSEmin');
xlabel('Number of iterations, k'); ylabel('MSEmin [dB]');

figure,
subplot 211, plot(real(W_av(1,:))),...
title('Evolution of the 1st coefficient (real part)');
xlabel('Number of iterations, k'); ylabel('Coefficient');
subplot 212, plot(imag(W_av(1,:))),...
title('Evolution of the 1st coefficient (imaginary part)');
xlabel('Number of iterations, k'); ylabel('Coefficient');

