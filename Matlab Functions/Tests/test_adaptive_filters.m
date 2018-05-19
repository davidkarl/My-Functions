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

clear all;

%Parameters:
number_of_ensemble_realizations = 100; %number of realizations within the ensemble
N_number_of_iterations = 500; %number of iterations
Wo = [0.32+0.21*1i,-0.3+0.7*1i,0.5-0.8*1i,0.2+0.5*1i].'; %unknown system
noise_variance = 0.04; %noise power
number_of_coefficients = 4; %number of coefficients of the adaptive filter
mu_convergence_factor = 0.1; %convergence factor (step)  (0 < mu < 1)
%LMS-newton:
inverse_Rx_estimate = 0.1*eye(number_of_coefficients);
Rx_estimate_smoothing_factor = 0.05;
%T-domain:
T = (1/sqrt(number_of_coefficients))*dftmtx(number_of_coefficients);
gamma_normalization_factor = 1e-12;
alpha_smoothing_factor = 0.1;
initial_power = 0;
%Affine Projection:
L_samples_in_memory = 1;
%RLS:
delta_initial_autocorrelation_matrix_factor = 0.2;
lambda_smoothing_factor = 0.97;
%SM-Affine Projection:
gamma_bar_error_bound = sqrt(5*noise_variance);
gamma_bar_error_bound_vec = sqrt(5*noise_variance)*ones(L_samples_in_memory+1,1);
%SM-AP-PU-Simplified:
coefficients_to_be_updated_each_iteration = randi([0,1],number_of_coefficients,N_number_of_iterations);  
%Lattice-RLS-aposteriori:
ladder_vector = zeros(number_of_coefficients,N_number_of_iterations,number_of_ensemble_realizations);    % ladder coefficients of the algorithm
kappa_vector = zeros(number_of_coefficients,N_number_of_iterations,number_of_ensemble_realizations);    % reflection coefficients of the lattice algorithm
aposteriori_error_over_time = zeros(number_of_coefficients+1,N_number_of_iterations,number_of_ensemble_realizations);    % error matrix   
initial_aposteriori_error_small = 10^-2;
flag_lattice_filter = 1;
flag_real_or_complex = 1;
%Lattice-RLS-apriori:
ladder_vector = zeros(number_of_coefficients,N_number_of_iterations+1,number_of_ensemble_realizations);    % ladder coefficients of the algorithm
kappa_vector = zeros(number_of_coefficients,N_number_of_iterations,number_of_ensemble_realizations);    % reflection coefficients of the lattice algorithm
aposteriori_error_over_time = zeros(number_of_coefficients+1,N_number_of_iterations,number_of_ensemble_realizations);    % error matrix   
initial_aposteriori_error_small = 10^-2;
flag_lattice_filter = 1;
flag_real_or_complex = 1;
%Lattice-NRLS-aposteriori:
ladder_vector = zeros(number_of_coefficients,N_number_of_iterations,number_of_ensemble_realizations);    % ladder coefficients of the algorithm
kappa_vector = zeros(number_of_coefficients,N_number_of_iterations,number_of_ensemble_realizations);    % reflection coefficients of the lattice algorithm
aposteriori_error_over_time = zeros(number_of_coefficients+1,N_number_of_iterations,number_of_ensemble_realizations);    % error matrix   
initial_aposteriori_error_small = 10^-2;
flag_lattice_filter = 1;
flag_real_or_complex = 1;
%Lattice-RLS-error-feedback:
ladder_vector = zeros(number_of_coefficients,N_number_of_iterations+1,number_of_ensemble_realizations);    % ladder coefficients of the algorithm
kappa_vector = zeros(number_of_coefficients,N_number_of_iterations,number_of_ensemble_realizations);    % reflection coefficients of the lattice algorithm
aposteriori_error_over_time = zeros(number_of_coefficients+1,N_number_of_iterations,number_of_ensemble_realizations);    % error matrix   
initial_aposteriori_error_small = 10^0;
flag_lattice_filter = 1;
flag_real_or_complex = 1;
%Fast-Transversal-RLS:
aposteriori_error_over_time = zeros(number_of_coefficients+1,N_number_of_iterations,number_of_ensemble_realizations);    % error matrix   
initial_aposteriori_error_small = 10^0;
flag_lattice_filter = 0;
flag_real_or_complex = 1;
%IIR-RLS-Error-Equation:
flag_IIR_filter = 1;
flag_lattice_filter = 0;
numerator_order = 3;
denominator_order = 2;
delta_regularization = 10^-5;
Wo = real(Wo);
%IIR-RLS-Gauss-Newton:
mu_step_size = 0.1;
delta_regularization = 10^-2; 
%IIR-RLS-Error-Equation:
delta_regularization = 10^-5;
%IIR-RLS-Steiglitz-McBride:
mu_step_size = 0.1;

%Initializing & Allocating memory:
filter_coefficients_over_time = ones(number_of_coefficients , N_number_of_iterations+1 , number_of_ensemble_realizations);   % coefficient vector for each iteration and realization; w(0) = [1 1 1 1].'
MSE_over_time = zeros(N_number_of_iterations,number_of_ensemble_realizations);% MSE for each realization
MSEmin_over_time = zeros(N_number_of_iterations,number_of_ensemble_realizations);% MSE_min for each realization


%Computing:
for l=1:number_of_ensemble_realizations
    
    %Initialize input signal to be used by real system weights W0 and desired signal:
    auxiliary_current_input_signal_last_few_samples = zeros(number_of_coefficients,1); %input at a certain iteration (tapped delay line)
    desired_signal = zeros(N_number_of_iterations,1);
    
    %Create actual current input signal and noise vec:
    if flag_real_or_complex==1
        input_signal = (sign(randn(N_number_of_iterations,1)))./sqrt(2);
        sigma_x2 = var(input_signal);
        noise_vec = sqrt(noise_variance/2)*(randn(N_number_of_iterations,1));
    else
        input_signal = (sign(randn(N_number_of_iterations,1)) + 1i*sign(randn(N_number_of_iterations,1)))./sqrt(2);
        sigma_x2 = var(input_signal);
        noise_vec = sqrt(noise_variance/2)*(randn(N_number_of_iterations,1)+1i*randn(N_number_of_iterations,1));
    end
    
    %build desired signal using decided upon system transfer function or W0:
    for k=1:N_number_of_iterations
        auxiliary_current_input_signal_last_few_samples = [input_signal(k,1) ; auxiliary_current_input_signal_last_few_samples(1:(number_of_coefficients-1),1)];
        desired_signal(k) = (Wo'*auxiliary_current_input_signal_last_few_samples(:,1))+noise_vec(k);
    end
    
    %Initialize adaptive filter initial guesses/parameters:
    initial_coefficients = filter_coefficients_over_time(:,1,l);
    filter_order = number_of_coefficients-1;
    
    %Use adaptive filters:
%     %LMS:
%     [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time(:,:,l)] = ...
%     adaptive_filter_LMS(desired_signal,input_signal,filter_order,initial_coefficients,mu_convergence_factor);
%     %LMS-newton: 
%     [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time(:,:,l)] = ...
%     adaptive_filter_LMS_newton(desired_signal,input_signal,filter_order,initial_coefficients,inverse_Rx_estimate,Rx_estimate_smoothing_factor,mu_convergence_factor);
%     %transform-domain:
%     [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time_in_T_domain,filter_coefficients_over_time(:,:,l)] = ...
%     adaptive_filter_transform_domain(desired_signal,input_signal,filter_order,initial_coefficients,T,mu_convergence_factor,gamma_normalization_factor,alpha_smoothing_factor,initial_power);
%     %transform-domain-DFT:
%     [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time_in_T_domain,filter_coefficients_over_time(:,:,l)] = ...
%     adaptive_filter_DFT_domain(desired_signal,input_signal,filter_order,initial_coefficients,mu_convergence_factor,gamma_normalization_factor,alpha_smoothing_factor,initial_power);
%     %NLMS:
%     [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time(:,:,l)] = ...
%     adaptive_filter_NLMS(desired_signal,input_signal,filter_order,initial_coefficients,gamma_normalization_factor,mu_convergence_factor);
%     %RLS alternative:
%     [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time(:,:,l),aposteriori_output_over_time,aposteriori_error_over_time] =  ...
%     adaptive_filter_RLS_alternative(desired_signal,input_signal,filter_order,delta_initial_autocorrelation_matrix_factor,lambda_smoothing_factor);
%     %RLS:
%     [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time(:,:,l),aposteriori_output_over_time,aposteriori_error_over_time] =  ...
%     adaptive_filter_RLS(desired_signal,input_signal,filter_order,delta_initial_autocorrelation_matrix_factor,lambda_smoothing_factor);
%     %Affine Projection: 
%     [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time(:,:,l)] = ...
%     adaptive_filter_affine_projection(desired_signal,input_signal,filter_order,initial_coefficients,gamma_normalization_factor,L_samples_in_memory,mu_convergence_factor);
%     %SM-Affine Projection:
%     [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time(:,:,l),number_of_coefficients_updates_done] = ...
%     adaptive_filter_SM_AP(desired_signal,input_signal,filter_order,initial_coefficients,gamma_bar_error_bound,gamma_bar_error_bound_vec,gamma_normalization_factor,L_samples_in_memory);
%     %SM-NLMS:
%      [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time(:,:,l),number_of_coefficients_updates_done] = ...
%     adaptive_filter_SM_NLMS(desired_signal,input_signal,filter_order,initial_coefficients,gamma_bar_error_bound,gamma_normalization_factor);
%     %SM-BNLMS:
%     [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time(:,:,l),number_of_coefficients_updates_done] = ...
%     adaptive_filter_SM_BNLMS(desired_signal,input_signal,filter_order,initial_coefficients,gamma_bar_error_bound,gamma_normalization_factor);
%     %SM-AP-Simplified:
%     [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time(:,:,l),number_of_coefficients_updates_done] = ...
%     adaptive_filter_SM_AP_simplified(desired_signal,input_signal,filter_order,initial_coefficients,gamma_bar_error_bound,gamma_normalization_factor,L_samples_in_memory);
%     %SM-AP-UP-Simplified:
%     [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time(:,:,l),number_of_coefficients_updates_done] = ...
%     adaptive_filter_SM_AP_PU_simplified(desired_signal,input_signal,filter_order,initial_coefficients,gamma_bar_error_bound,gamma_normalization_factor,coefficients_to_be_updated_each_iteration,L_samples_in_memory);
%     %Lattice-RLS-aposteriori-errors:
%     [ladder_vector(:,:,l), kappa_vector(:,:,l), aposteriori_error_over_time(:,:,l)] = ...
%         adaptive_filter_lattice_RLS_aposteriori_errors(...
%              desired_signal, input_signal, filter_order, lambda_smoothing_factor, initial_aposteriori_error_small);
%     filter_error_over_time(:,l) = aposteriori_error_over_time(end,:,l);
%     %Lattice-RLS-apriori-errors:
%     [ladder_vector(:,:,l), kappa_vector(:,:,l), aposteriori_error_over_time(:,:,l)] = ...
%         adaptive_filter_lattice_RLS_apriori_errors(...
%              desired_signal, input_signal, filter_order, lambda_smoothing_factor, initial_aposteriori_error_small);
%     filter_error_over_time(:,l) = aposteriori_error_over_time(end,:,l);
%     %Lattice-NRLS-aposteriori-errors:
%     [ladder_vector(:,:,l), kappa_vector(:,:,l), aposteriori_error_over_time(:,:,l)] = ...
%         adaptive_filter_lattice_NRLS_aposteriori_errors(...
%              desired_signal, input_signal, filter_order, lambda_smoothing_factor, initial_aposteriori_error_small);
%     filter_error_over_time(:,l) = aposteriori_error_over_time(end,:,l);
%     %Lattice-RLS-apriori-errors:
%     [ladder_vector(:,:,l), kappa_vector(:,:,l), aposteriori_error_over_time(:,:,l)] = ...
%         adaptive_filter_lattice_error_feedback(...
%              desired_signal, input_signal, filter_order, lambda_smoothing_factor, initial_aposteriori_error_small);
%     filter_error_over_time(:,l) = aposteriori_error_over_time(end,:,l);
%     %Fast-Transversal-RLS:
%     [aposteriori_error_vec,...
%           apriori_error_vec,...
%           filter_coefficients_over_time(:,:,l)] = adaptive_filter_fast_transversal_RLS(...
%           desired_signal,input_signal,filter_order , lambda_smoothing_factor, initial_aposteriori_error_small);
%      filter_error_over_time(:,l) = apriori_error_vec; 
%     %Fast-Transversal-RLS-Stabilized:
%     [aposteriori_error_vec,...
%           apriori_error_vec,...
%           filter_coefficients_over_time(:,:,l)] = adaptive_filter_fast_transversal_RLS_stabilized(...
%           desired_signal,input_signal,filter_order , lambda_smoothing_factor, initial_aposteriori_error_small);
%      filter_error_over_time(:,l) = apriori_error_vec; 
%      %IIR-RLS:
%      [output_signal_final, ...
%           filter_error_over_time(:,l),...
%           theta_total_coefficients] = adaptive_filter_IIR_RLS(...
%           desired_signal,input_signal,numerator_order,denominator_order,lambda_smoothing_factor,delta_regularization);
%      %IIR-RLS-Error-Equation:
%      [filter_desired_output_error, ...
%           filter_output_input_error,...
%           theta_total_coefficients,...
%           filter_desired_input_error] = adaptive_filter_IIR_RLS_error_equations(...
%             desired_signal,input_signal,numerator_order,denominator_order,lambda_smoothing_factor,delta_regularization);
%      filter_error_over_time(:,l) = filter_desired_output_error;
%      %IIR-RLS-Gauss-Newton:
%      [output_signal_final, ...
%           filter_error_over_time(:,l),...
%           theta_total_coefficients] = adaptive_filter_IIR_RLS_gauss_newton(...
%           desired_signal,input_signal,numerator_order,denominator_order,mu_step_size,lambda_smoothing_factor,delta_regularization);
%     %IIR-RLS-Gauss-Newton-Gradient-Based:
%     [output_signal_final, ...
%           filter_error_over_time(:,l),...
%           theta_total_coefficients] = adaptive_filter_IIR_RLS_gauss_newton_gradient_based(...
%           desired_signal,input_signal,numerator_order,denominator_order,mu_step_size);
%     %IIR-RLS-Error-Equation:
%     [filter_output_input, ...
%           filter_desired_output_error,...
%           theta_total_coefficients,...
%           filter_desired_input_error] = adaptive_filter_IIR_RLS_error_equations(...
%             desired_signal,input_signal,numerator_order,denominator_order,lambda_smoothing_factor,delta_regularization);
%     filter_error_over_time(:,l) = filter_desired_output_error;  
    %IIR-RLS-Steiglitz-McBride:
    [filter_output_signal, ...
          filter_output_error,...
          theta_total_coefficients,...
          errorVector_s] = adaptive_filter_IIR_Steiglitz_McBride(...
            desired_signal,input_signal,numerator_order,denominator_order,mu_step_size);
    filter_error_over_time(:,l) = filter_output_error;  
    
    
    %get  
    MSE_over_time(:,l)    =   MSE_over_time(:,l)+(abs(filter_error_over_time(:,l))).^2;
    MSEmin_over_time(:,l) =   MSEmin_over_time(:,l)+(abs(noise_vec(:))).^2;

end

 
%   Averaging:
if flag_lattice_filter == 1
    kappa_av = sum(kappa_vector,3)/number_of_ensemble_realizations;
    ladder_av = sum(ladder_vector,3)/number_of_ensemble_realizations;
    W_av = zeros(number_of_coefficients+1,N_number_of_iterations);                % estimate of Wo
    for k=1:N_number_of_iterations
        W_av(:,k) = latc2tf(-kappa_av(:,k),ladder_av(:,k));
    end
elseif flag_IIR_filter == 1
    theta_av = sum(theta_total_coefficients,3)/number_of_ensemble_realizations;
else
    W_av = sum(filter_coefficients_over_time,3)/number_of_ensemble_realizations;
end
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

if flag_IIR_filter == 1 
    figure
    subplot(2,1,1);
    plot(real(theta_av(1,:)));
    title('Evolution of the 1st coefficient (real part) of the numerator');
    xlabel('Number of iterations, k');
    ylabel('Coefficient');
    subplot(2,1,2);
    plot(real(theta_av(numerator_order+1,:)));
    title('Evolution of the 1st coefficient (real part) of the denominator');
    xlabel('Number of iterations, k');
    ylabel('Coefficient');
else
    figure
    subplot(2,1,1); 
    plot(real(W_av(1,:)));
    title('Evolution of the 1st coefficient (real part)');
    xlabel('Number of iterations, k'); 
    ylabel('Coefficient');
    subplot(2,1,2) 
    plot(imag(W_av(1,:)));
    title('Evolution of the 1st coefficient (imaginary part)');
    xlabel('Number of iterations, k'); 
    ylabel('Coefficient');
end

