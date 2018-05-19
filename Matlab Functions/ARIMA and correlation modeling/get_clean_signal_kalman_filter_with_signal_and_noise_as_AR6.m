function [y_signal_estimate, total_error]=get_clean_signal_kalman_filter_with_signal_and_noise_as_AR6(input_signal,...
    AR_parameters_signal,g_signal,auto_correlation_signal,...
    AR_parameters_noise,g_noise,auto_correlation_noise)
% This function takes the corrupted signal z=signal+noise,
% AR parameters of signal (as,bs) and noise (an,bn)
% and signal resp. noise autocorrelations (cs,cn) and produces the smoothed
% estimate y of signal using unidirectional nonstationary Kalman filter
% te is the total forward prediction error
% it is optimized for AR models but not with respect to Matlab
% but rather with respect to the number of operations
%
% Covariance Square Root Filter
%
% Usage [y, te]=darkalms(z,as,bs,cs,an,bn,cn)

[A_state_transfer_matrix,H_measurement_matrix,Q_process_noise_covariance,R_measurement_covariance,P_state_estimate_covariance] = ...
    AR_parameters_to_system_matrices(AR_parameters_signal,g_signal,auto_correlation_signal,AR_parameters_noise,g_noise,auto_correlation_noise) ;

AR_model_order_signal = length(AR_parameters_signal);
AR_model_order_noise = length(AR_parameters_noise);
AR_model_order_total = length(A_state_transfer_matrix);
input_signal_length = length(input_signal);

% initial uninformative guess
x_state_estimate = zeros(AR_model_order_total,1);
y_signal_estimate = zeros(1,AR_model_order_total);

total_error = 0;
P_state_estimate_covariance = dlyap1(A_state_transfer_matrix,Q_process_noise_covariance);
M = chol(P_state_estimate_covariance); % M is now upper triangular
suare_root_R = sqrt(R_measurement_covariance);
Qs = diag(sqrt(diag(Q_process_noise_covariance)));

%Forward run:
for i = 1:input_signal_length
    % sum(diag(P*P'))
    %Data step:
    residual_error = input_signal(i)-(x_state_estimate(AR_model_order_noise)+x_state_estimate(AR_model_order_total)) ;          % prediction error
    total_error = total_error + residual_error^2 ;
    Z = [suare_root_R , zeros(1,AR_model_order_total) ; M*H_measurement_matrix' , M ] ;
    % [zq Z]=qr(Z) ;
    Z = chol(Z'*Z);
    kalman_gain = Z(1,2:AR_model_order_total+1)'/Z(1,1);
    M = Z(2:AR_model_order_total+1,2:AR_model_order_total+1);
    
    % L=(P(:,n)+P(:,nn)) / (P(n,n)+P(ns,ns)+2*P(ns,n)+r) ; % Kalman gain
    
    x_state_estimate = x_state_estimate + kalman_gain*residual_error ;             % correct the state estimate
    
    % P=P-L*C*P ;
    %  P=P-L*pom*L' ;
    %  IL=eye(n)-L*C ;
    %  P=IL*P ;
    %  P=IL*P*IL'+L*r*L' ;
    % P=P-L*(P(n,:)+P(nn,:)) ;
    
    y_signal_estimate(i) = x_state_estimate(AR_model_order_total) ;
    
    % Time step
    x_state_estimate = A_state_transfer_matrix*x_state_estimate ;
    % xnew=A(n,nn+1:n)*x(nn+1:n) ;
    % x(nn+1:n-1)=x(nn+2:n) ;
    % x(n)=xnew ;
    % xnew=A(nn,1:nn)*x(1:nn) ;
    % x(1:nn-1)=x(2:nn) ;
    % x(nn)=xnew ;
    
    % PfA=P*A' ;
    % P=A*PfA+Q ;
    Z = [M*A_state_transfer_matrix' ; Qs] ;
    %  [zq Z]=qr(Z) ;
    Z = chol(Z'*Z) ;
    M = Z(1:AR_model_order_total,:) ;
end




