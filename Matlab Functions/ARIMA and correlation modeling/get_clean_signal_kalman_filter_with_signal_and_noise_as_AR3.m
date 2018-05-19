function [y_signal_estimate] = get_clean_signal_kalman_filter_with_signal_and_noise_as_AR3(input_signal,...
    AR_parameters_signal,g_signal,auto_correlation_signal,...
    AR_parameters_noise,g_noise,auto_correlation_noise)
% This function takes the corrupted signal z=signal+noise,
% AR parameters of signal (as,bs) and noise (an,bn)
% and signal resp. noise autocorrelations (cs,cn) and produces the smoothed
% estimate y of signal using bidirectional nonstationary Kalman filter
% It uses the modified smoothing formulas and is therefore quicker than darkalm
% it is optimized for AR models but not as much as darklam3

[A_state_transition_matrix,H_measurement_matrix,Q_process_noise_covariance,R_measurement_covariance,P_state_estimate_covariance] = ...
    get_state_space_model_using_signal_and_noise_AR_parameters(...
                                        AR_parameters_signal,...
                                        g_signal,...
                                        auto_correlation_signal,...
                                        AR_parameters_noise,...
                                        g_noise,...
                                        auto_correlation_noise); 


% r=(cn(1)+cs(1))*1e-6 ;

AR_model_order_signal = length(AR_parameters_signal);
AR_model_order_noise = length(AR_parameters_noise);
state_vector_length = length(A_state_transition_matrix);
input_signal_length = length(input_signal);

%initial uninformative guess:
x_state_estimate = zeros(state_vector_length,1) ;
x_state_estimate_total_filtered1 = zeros(state_vector_length,input_signal_length) ;
Pp = zeros(input_signal_length,state_vector_length*state_vector_length) ;
kalman_gain_over_time = zeros(input_signal_length,state_vector_length) ;
pomp = zeros(input_signal_length) ;
y_signal_estimate = zeros(1,state_vector_length) ;
residual_error_over_time = zeros(input_signal_length) ;


%Forward run
for i = 1:input_signal_length
    % Data step
    % e=z(i)-C*x ;
    residual_error = input_signal(i)-(x_state_estimate(AR_model_order_noise)+x_state_estimate(state_vector_length)) ;          % prediction error
    residual_error_over_time(i) = residual_error ;
    
    % K=C' / (C*P*C'+r) ;
    % L=P*K ;
    pom =  1 / (P_state_estimate_covariance(state_vector_length,state_vector_length) + ...
                P_state_estimate_covariance(AR_model_order_signal,AR_model_order_signal) + ...
                2*P_state_estimate_covariance(AR_model_order_signal,state_vector_length) + ...
                R_measurement_covariance);
    
    pomp(i) = pom;
    K = H_measurement_matrix' * pom;
    kalman_gain = (P_state_estimate_covariance(:,state_vector_length) + ...
                    P_state_estimate_covariance(:,AR_model_order_noise)) * pom  ; % Kalman gain
    kalman_gain_over_time(i,:) = kalman_gain' ;
    x_state_estimate = x_state_estimate + kalman_gain*residual_error ;             % correct the state estimate
    
    % P=P-L*C*P ;
    P_state_estimate_covariance = P_state_estimate_covariance - ...
        kalman_gain*(P_state_estimate_covariance(state_vector_length,:) + ...
                     P_state_estimate_covariance(AR_model_order_noise,:));
    
    x_state_estimate_total_filtered1(:,i) = x_state_estimate ;              % filtered estimate
    Pp(i,:) = reshape(P_state_estimate_covariance,1,state_vector_length*state_vector_length) ;
    
    % Time step
    x_state_estimate = A_state_transition_matrix * x_state_estimate ;
    P_state_estimate_covariance = A_state_transition_matrix*P_state_estimate_covariance*A_state_transition_matrix'+Q_process_noise_covariance ;
end

%Backward run:
lmb = zeros(state_vector_length,1) ;
for i = input_signal_length:-1:1
    K = H_measurement_matrix' * pom;
    kalman_gain = kalman_gain_over_time(i,:)';
    P_state_estimate_covariance = reshape(Pp(i,:),state_vector_length,state_vector_length);
    Almb = A_state_transition_matrix' * lmb;
    PAlmb = P_state_estimate_covariance * Almb;
    x_state_estimate = x_state_estimate_total_filtered1(:,i)+PAlmb;
    y_signal_estimate(i) = x_state_estimate(state_vector_length);
    % y(i)=xf(n,i) ; % this would use only the forward run
    
    %  lmb=Almb+(C'/r*C*Lk(i,:)'+K)*((1-C*Lk(i,:)')*ek(i)-C*PAlmb) ;
    %  lmb=Almb+(C'*(L(n)+L(nn))/r+K)*((1-L(n)-L(nn))*ek(i)-PAlmb(n)-PAlmb(nn)) ;
    
    Ls = kalman_gain(state_vector_length) + kalman_gain(AR_model_order_noise); 
    Lr = Ls/R_measurement_covariance;
    KLr = K; 
    KLr(state_vector_length) = KLr(state_vector_length) + Lr; 
    KLr(AR_model_order_noise) = KLr(AR_model_order_noise) + Lr;
    lmb = Almb + KLr * ((1-Ls)*residual_error_over_time(i)-PAlmb(state_vector_length)-PAlmb(AR_model_order_noise));
end

% Here we check for overflows
sy=sum(abs(y_signal_estimate))/input_signal_length ;
if isnan(sy) || sy>10*max(abs(input_signal)),
    y_signal_estimate=x_state_estimate_total_filtered1(state_vector_length,:) ;
end ;



