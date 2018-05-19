% 
% Matlab script demonstration from the textbook: 
% Kalman Filtering: Theory and Practice, 
% by M. S. Grewal and A. P. Andrews, 
% published by Wiley,2000  
% This demonstration uses the problem in Example 4.3 
% to demonstrate the effects of 
% approximations commonly used in converting 
% continuous time models to discrete time models. 
% 
clf; 
disp('Grewal & Andrews,'); 
disp('Kalman Filtering: Theory and Practice'); 
disp('published by Wiley,2000.'); 
disp(' '); 
disp('Demonstration of errors due to first-order'); 
disp('approximations in converting a continuous time'); 
disp('model with a differential equation to a discrete'); 
disp('time model with a state transition matrix.'); 
disp('(See Example 4.3)'); 
disp(' '); 
% 
% Approximations include the following three: 
% 
% 1   Approximating the state transition matrix (Phi) 
%     by the first two terms in the power series 
%     expansion of Phi = exp(T*F), which is 
% 
%        Phia = I + T*F, 
% 
%     where 
% 
%       Phia is the approximated value of Phi 
%       I    is a 2x2 identity matrix 
%       T    is the sample interval in seconds 
%       F    is the dynamic coefficient matrix 
% 
% 2   Approximating the forcing step in discrete time 
%     as delta = T*f, where f is the forcing function 
%     in the differential equation x-dot = F x + f and 
%     delta is the additive forcing step: 
%        x(k+1) = Phi * x(k) + delta 
%     in the discrete time model. 
% 
% 3   Approximating the additive process noise covariance 
%     as Qd = T* Qc, where Qd is the process noise 
%     covariance the discrete time Riccati equation  
%        P(k+1) = Phi*P(k)*Phi^T + Qd 
%     and Qc is the covariance in the continuous equation 
%        P-dot = F*P + P*F^T + Qc 
%     The exact solution is defined by an integral 
%                      T 
%        Qd = Phi*[ int  e^(-sF)*Qc*e^(-sF^T) ds ]*Phi^T 
%                      0 
% 
% The parameters of the problem are: 
% 

%Initialize basic damped harmonic oscillator and dynamic process parameters:
I = eye(2);% (identity matrix) 
T_sample_interval = 0.02;   % intersample time interval (s)  
qc_continuous_process_noise_variance = 4.47;  % continuous process noise covariance (ft^2/s^3) 
R_measurement_noise_variance = 0.01;   % measurement noise variance (s*ft^2) 
zeta_damping_factor = 0.2;   % damping factor (unitless) 
omega0_undamped_resonant_frequency = 5;     % undamped resonant frequency (rad/s) 
f_non_homogenous_force_term = [0;12];% non-homogeneous forcing (constant) 
H_measurement_matrix = [1,0]; % measurement sensitivity matrix 

% The dynamic coefficient matrix is a function of these:  
F = [0 , 1;...
    -omega0_undamped_resonant_frequency^2 , -2*zeta_damping_factor*omega0_undamped_resonant_frequency]; 

% Approximated discrete time model parameters:
Phi_state_transition_matrix_approximation   = I + T_sample_interval*F; % approximated Phi 
delta_forcing_term_approximation = T_sample_interval*f_non_homogenous_force_term;     % approximated forcing step 
Qd_process_noise_covariance_approximation    = [0,0;0,qc_continuous_process_noise_variance*T_sample_interval]; 

% Exact values of the model parameters: 
lambda = exp(-T_sample_interval*omega0_undamped_resonant_frequency*T_sample_interval); 
psi    = 1 - zeta_damping_factor^2; 
xi     = sqrt(psi); 
theta  = xi*omega0_undamped_resonant_frequency*T_sample_interval; 
c      = cos(theta); 
s      = sin(theta); 
Phi_state_transition_exact    = zeros(2); 

% The exact solution for exp(T*F) is  
Phi_state_transition_exact(1,1) = lambda*c + zeta_damping_factor*s/xi; 
Phi_state_transition_exact(1,2) = lambda*s/(omega0_undamped_resonant_frequency*xi); 
Phi_state_transition_exact(2,1) = -omega0_undamped_resonant_frequency*lambda*s/xi; 
Phi_state_transition_exact(2,2) = lambda*c - zeta_damping_factor*s/xi; 

% and the exact forcing term is (check this): 
delta_forcing_term_exact = f_non_homogenous_force_term(2)*[(1 - lambda*(c-zeta_damping_factor*s*xi/psi))/omega0_undamped_resonant_frequency^2; 
              lambda*s/(omega0_undamped_resonant_frequency*xi)]; 
 
% discrete time process noise covariance : 
l1  = zeta_damping_factor^2; 
l4  = exp(-2*omega0_undamped_resonant_frequency*zeta_damping_factor*T_sample_interval); 
l6  = sqrt(1-l1); 
l8  = l6*omega0_undamped_resonant_frequency*T_sample_interval; 
l9  = cos(2*l8); 
l11 = l4*l9*l1; 
l15 = l4*sin(2*l8)*l6*zeta_damping_factor; 
l19 = 1/(l1-1); 
l20 = omega0_undamped_resonant_frequency^2; 
l24 = 1/zeta_damping_factor; 
l32 = l4*(l9-1)*l19/l20; 
Qd_process_noise_covariance_exact  = (qc_continuous_process_noise_variance/4)*[(l1-l11+l15-1+l4)*l19*l24,l32; 
             l32,(l1-l11-l15-1+l4)*l19*l24/omega0_undamped_resonant_frequency]; 
disp(' '); 
disp('Exact and approximated values of state transition matrix (Phi)'); 
disp([Phi_state_transition_exact,Phi_state_transition_matrix_approximation]); 
disp('Exact and approximated values of forcing step (delta)'); 
disp([delta_forcing_term_exact,delta_forcing_term_approximation]); 
disp('Exact and approximated values of process noise cov. (Qd)'); 
disp([Qd_process_noise_covariance_exact,Qd_process_noise_covariance_approximation]); 
disp('(See the script for details.)'); 
disp(' '); 
disp('(Allow a moment for simulation.)'); 


% Initial values: 
x_state_exact    = [0;0];       % "true" state value 
x_state_approximation   = [0;0];       % approximated value 
P_estimation_covariance_exact    = [2,0;0,2];   % covariance of estimation uncertainty 
P_estimation_covariance_approximation   = [2,0;0,2];   % ditto, approximated 
process_noise   = [sqrt(P_estimation_covariance_exact(1,1))*randn(1) ; sqrt(P_estimation_covariance_exact(2,2))*randn(1)]; 
x_state_exact_estimated   = x_state_exact + process_noise;  % estimated state 
x_state_approximation_estimated  = x_state_approximation + process_noise; % approximated state 

% The following are for pseudonoise scaling:
RMS_measurement_noise = sqrt(R_measurement_noise_variance);     % RMS measurement noise 
RMS_process_noise = zeros(2);    % Process noise weighting matrix... 
RMS_process_noise(2,2) = sqrt(Qd_process_noise_covariance_exact(2,2)); % is upper triangular... 
RMS_process_noise(1,2) = Qd_process_noise_covariance_exact(1,2)/RMS_process_noise(2,2); % cholesky factor... 
RMS_process_noise(1,1) = sqrt(Qd_process_noise_covariance_exact(1,1)-RMS_process_noise(1,2)^2); % of Qd. 

% Initialize arrays for simulated values: 
m=0;
time_vec_of_consecutive_apriori_and_aposteriori_values=0;
time_vec=0;
x_true_state1_over_time=0;
x_true_state2_over_time=0;
x_approximation_state1_over_time=0;
x_approximation_state2_over_time=0;
K_gain_exact1_over_time=0;
K_gain_exact2_over_time=0; 

z_measurement_over_time=0;
x_state_exact_estimated_apriori_and_aposteriori1=0;
x_state_exact_estimated_apriori_and_aposteroiri2=0;
u1=0;
l1=0;
u2=0;
l2=0;
sn1=0;
sn2=0; 

% Simulate and save values (identified by comments) :
for k=1:101,
    time_vec(k)   = (k-1)*T_sample_interval; % time starting with t=0
    x_true_state1_over_time(k)  = x_state_exact(1);    % true state variables
    x_true_state2_over_time(k)  = x_state_exact(2);
    x_approximation_state1_over_time(k) = x_state_approximation(1);   % approximated state variables
    x_approximation_state2_over_time(k) = x_state_approximation(2);
    w_process_noise = RMS_process_noise*[randn(1);randn(1)]; % process noise
    x_state_exact = Phi_state_transition_exact*x_state_exact + delta_forcing_term_exact + w_process_noise;
    x_state_approximation = Phi_state_transition_matrix_approximation*x_state_approximation + delta_forcing_term_approximation + w_process_noise;
    zn = x_state_exact(1) + RMS_measurement_noise*randn(1); % Measurement equals true
    z_measurement_over_time(k) = zn;                    % position plus noise.
    
    % The forcing term has no effect on P.  Why?:
    %UPDATE STATE using Phi transition matrices (both exact and approximative):
    x_state_exact_estimated = Phi_state_transition_exact*x_state_exact_estimated + delta_forcing_term_exact;        % a priori estimates
    x_state_approximation_estimated = Phi_state_transition_matrix_approximation*x_state_approximation_estimated + delta_forcing_term_approximation;
    
    % Use double indexing for priori/posteriori values:
    m = m+1;
    time_vec_of_consecutive_apriori_and_aposteriori_values(m) = time_vec(k);  % time of a priori value
    x_state_exact_estimated_apriori_and_aposteriori1(m) = x_state_exact_estimated(1); % a priori estimates
    x_state_exact_estimated_apriori_and_aposteroiri2(m) = x_state_exact_estimated(2);
    x_state_approximation_estimated_apriori_and_aposteriori1(m)= x_state_approximation_estimated(1); % (approx)
    x_state_approximation_estimated_apriori_and_aposteriori2(m)= x_state_approximation_estimated(2);
    P_estimation_covariance_exact = Phi_state_transition_exact*P_estimation_covariance_exact*Phi_state_transition_exact' + Qd_process_noise_covariance_exact;
    K_gain_matrix_exact_current = P_estimation_covariance_exact*H_measurement_matrix'/(H_measurement_matrix*P_estimation_covariance_exact*H_measurement_matrix'+R_measurement_noise_variance);
    K_gain_exact1_over_time(k) = K_gain_matrix_exact_current(1); % Kalman gains
    K_gain_exact2_over_time(k) = K_gain_matrix_exact_current(2);
    P_estimation_covariance_approximation = Phi_state_transition_matrix_approximation*P_estimation_covariance_approximation*Phi_state_transition_matrix_approximation' + Qd_process_noise_covariance_approximation;   % Approx. temporal update
    K_gain_matrix_approximation_current = P_estimation_covariance_approximation*H_measurement_matrix'/(H_measurement_matrix*P_estimation_covariance_approximation*H_measurement_matrix' + R_measurement_noise_variance);
    K_gain_approximation1_over_time(k) = K_gain_matrix_approximation_current(1); % approximated Kalman gains
    K_gain_approximation2_over_time(k) = K_gain_matrix_approximation_current(2);
    sigma_estimate_uncertainty_exact1_over_time(m)  = sqrt(P_estimation_covariance_exact(1,1)); % RMS uncertainties of estimates
    sigma_estimate_uncertainty_exact2_over_time(m)  = sqrt(P_estimation_covariance_exact(2,2)); % (exact)
    sigma_estimate_uncertainty_approximation1_over_time(m) = sqrt(P_estimation_covariance_approximation(1,1)); % RMS uncertainties of estimates
    sigma_estimate_uncertainty_approximation2_over_time(m) = sqrt(P_estimation_covariance_approximation(2,2)); % (approx)
    
    % posteriori values:
    m = m+1;
    time_vec_of_consecutive_apriori_and_aposteriori_values(m) = time_vec(k);  % Time of a posteriori values.
    x_state_exact_estimated = x_state_exact_estimated + K_gain_matrix_exact_current*(zn - H_measurement_matrix*x_state_exact_estimated); % a posteriori estimate
    x_state_approximation_estimated = x_state_approximation_estimated+ K_gain_matrix_approximation_current*(zn - H_measurement_matrix*x_state_approximation_estimated); % a posteriori estimate
    x_state_exact_estimated_apriori_and_aposteriori1(m) = x_state_exact_estimated(1); % a posteriori estimates
    x_state_exact_estimated_apriori_and_aposteroiri2(m) = x_state_exact_estimated(2);
    x_state_approximation_estimated_apriori_and_aposteriori1(m) = x_state_approximation_estimated(1); % (approx)
    x_state_approximation_estimated_apriori_and_aposteriori2(m) = x_state_approximation_estimated(2);
    P_estimation_covariance_exact = P_estimation_covariance_exact - K_gain_matrix_exact_current*H_measurement_matrix*P_estimation_covariance_exact;
    P_estimation_covariance_exact = 0.5*(P_estimation_covariance_exact+P_estimation_covariance_exact'); % preserve symmetry of P
    P_estimation_covariance_approximation = P_estimation_covariance_approximation - K_gain_matrix_approximation_current*H_measurement_matrix*P_estimation_covariance_approximation;
    P_estimation_covariance_approximation = 0.5*(P_estimation_covariance_approximation+P_estimation_covariance_approximation');
    sigma_estimate_uncertainty_exact1_over_time(m) = sqrt(P_estimation_covariance_exact(1,1)); % RMS uncertainties of estimates
    sigma_estimate_uncertainty_exact2_over_time(m) = sqrt(P_estimation_covariance_exact(2,2)); % (exact)
    sigma_estimate_uncertainty_approximation1_over_time(m) = sqrt(P_estimation_covariance_approximation(1,1)); % RMS uncertainties of estimates
    sigma_estimate_uncertainty_approximation2_over_time(m) = sqrt(P_estimation_covariance_approximation(2,2)); % (approx)
    x_state_exact_estimated_apriori_and_aposteriori1(m)= x_state_exact_estimated(1); % a posteriori estimates
    x_state_exact_estimated_apriori_and_aposteroiri2(m)= x_state_exact_estimated(2);
end;
disp('Done.');
disp(' ');
disp('Behold the plots in the accompanying figure.');
disp(' ');
disp('These plots show the "true" state dynamics');
disp('and values using approximated parameters.');
disp(' ');
disp('Is the error due to approximation discernable?');
disp(' ');
disp('Press <ENTER> key to continue');
subplot(2,1,1),plot(time_vec,x_true_state1_over_time,'b',time_vec,x_approximation_state1_over_time,'y');xlabel('Time (sec)');ylabel('Position');title('Example 4.3: True (blue) and approx. (yel) state dynamics');
subplot(2,1,2),plot(time_vec,x_true_state2_over_time,'b',time_vec,x_approximation_state2_over_time,'y');xlabel('Time (sec)');ylabel('Velocity');
pause
 
% The second set of plots are the rms estimation uncertainties 
% and correlation coefficient as a function of time for the 
% the two cases (exact and approximated parameters): 
disp(' '); 
disp('Displayed plots show RMS state uncertainties'); 
disp('for the exact and approximated models.'); 
disp(' '); 
subplot(2,1,1),plot(time_vec_of_consecutive_apriori_and_aposteriori_values,sigma_estimate_uncertainty_exact1_over_time,'b',time_vec_of_consecutive_apriori_and_aposteriori_values,sigma_estimate_uncertainty_approximation1_over_time,'y');xlabel('Time (sec)');ylabel('RMS Pos.');title('Example 4.3: Uncertainties, true (blue) and approx. (yel)'); 
subplot(2,1,2),plot(time_vec_of_consecutive_apriori_and_aposteriori_values,sigma_estimate_uncertainty_exact2_over_time,'b',time_vec_of_consecutive_apriori_and_aposteriori_values,sigma_estimate_uncertainty_approximation2_over_time,'y');xlabel('Time (sec)');ylabel('RMS Vel.'); 
pause 
disp(' '); 
disp('Displayed state estimation plots are:'); 
disp(' 1. True, estimated and measured position (top).'); 
disp('    (including estimates with exact and approx. models)'); 
disp(' 2. True and estimated velocity (bottom).'); 
disp('    (including estimates with exact and approx. models)'); 
disp(' '); 
disp('NOTE: Different pseudorandom sequences will make these'); 
disp('      plots different on repeated runs.'); 
disp(' '); 
subplot(2,1,1),plot(time_vec,x_true_state1_over_time,'b',time_vec,z_measurement_over_time,'r',time_vec_of_consecutive_apriori_and_aposteriori_values,x_state_exact_estimated_apriori_and_aposteriori1,'c',time_vec_of_consecutive_apriori_and_aposteriori_values,x_state_approximation_estimated_apriori_and_aposteriori1,'y');xlabel('Time (sec)');ylabel('Position');title('True (blu), estim. (lt.blu) est/approx (yel) & meas. (red) states'); 
subplot(2,1,2),plot(time_vec,x_true_state2_over_time,'b',time_vec_of_consecutive_apriori_and_aposteriori_values,x_state_exact_estimated_apriori_and_aposteroiri2,'c',time_vec_of_consecutive_apriori_and_aposteriori_values,x_state_approximation_estimated_apriori_and_aposteriori2,'y');xlabel('Time (sec)');ylabel('Velocity'); 