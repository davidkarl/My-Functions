% Nonlinear estimation of damping coefficient. 
% 
% Example 5.3, pp. 173--174 in 
% Kalman Filtering: Theory and Practice, 
% by M. S. Grewal and A. P. Andrews, 
% published by Wiley, 2000. 
disp('See Example 5.3, pp. 173--174 in'); 
disp('Kalman Filtering: Theory and Practice,'); 
disp('by M. S. Grewal and A. P. Andrews,'); 
disp('published by Wiley, 2000.'); 
disp(' '); 
disp('Demonstration of extended Kalman filter'); 
disp('estimating the position, velocity and'); 
disp('damping factor of a damped harmonic'); 
disp('oscillator with constant forcing.'); 
disp(' '); 
disp('This would be an appropriate model for'); 
disp('a VERTICAL mass-spring system with UNKNOWN'); 
disp('constant damping and known gravity forcing.'); 
disp(' '); 
% 
% Model parameters 
% 
damping_factor_true  = 0.2;     % TRUE damping factor (unitless) 
I3    = eye(3);  % (identity matrix) 
T_sampling_interval     = 0.01;     % intersample time interval (s) 
                 % (not specified in Example 5.3) 
Q_continuous_process_noise_covariance    = 4.47;    % continuous process noise covariance (ft^2/s^3) 
R_measurement_noise_covariance     = 0.01;     % measurement noise variance (ft^2) 
omega0_undamped_resonant_frequency = 5;       % undamped resonant frequency (rad/s) 
f_non_homogeneous_forcing_term     = [0;12;0];% non-homogeneous forcing (constant) 
H_measurement_sensitivity_matrix     = [1,0,0]; % measurement sensitivity matrix 
lambda = exp(-T_sampling_interval*omega0_undamped_resonant_frequency*damping_factor_true); 
psi    = 1 - damping_factor_true^2; 
xi     = sqrt(psi); 
theta  = xi*omega0_undamped_resonant_frequency*T_sampling_interval; 
cos_theta      = cos(theta); 
sin_theta      = sin(theta); 

% This implementation uses the exact solution for 
% the matrix exponential, exp(T*F) for the state 
% transition matrix, derived using  
% symbolic math software.  The values can also be 
% calculated numerically in Matlab using Runge-Kutta 
% integration of the formulas on pp. 140-141. 
Phi_state_transition_matrix_exact    = zeros(3); 
Phi_state_transition_matrix_exact(1,1) = lambda*cos_theta + damping_factor_true*sin_theta/xi; 
Phi_state_transition_matrix_exact(1,2) = lambda*sin_theta/(omega0_undamped_resonant_frequency*xi); 
Phi_state_transition_matrix_exact(2,1) = -omega0_undamped_resonant_frequency*lambda*sin_theta/xi; 
Phi_state_transition_matrix_exact(2,2) = lambda*cos_theta - damping_factor_true*sin_theta/xi; 
Phi_state_transition_matrix_exact(3,3) = 1; 

% The discrete time forcing term is: 
delta = f_non_homogeneous_forcing_term(2)*[(1 - lambda*(cos_theta-damping_factor_true*sin_theta*xi/psi))/omega0_undamped_resonant_frequency^2; 
              lambda*sin_theta/(omega0_undamped_resonant_frequency*xi); 
              0]; 

% The EKF approximation of Qd will be T*Qc: 
Qd     = T_sampling_interval*[0,0,0;0,Q_continuous_process_noise_covariance,0;0,0,0];  
x_true_state_current     = [0;0;damping_factor_true]; 
P_state_estimate_covariance_matrix     = [2,0,0;0,2,0;0,0,2]; 
H_measurement_sensitivity_matrix     = [1,0,0]; 
R_measurement_noise_covariance     = 0.01; 
x_estimated_state_current    = [0;0;0]; 
steps_counter     = 0; 
t_vec=0;
x_true_state1_over_time=0;
x_true_state2_over_time=0;
x_true_state3_over_time=0;
x_estimated_state1_over_time=0;
x_estimated_state2_over_time=0;
x_estimated_state3_over_time=0;
P_covariance_matrix_diagonal1_over_time=0;
P_covariance_matrix_diagonal2_over_time=0;
P_covariance_matrix_diagonal3_over_time=0; 

for k=1:101,
    t_vec(k)   = T_sampling_interval*(k-1);
    x_true_state1_over_time(k)  = x_true_state_current(1);
    x_true_state2_over_time(k)  = x_true_state_current(2);
    x_true_state3_over_time(k)  = x_true_state_current(3);
    
    % a priori values:
    steps_counter      = steps_counter+1;
    t_vec_double_steps(steps_counter)  = t_vec(k);
    x_estimated_state1_over_time(steps_counter) = x_estimated_state_current(1);
    x_estimated_state2_over_time(steps_counter) = x_estimated_state_current(2);
    x_estimated_state3_over_time(steps_counter) = x_estimated_state_current(3);
    P_covariance_matrix_diagonal1_over_time(steps_counter) = P_state_estimate_covariance_matrix(1,1);
    P_covariance_matrix_diagonal2_over_time(steps_counter) = P_state_estimate_covariance_matrix(2,2);
    P_covariance_matrix_diagonal3_over_time(steps_counter) = P_state_estimate_covariance_matrix(3,3);

    % a posteriori values following observational update:
    steps_counter      = steps_counter+1;
    z      = x_true_state_current(1);
    K      = P_state_estimate_covariance_matrix*H_measurement_sensitivity_matrix'/(H_measurement_sensitivity_matrix*P_state_estimate_covariance_matrix*H_measurement_sensitivity_matrix'+R_measurement_noise_covariance);
    x_estimated_state_current     = x_estimated_state_current + K*(z-H_measurement_sensitivity_matrix*x_estimated_state_current);
    P_state_estimate_covariance_matrix      = P_state_estimate_covariance_matrix - K*H_measurement_sensitivity_matrix*P_state_estimate_covariance_matrix;
    P_state_estimate_covariance_matrix      = 0.5*(P_state_estimate_covariance_matrix+P_state_estimate_covariance_matrix');
    t_vec_double_steps(steps_counter)  = t_vec(k);
    x_estimated_state1_over_time(steps_counter) = x_estimated_state_current(1);
    x_estimated_state2_over_time(steps_counter) = x_estimated_state_current(2);
    x_estimated_state3_over_time(steps_counter) = x_estimated_state_current(3);
    P_covariance_matrix_diagonal1_over_time(steps_counter) = P_state_estimate_covariance_matrix(1,1);
    P_covariance_matrix_diagonal2_over_time(steps_counter) = P_state_estimate_covariance_matrix(2,2);
    P_covariance_matrix_diagonal3_over_time(steps_counter) = P_state_estimate_covariance_matrix(3,3);

    % Temporal update:
    x_true_state_current      = Phi_state_transition_matrix_exact*x_true_state_current + delta; % update of true state w/o noise
    %
    % The EKF approximation for Phi is I3 + T*F,
    % where, for (d/dt) x(t) = a(x(t)), F is the
    % partial derivative F = (d/dx)a(x) evaluated
    % at the estimated value of x.
    Phi_state_transition_matrix_approximation   = I3 + T_sampling_interval*[0,1,0;
        -omega0_undamped_resonant_frequency^2,-2*omega0_undamped_resonant_frequency*x_estimated_state_current(3),-2*omega0_undamped_resonant_frequency*x_estimated_state_current(2);
        0,0,0];
    x_estimated_state_current    = Phi_state_transition_matrix_approximation*x_estimated_state_current + delta;
    P_state_estimate_covariance_matrix      = ...
        Phi_state_transition_matrix_approximation*P_state_estimate_covariance_matrix*Phi_state_transition_matrix_approximation'...
        + Qd;
end;


clf;
subplot(3,1,1),plot(t_vec,x_true_state1_over_time,'b-',t_vec_double_steps,x_estimated_state1_over_time,'g-');
%legend('True','Est.');
xlabel('Time (sec)');ylabel('Position (ft)');title('True and extimated states. (Press <ENTER> to continue.)');
subplot(3,1,2),plot(t_vec,x_true_state2_over_time,'b-',t_vec_double_steps,x_estimated_state2_over_time,'g-');
%legend('True','Est.');
xlabel('Time (sec)');ylabel('Velocity (fps)');
subplot(3,1,3),plot(t_vec,x_true_state3_over_time,'b-',t_vec_double_steps,x_estimated_state3_over_time,'g-');
%legend('True','Est.');
xlabel('Time (sec)');ylabel('Damp. Factor');
disp('Press <ENTER> to continue.');
pause
subplot(3,1,1),plot(t_vec_double_steps,P_covariance_matrix_diagonal1_over_time);xlabel('Time (sec)');ylabel('Position');title('Mean squared estimation uncertainties');
subplot(3,1,2),plot(t_vec_double_steps,P_covariance_matrix_diagonal2_over_time);xlabel('Time (sec)');ylabel('Velocity');
subplot(3,1,3),plot(t_vec_double_steps,P_covariance_matrix_diagonal3_over_time);xlabel('Time (sec)');ylabel('Damp. Factor');
disp('DONE');