% M. S. Grewal and A. P. Andrews, 
% Kalman Filtering: Theory and Practice, 
% published by Wiley ,2000. 
% Demonstration of probability conditioning on 
% measurements. 
disp('M. S. Grewal and A. P. Andrews,'); 
disp('Kalman Filtering: Theory and Practice,'); 
disp('published by Wiley ,2000.'); 
disp(' '); 
disp('This demonstration should help you visualize'); 
disp('how estimation influences uncertainty.'); 
disp(' '); 
disp('It demonstrates the effect that the Kalman filter'); 
disp('has in terms of its conditioning of the probability'); 
disp('distribution of the state of a random walk.'); 

%Initialize Parameters:
P_estimate_covariance = 0.4; 
x_state = sqrt(P_estimate_covariance)*randn(1); 
F = 1; %random walk: x(k)=x(k-1)+w
xhat_state_estimate = 0; 
H_measurement_matrix = 1; 
Q_process_noise_covariance = 0.01;  
R_measurement_noise_covariance = 0.04; 
A = zeros(101,101);
nm = 0; 
mi = 21;

for k=1:101,
    %get actual state, state estimate, and upper and lower bounds:
    x_acutal_state_over_time(k) = x_state;
    x_state_estimate_over_time(k) = xhat_state_estimate;
    x_state_estimate_upper_bound_over_time(k) = xhat_state_estimate+sqrt(P_estimate_covariance);
    x_state_estimate_lower_bound_over_time(k) = xhat_state_estimate-sqrt(P_estimate_covariance);
    
    %get probability distribution assuming gaussian distribution, mean
    %being xhat_state_estimate and sigma^2=P_estimate_covariance:
    for m=1:101,
        deviation_from_mean_estimate  = (m-51)/25 - xhat_state_estimate;
        prob = exp(-deviation_from_mean_estimate^2/(2*P_estimate_covariance))/sqrt(2*pi*P_estimate_covariance);
        A(m,k) = prob;
    end;
    
    %Update P_estimate_covariance (we can see that if no update is done then estimate covariance, 
    %or error, keeps increasing because of the noise term in the state update equation:
    P_estimate_covariance = F*P_estimate_covariance*F + Q_process_noise_covariance;
    
    %Update x_hat estimate and P_estimate_covariance every mi steps:
    if  mod(k,mi)==0
        disp(['Measurement taken at t = ',num2str(k),' seconds.']);
        %get noisy observation using H_measurement matrix:
        z = H_measurement_matrix*x_state + randn(1)*sqrt(R_measurement_noise_covariance);
        %calculate the optimal kalman gain:
        K_gain = P_estimate_covariance*H_measurement_matrix/(H_measurement_matrix^2*P_estimate_covariance+R_measurement_noise_covariance);
        %update xhat_state estimate using the optimal kalman gain and measurement residual:
        z_tilde_measurement_residual = z - H_measurement_matrix*xhat_state_estimate;
        %update xhat_state_estimate:
        xhat_state_estimate = xhat_state_estimate + K_gain*z_tilde_measurement_residual;
        %update P_estimate_covariance:
        P_estimate_covariance = P_estimate_covariance - K_gain*H_measurement_matrix*P_estimate_covariance;
        nm = nm + 1;
        t0(nm) = k-1;
        z0(nm) = z;
    end
    
    %evolve state using state dynamic equation:
    x_state = F*x_state + randn(1)*sqrt(Q_process_noise_covariance);
    state_value_vec(k) = (k-51)/25;
    time_vec(k) = k-1;
end;

clf; 
surf(time_vec,state_value_vec,A); 
zlabel('Probability Density'); 
xlabel('Time in Seconds'); 
ylabel('State Value'); 
title('Evolution of probability distribution of the state of a random walk'); 
disp('Plot shows evolution of inferred probability distribution'); 
disp('of the state of a random walk.'); 
disp(' '); 
disp('Note the broadening of the probability distribution'); 
disp(['except when measurements are taken (every ',num2str(mi),' seconds).']); 
disp(' '); 



plot(time_vec,x_acutal_state_over_time,time_vec,x_state_estimate_over_time,time_vec,x_state_estimate_upper_bound_over_time,time_vec,x_state_estimate_lower_bound_over_time);
for n=1:nm,
    text(t0(n),z0(n),'z');
end;
xlabel('Time in Seconds'); 
ylabel('State Value'); 
title('True and estimated state of random walk.  "z" at measured values.'); 
disp('Plots show true and estimated (+/- 1 sigma) state values.'); 
disp('The true value should be within the +/- bounds'); 
disp(' more than half the time.  Is it?'); 
disp(' '); 
disp('Measured values (with measurement noise)'); 
disp(' are indicated by "z"'); 
disp(' '); 
disp('Why do the +/- 1-sigma bounds contract'); 
disp('whenever measurements are taken?'); 
disp('Why do they expand between measurements?'); 
disp(' '); 
disp('NOTE: Results will be different each time you run this.'); 
disp('      Try it and see.'); 





