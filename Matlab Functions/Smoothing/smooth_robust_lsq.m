% function [lsq_solution_signal_current] = smooth_robust_lsq(input_signal,smoothing_tau,differentiation_order)

input_signal = phase_signal_after_mask;
smoothing_tau = 10;
differentiation_order = 1;

%assign parameters:
[m,n] = size(input_signal);
ordstart = differentiation_order;

maximum_number_of_reweighting_iterations=25;
normalized_residual_std_change_tolerance=1e-6;
gaussian_error_breakpoint_in_terms_of_sigma = 1.5; 
iterative_reweighting_loop_gain = 1.3;

%Derived parameter:
robust_gaussian_error_e2_to_median2_factor = (gaussian_error_breakpoint_in_terms_of_sigma/0.6745)^2; 



% ------------------------------- Algorithm -------------------------------

%Tau normalizers for orders 1:5
smoothing_tau_normalizers=[4.000 3.416 3.404 3.411 3.417];

%disable warning message presentation:
s0=warning('off','MATLAB:rankDeficientMatrix');     

% If necessary, repeat calculations until conditioning is good
flag_is_equation_ill_conditioned_or_error_grown=true;
current_differentiation_order = differentiation_order;
while flag_is_equation_ill_conditioned_or_error_grown && current_differentiation_order>=1
    
    %clear warning indicator:
    lastwarn('');
    
    %Compute high order differentiation kernel and auxiliary mat (can't give order=0 because ...
    %the smoothing comes from the derivative constraints):
    high_order_diff_kernel = [-1 1]; 
    for i=1:current_differentiation_order-1
        high_order_diff_kernel = conv(high_order_diff_kernel,[-1 1]);
    end    
    high_order_diff_kernel_auxilliary_mat = repmat(high_order_diff_kernel,m-current_differentiation_order,1);
    
    %Initialize raw signal and signal differentiation weights:
    signal_weights_current = ones(m,1);
    signal_differentiation_weights = (smoothing_tau/smoothing_tau_normalizers(current_differentiation_order))^current_differentiation_order;
    
    %Construct Initial Least Squares Solution:
    %get main zero order part of the design matrix (unusual for least squares, ...
    %it doesn't fit coefficients but is tautological, very clever):
    zero_order_part_of_design_matrix = speye(m);
    %get signal differentiation part of the design matrix:
    higher_order_part_of_design_matrix = signal_differentiation_weights*spdiags(high_order_diff_kernel_auxilliary_mat,0:current_differentiation_order,m-current_differentiation_order,m);
    %get final design matrix:
    final_design_matrix_M = [zero_order_part_of_design_matrix; higher_order_part_of_design_matrix];
    %get target vector (first part is original vec and second part is added constraint of zeroing high order derivative:
    v = [input_signal; zeros(m-current_differentiation_order,1)];
    %get initial least squares solution:
    lsq_solution_signal_initial = final_design_matrix_M\v;
    
    
    %get last warninig ID and check if a warning about rank deficiency is returned:
    [lastmsg,lastID] = lastwarn;
    flag_is_equation_ill_conditioned_or_error_grown = strcmp(lastID,'MATLAB:rankDeficientMatrix');
    
    %get coefficient of determination (error std divided by total std) / normalized residual error:
    normalized_residual_std_initial = std(lsq_solution_signal_initial-input_signal)/std(lsq_solution_signal_initial);
    
    %Initialize iterative lsq loop counter and loop over signal reweighting it iteratively:
    iterative_lsq_loop_counter = 0;
    lsq_solution_signal_current = lsq_solution_signal_initial;
    normalized_residual_std_change_current = normalized_residual_std_initial;
    while ~flag_is_equation_ill_conditioned_or_error_grown ...
            && normalized_residual_std_change_current > normalized_residual_std_change_tolerance ...
            && iterative_lsq_loop_counter < maximum_number_of_reweighting_iterations
        
        %keep track of previous solution, normalized residual std, and absolute squared error signal:
        lsq_solution_signal_previous = lsq_solution_signal_current;
        normalized_residual_std_change_previous = normalized_residual_std_change_current;
        absolute_error_signal_previous = abs(input_signal-lsq_solution_signal_current);
        
        %calculate more robust estimate of error std using median to keep out influence of outliers:
        robust_gaussian_assumed_error_e2_point_estimation = robust_gaussian_error_e2_to_median2_factor * median(absolute_error_signal_previous.^2); 
        
        %get new signal weights before loop gains:
        signal_weights_new = sqrt(robust_gaussian_assumed_error_e2_point_estimation ./ (robust_gaussian_assumed_error_e2_point_estimation+absolute_error_signal_previous.^2) ); 
        
        %get current signal weights after loop gains:
        signal_weights_current = signal_weights_current.^(1-iterative_reweighting_loop_gain) .* signal_weights_new.^iterative_reweighting_loop_gain;
        
        %preserve ratio of mean(signal_weights) / mean(signal_differentiation_weights):
        signal_weights_current = signal_weights_current/mean(signal_weights_current); 
        
        %get final, reweighted, design matrix:
        final_design_matrix_M = [spdiags(signal_weights_current,0,m,m); higher_order_part_of_design_matrix];
        %get final, reweighted, target vector:
        v = [signal_weights_current.*input_signal; zeros(m-current_differentiation_order,1)]; 
        %get current lsq solution signal:
        lsq_solution_signal_current = final_design_matrix_M\v;
        
        %get last warninig ID and check if a warning about rank deficiency is returned:
        [lastmsg,lastID] = lastwarn;
        normalized_residual_std_change_current = std(lsq_solution_signal_current-lsq_solution_signal_previous)/std(lsq_solution_signal_current); 
        flag_has_normalized_residual_grown = normalized_residual_std_change_current >= normalized_residual_std_change_previous;
        flag_is_equation_ill_conditioned_or_error_grown = strcmp(lastID,'MATLAB:rankDeficientMatrix') || flag_has_normalized_residual_grown;
        
        %update loop counter:
        iterative_lsq_loop_counter = iterative_lsq_loop_counter+1;
    end
    %count
    
    %downgrade differentiation order for next potential loop:
    current_differentiation_order = current_differentiation_order - 1;
    
end
%restore warning state:
warning(s0);                          

% -------------------------------- Notices --------------------------------

if current_differentiation_order < ordstart                     % if order was reduced
    disp(['Notice: IRLSSMOOTH order ORD was reduced from ' ...
          int2str(ordstart) ' to ' int2str(current_differentiation_order) ...
          ' for conditioning.'])
    disp('  ')
end
if iterative_lsq_loop_counter == maximum_number_of_reweighting_iterations                  % if max count was reached
    warning(['Maximum number of iterations reached.  ' ...
             'Per-iteration relative output change = ' num2str(normalized_residual_std_change_current)])
end










