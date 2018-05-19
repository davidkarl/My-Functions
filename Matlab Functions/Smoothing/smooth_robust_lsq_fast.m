% function [lsq_solution_signal_current] = smooth_robust_lsq_fast(input_signal,smoothing_tau,differentiation_order)

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
    tic
    high_order_diff_kernel = [-1 1]; 
    for i=1:current_differentiation_order-1
        high_order_diff_kernel = conv(high_order_diff_kernel,[-1 1]);
    end    
    high_order_diff_kernel_auxilliary_mat = repmat(high_order_diff_kernel,m-current_differentiation_order,1);
    disp('(1). repmat of high_order_diff_kernel:');
    toc
    
    %Initialize raw signal and signal differentiation weights:
    tic
    signal_weights_current = ones(m,1);
    signal_differentiation_weights = (smoothing_tau/smoothing_tau_normalizers(current_differentiation_order))^current_differentiation_order;
    disp('(2). signal_weights=ones(m,1) & get signal_differentiation_weight:');
    toc
    
    %Construct Initial Least Squares Solution:
    %get main zero order part of the design matrix (unusual for least squares, ...
    %it doesn't fit coefficients but is tautological, very clever):
    tic
    zero_order_part_of_design_matrix = speye(m);
    disp('(3). speye(m):');
    toc
    %get signal differentiation part of the design matrix:
    tic
    higher_order_part_of_design_matrix = signal_differentiation_weights*spdiags(high_order_diff_kernel_auxilliary_mat,0:current_differentiation_order,m-current_differentiation_order,m);
    disp('(4). high_order_design_matrix = spdiags(high_order_diff_kernel):');
    toc
    %get final design matrix:
    tic
    final_design_matrix_M = [zero_order_part_of_design_matrix; higher_order_part_of_design_matrix];
    disp('(5). final_desigm_matrix = [zero_order_part;higher_order_part] :');
    toc
    %get target vector (first part is original vec and second part is added constraint of zeroing high order derivative:
    tic
    v = [input_signal; zeros(m-current_differentiation_order,1)];
    disp('(6). v = [input_signal;zeros(..)] :');
    toc
    %get initial least squares solution:
    tic
    lsq_solution_signal_initial = final_design_matrix_M\v;
    disp('(7). lsq_solution = final_design_matrix\v :');
    toc
    
    %get last warninig ID and check if a warning about rank deficiency is returned:
    tic
    [lastmsg,lastID] = lastwarn;
    flag_is_equation_ill_conditioned_or_error_grown = strcmp(lastID,'MATLAB:rankDeficientMatrix');
    disp('(8). strcmp to check if rank deficient');
    toc
    
    %get coefficient of determination (error std divided by total std) / normalized residual error:
    tic
    normalized_residual_std_initial = std(lsq_solution_signal_initial-input_signal)/std(lsq_solution_signal_initial);
    disp('(9). normalized_std = std(new-old)/std(old) :');
    toc
    
    %Initialize iterative lsq loop counter and loop over signal reweighting it iteratively:
    iterative_lsq_loop_counter = 0;
    lsq_solution_signal_current = lsq_solution_signal_initial;
    normalized_residual_std_change_current = normalized_residual_std_initial;
    while ~flag_is_equation_ill_conditioned_or_error_grown ...
            && normalized_residual_std_change_current > normalized_residual_std_change_tolerance ...
            && iterative_lsq_loop_counter < maximum_number_of_reweighting_iterations
        
        %keep track of previous solution, normalized residual std, and absolute squared error signal:
        tic
        lsq_solution_signal_previous = lsq_solution_signal_current;
        normalized_residual_std_change_previous = normalized_residual_std_change_current;
        absolute_error_signal_previous = abs(lsq_solution_signal_current-input_signal);
        disp('(10). absolution_error_signal = abs(new-input_signal) :');
        toc
        
        %calculate more robust estimate of error std using median to keep out influence of outliers:
        tic
        robust_gaussian_assumed_error_e2_point_estimation = robust_gaussian_error_e2_to_median2_factor * median(absolute_error_signal_previous.^2); 
        disp('(11). robust gaussian std = factor*median() :');
        toc
        
        %get new signal weights before loop gains:
        tic
        signal_weights_new = sqrt(robust_gaussian_assumed_error_e2_point_estimation ./ (robust_gaussian_assumed_error_e2_point_estimation+absolute_error_signal_previous.^2) ); 
        disp('(12). new_weights = sqrt(bla/(bla+blabla)) :');
        toc
        
        %get current signal weights after loop gains:
        tic
        signal_weights_current = signal_weights_current.^(1-iterative_reweighting_loop_gain) .* signal_weights_new.^iterative_reweighting_loop_gain;
        disp('(13). current signal weights = bla.^(1-g).*blabla.^(g) :');
        toc
        
        %preserve ratio of mean(signal_weights) / mean(signal_differentiation_weights):
        tic
        signal_weights_current = signal_weights_current/mean(signal_weights_current); 
        disp('(14). weights = weights/mean(weights)');
        toc
        
        %get final, reweighted, design matrix:
        tic
        final_design_matrix_M = [spdiags(signal_weights_current,0,m,m); higher_order_part_of_design_matrix];
        disp('(15). final_design_matrix = [spdiags(weights);high_order_design_matrix] :');
        toc
        %get final, reweighted, target vector:
        tic
        v = [signal_weights_current.*input_signal; zeros(m-current_differentiation_order,1)]; 
        disp('(16). v = [weights.*input_signal;zeros(...)] :');
        toc
        %get current lsq solution signal:
        tic
        lsq_solution_signal_current = final_design_matrix_M\v;
        disp('(17). lsq_solution = design_matrix\v :');
        toc
        
        %get last warninig ID and check if a warning about rank deficiency is returned:
        [lastmsg,lastID] = lastwarn;
        tic
        normalized_residual_std_change_current = std(lsq_solution_signal_current-lsq_solution_signal_previous)/std(lsq_solution_signal_current); 
        disp('(18). normalized_std = std(bla-bli)/std(bli) :');
        toc
        tic
        flag_has_normalized_residual_grown = normalized_residual_std_change_current >= normalized_residual_std_change_previous;
        flag_is_equation_ill_conditioned_or_error_grown = strcmp(lastID,'MATLAB:rankDeficientMatrix') || flag_has_normalized_residual_grown;
        disp('(19). strcmp to check if rank deficiency :');
        toc
        
        %update loop counter:
        iterative_lsq_loop_counter = iterative_lsq_loop_counter+1;
        disp('*************************************');
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










