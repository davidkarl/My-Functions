function [lsq_solution_signal_final] = smooth_lsq_minimum_iterations_with_preweights2(input_signal,raw_signal_weights,derivative_signal_weight_factor,differentiation_order)
%this function can be very sensitive to parameter choice and it can be
%played with endlessely such as adding different constraints on different
%derivative orders etc'.

% %Input Parameter Example:
% input_signal = phase_signal_difference; %raw FM signal
% binary_mask = ones(size(input_signal));
% binary_mask(indices_containing_spikes)=0; %give zero weight where an earilier function found a spike
% raw_signal_weights = abs(analytic_signal).*binary_mask;
% derivative_signal_weight_factor = 4;
% differentiation_order = 2;

%assign parameters:
[input_signal_length,n] = size(input_signal);
ordstart = differentiation_order;



% ------------------------------- Algorithm -------------------------------

%Tau normalizers for orders 1:5
derivative_signal_weight_coefficient_normalizers=[4.000 3.416 3.404 3.411 3.417];

%disable warning message presentation:
s0=warning('off','MATLAB:rankDeficientMatrix');     

%Start loop which should go only 1 time as under my parameters equations
%are well conditioned but in any case the loop tries to calculate LSQ
%solution and if it can't it lowers the maximum differentiation order which
%is tries to equate to zero:
flag_lsq_equation_is_ill_conditioned=true;
current_differentiation_order = differentiation_order;
differentiated_signal = diff(input_signal);
while flag_lsq_equation_is_ill_conditioned && current_differentiation_order>=1
    
    %clear warning indicator:
    lastwarn('');
    
    %Compute high order differentiation kernel and auxiliary mat (can't give order=0 because ...
    %the smoothing comes from the derivative constraints and i don't want to choke signal change totally):
    high_order_diff_kernel = [-1 1]; 
    for i=1:current_differentiation_order-1
        high_order_diff_kernel = conv(high_order_diff_kernel,[-1 1]);
        differentiated_signal = diff(differentiated_signal);
    end    
    high_order_diff_kernel_auxilliary_mat = repmat(high_order_diff_kernel,input_signal_length-current_differentiation_order,1);
    
    %Initialize signal differentiation weights (the larger it is the more smoothing):
    normalized_derivative_signal_weight_factor = (derivative_signal_weight_factor/derivative_signal_weight_coefficient_normalizers(current_differentiation_order))^current_differentiation_order;
    
    %Construct Initial Least Squares Solution:
    %get zero order part of the design matrix (with raw signal weights):
    zero_order_part_of_design_matrix = spdiags(raw_signal_weights,0,input_signal_length,input_signal_length);
    %get higher order differentiation part of the design matrix:
    buffered_signal_weights = buffer(raw_signal_weights,length(high_order_diff_kernel),length(high_order_diff_kernel)-1,'nodelay');
    derivative_signal_weights = min(buffered_signal_weights,[],1)';
    derivative_signal_weights = derivative_signal_weights(1:input_signal_length-current_differentiation_order);

%     high_order_diff_kernel_auxilliary_mat = bsxfun(@times,high_order_diff_kernel_auxilliary_mat,derivative_signal_weights);
    higher_order_part_of_design_matrix = normalized_derivative_signal_weight_factor*spdiags(high_order_diff_kernel_auxilliary_mat,0:current_differentiation_order,input_signal_length-current_differentiation_order,input_signal_length);
    %get final design matrix:
    final_design_matrix_M = [zero_order_part_of_design_matrix; higher_order_part_of_design_matrix];
    %get target vector (first part is original vec and second part is added constraint of zeroing high order derivative:
%     v = [raw_signal_weights.*input_signal; zeros(input_signal_length-current_differentiation_order,1)];
    v = [raw_signal_weights.*input_signal; normalized_derivative_signal_weight_factor*derivative_signal_weights.^1 .*differentiated_signal];
    %get initial least squares solution:
    lsq_solution_signal_final = final_design_matrix_M\v;
    
    %get last warninig ID and check if a warning about rank deficiency is returned:
    [lastmsg,lastID] = lastwarn;
    flag_lsq_equation_is_ill_conditioned = strcmp(lastID,'MATLAB:rankDeficientMatrix');
    if flag_lsq_equation_is_ill_conditioned==1
       differentiated_signal = diff(input_signal); 
    end
        
    %downgrade differentiation order for next potential loop:
    current_differentiation_order = current_differentiation_order - 1;
end
%restore warning state:
warning(s0);                          

% -------------------------------- Notices --------------------------------

if current_differentiation_order < ordstart-1                     % if order was reduced
    disp(['Notice: IRLSSMOOTH order ORD was reduced from ' ...
          int2str(ordstart) ' to ' int2str(current_differentiation_order) ...
          ' for conditioning.'])
    disp('  ')
end











