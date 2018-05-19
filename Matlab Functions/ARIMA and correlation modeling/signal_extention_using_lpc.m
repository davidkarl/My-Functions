% function [output_signal_post , a_lpc_coefficients] = ...
%     signal_extention_using_lpc(input_signal, AR_model_order, number_of_samples_post,number_of_samples_pre, pos_string)
% LPREDICT estimates the values of a data set before/after the observed set.
%
% LPREDICT uses linear prediction to extrapolate data, typically a
% timeseries. Note that this is not the same as linear extrapolation. 
% A window of autocorrelation coefficients is moved beyond the data
% limits to extrapolate the data. For a discussion, see Press et. al. [1].
%
% The required coefficients are derived from a call to LPC in MATLAB's
% Signal Processing Toolbox
%
% Example:
% y=LPREDICT(x, np, npred, pos)
% [y, a]=LPREDICT(x, np, npred, pos)
%      x:       the input data series as a column vector or a matrix 
%                   with series organized in columns
%      np:      the number of predictor coefficients to use (>=2)
%      npred:   the number of data values to return in the output
%      pos:     a string 'pre' or 'post' (default: post)
%                   This determines whether extrapolation occurs before or
%                   after the observed series x.
%
%      y:       the output, appropriately sequenced for concatenation with
%                   input x
%      a:       the coefficients returned by LPC (organized in rows).
%                   These can be used to check the quality/stability of the
%                   fit to the observed data as described in the
%                   documenation to the LPC function.
%
% The output y is given by:
%       y(k) = -a(2)*y(k-1) - a(3)*y(k-2) - ... - a(np)*y(k-np)
%                where y(n) => x(end-n) for n<=0
% 
% Note that sum(-a(2:end))is always less than unity. The output will
% therefore approach zero as npred increases. This may be a problem if x
% has a large DC offset. Subtract the the column mean(s) of x from x on
% input and add them to the output column(s) to restore DC. For a more
% accurate DC correction, see [1].
%
% To pre-pend data, the input sequence is reversed and the output is
% similarly reversed before being returned. The output may always be
% vertically concatenated with the input to extend the timeseries e.g:
%       k=(1:100)';
%       x=exp(-k/100).*sin(k/5);
%       x=[lpredict(x, 5, 100, 'pre'); x; lpredict(x, 5, 100, 'post')];
% 
% 
% See also LPC
%
% References:
% [1] Press et al. (1992) Numerical Recipes in C. (Ed. 2, Section 13.6).
%
% Toolboxes Required: Signal Processing
%
% Revisions:    10.07 renamed to avoid filename clash with System ID
%                     Toolbox
%                     DC correction help text corrected.
%
% -------------------------------------------------------------------------
% Author: Malcolm Lidierth 10/07
% Copyright © The Author & King's College London 2007
% -------------------------------------------------------------------------
  

N = 500;
AR_model_order = 5; 
poles = 0.6*rand(AR_model_order,1);
poles = min(abs(poles),0.5).*sign(poles);
AR_parameters = poly(poles);
% AR_parameters = [1;-0.44*randn(AR_model_order,1)]; 
% AR_parameters = [1,-1.5,0.8]';
number_of_samples_post = 20;
number_of_samples_pre = 20;
input_signal = filter(1,AR_parameters,randn(N,1)); 


%Get the forward linear predictor coefficients via the LPC function:
a_lpc_coefficients = lpc(input_signal,AR_model_order);

%Negate coefficients, and get rid of a(1)
cc = -a_lpc_coefficients(2:end);

%Pre-allocate output: 
output_signal_post = zeros(number_of_samples_post,1); 
output_signal_pre = zeros(number_of_samples_pre,1); 
 
%EXTRAPOLATE FORWARD:
%Seed y with the first value
output_signal_post(1) = cc*input_signal(end:-1:end-AR_model_order+1);
% Next np-1 values
for k=2:min(AR_model_order,number_of_samples_post)
    output_signal_post(k)=cc*[output_signal_post(k-1:-1:1); input_signal(end:-1:end-AR_model_order+k)];
end
% Now do the rest
for k=AR_model_order+1:number_of_samples_post
    output_signal_post(k)=cc*output_signal_post(k-1:-1:k-AR_model_order);
end
%EXTRAPOLATE BACKWARD: 
reversed_input_signal = flip(input_signal);
%Seed y with the first value:
output_signal_pre(1) = cc*input_signal(end:-1:end-AR_model_order+1);
% Next np-1 values
for k=2:min(AR_model_order,number_of_samples_pre)
    output_signal_pre(k)=cc*[output_signal_pre(k-1:-1:1); input_signal(end:-1:end-AR_model_order+k)];
end
% Now do the rest
for k=AR_model_order+1:number_of_samples_pre
    output_signal_pre(k)=cc*output_signal_pre(k-1:-1:k-AR_model_order); 
end
%Construct total signal:
total_output_signal = [flip(output_signal_pre(:)) ; input_signal(:) ; output_signal_post(:)];

figure;
scatter(1:length(input_signal),input_signal);
hold on; 
plot(-number_of_samples_pre+1:length(input_signal)+number_of_samples_post, total_output_signal,'r');

[extended_signal] = extend_signal_using_edft(input_signal',280,10,1);
plot(extended_signal);
 
% Order the output sequence if required
if nargin==4 && strcmpi(pos_string,'pre') 
    output_signal_post=output_signal_post(end:-1:1);
end

% return 
% end
