function [beta_coarse_samples, alpha_fine_samples] = wavelet_downsample_symmetric_dual_filter(vec_in,QMF,dual_QMF)
% DownDyad_SBS -- Symmetric Downsampling operator
%  Usage
%    [beta,alpha] = DownDyad_SBS(x,qmf,dqmf)
%  Inputs
%    x     1-d signal at fine scale
%    qmf   quadrature mirror filter
%    dqmf  dual quadrature mirror filter
%  Outputs
%    beta  coarse coefficients
%    alpha fine coefficients
%  See Also
%    FWT_SBS
%

% oddf = (rem(length(qmf),2)==1);
flag_filter_odd = ~(QMF(1)==0 & QMF(length(QMF))~=0);
flag_is_vec_in_odd = (rem(length(vec_in),2)==1);

%symmetric extension of x - determine how to extend according to input filter:
if flag_filter_odd == 1
    flag_dont_repeat_last_element = 1; 
    flag_reach_only_up_to_second_element = 1;
else
    flag_dont_repeat_last_element = 2;
    flag_reach_only_up_to_second_element = 2;
end

%Actually extend signal:
extended_signal = extend_signal_according_to_2_parameters(vec_in,flag_dont_repeat_last_element,flag_reach_only_up_to_second_element);

%Convolution and downsampling:
downsampled_coarse_samples = wavelet_downsample_low_pass_periodized_symmetric_filter(extended_signal,QMF);
downsampled_fine_samples = wavelet_downsample_high_pass_periodized_symmetric_filter(extended_signal,dual_QMF);

%Project:
if flag_is_vec_in_odd
    beta_coarse_samples = downsampled_coarse_samples(1:(length(vec_in)+1)/2);
    alpha_fine_samples = downsampled_fine_samples(1:(length(vec_in)-1)/2);
else
    beta_coarse_samples = downsampled_coarse_samples(1:length(vec_in)/2);
    alpha_fine_samples = downsampled_fine_samples(1:length(vec_in)/2);
end




