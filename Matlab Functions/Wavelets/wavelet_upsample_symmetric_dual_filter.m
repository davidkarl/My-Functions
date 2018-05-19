function x = wavelet_upsample_symmetric_dual_filter(beta_coarse_samples,alpha_fine_samples,QMF,dual_QMF)
% UpDyad_SBS -- Symmetric Upsampling operator
%  Usage
%    x = UpDyad_SBS(beta,alpha,qmf,dqmf)
%  Inputs
%    beta  coarse coefficients
%    alpha fine coefficients
%    qmf   quadrature mirror filter
%    dqmf  dual quadrature mirror filter
%  Outputs
%    x     1-d signal at fine scale
%  See Also
%    DownDyad_SBS, IWT_SBS
%

% oddf = (rem(length(qmf),2)==1);
flag_filter_odd = ~(QMF(1)==0 & QMF(length(QMF))~=0);
%if vec is odd does that nescecerily say that the number of fine and coarse
%coefficients aren't equal in size????
flag_is_vec_odd = ( length(beta_coarse_samples) ~= length(alpha_fine_samples) );


L_total_size = length(beta_coarse_samples) + length(alpha_fine_samples);

if flag_filter_odd,
    if flag_is_vec_odd,
        extended_beta_coarse_samples = extend_signal_according_to_2_parameters(beta_coarse_samples,1,1);
        extended_alpha_fine_samples = extend_signal_according_to_2_parameters(alpha_fine_samples,2,2);
    else
        extended_beta_coarse_samples = extend_signal_according_to_2_parameters(beta_coarse_samples,2,1);
        extended_alpha_fine_samples = extend_signal_according_to_2_parameters(alpha_fine_samples,1,2);
    end
else
    if flag_is_vec_odd,
        extended_beta_coarse_samples = extend_signal_according_to_2_parameters(beta_coarse_samples,1,2);
        extended_alpha_fine_samples = [alpha_fine_samples , 0 , -reverse_vec(alpha_fine_samples)];
    else
        extended_beta_coarse_samples = extend_signal_according_to_2_parameters(beta_coarse_samples,2,2);
        extended_alpha_fine_samples = [alpha_fine_samples , -reverse_vec(alpha_fine_samples)];
    end
end

%Upsample coarse (low passed) samples:
coarse = wavelet_upsample_low_pass_periodized_symmetric_filter(extended_beta_coarse_samples,dual_QMF);

%Upsample fine (detailed, high passed) samples:
fine = wavelet_upsample_high_pass_periodized_symmetric_filter(extended_alpha_fine_samples,QMF);

%Combine coarse and fine scale samples to reconstruct upsampled signal:
x = coarse + fine;
x = x(1:L_total_size);




