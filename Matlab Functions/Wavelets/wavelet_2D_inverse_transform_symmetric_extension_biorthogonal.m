function reconstructed_mat = wavelet_2D_inverse_transform_symmetric_extension_biorthogonal(wavelet_coefficients,L_coarsest_scale,QMF,dual_QMF)
% IWT2_SBS -- Inverse 2d Wavelet Transform
%            (symmetric extention, bi-orthogonal)
%  Usage
%    x = IWT2_SBS(wc,L,qmf,dqmf)
%  Inputs
%      wc    2-d wavelet transform [n by n array, n arbitrary]
%      L     coarse level
%      qmf   low-pass quadrature mirror filter
%      dqmf  high-pas dual quadrature mirror filter
%  Outputs
%      x     2-d signal reconstructed from wc
%  Description
%      If wc is the result of a forward 2d wavelet transform, with
%           wc = FWT2_SBS(x,L,qmf,dqmf)
%      then x = IWT2_SBS(wc,L,qmf,dqmf) reconstructs x exactly if qmf is a nice
%      quadrature mirror filter, e.g. one made by MakeBioFilter
%  See Also:
%    FWT2_SBS, MakeBioFilter
%

[m_row_size,J_dyadic_length_rows] = dyadlength(wavelet_coefficients(:,1));
[n_column_size,K_dyadic_length_columns] = dyadlength(wavelet_coefficients(1,:));
% assume m==n, J==K

reconstructed_mat = wavelet_coefficients;

dyadic_partition_rows = get_dyadic_partition_of_nondyadic_signals(m_row_size);

for j_current_scale = L_coarsest_scale : J_dyadic_length_rows-1,
    bottom_indices = 1:dyadic_partition_rows(j_current_scale+1);
    top_indices = (dyadic_partition_rows(j_current_scale+1)+1):dyadic_partition_rows(j_current_scale+2);
    all_indices = [bottom_indices , top_indices];

    current_scale_number_of_samples = length(all_indices);

    for column_counter = 1:current_scale_number_of_samples,
        reconstructed_mat(all_indices,column_counter) =  ...
            wavelet_upsample_symmetric_dual_filter(reconstructed_mat(bottom_indices,column_counter)', ...
                                                   reconstructed_mat(top_indices,column_counter)', QMF, dual_QMF)';
    end
    for rows_counter = 1:current_scale_number_of_samples,
        reconstructed_mat(rows_counter,all_indices) = ...
            wavelet_upsample_symmetric_dual_filter(reconstructed_mat(rows_counter,bottom_indices), ...
                                                   reconstructed_mat(rows_counter,top_indices), QMF, dual_QMF);
    end
end