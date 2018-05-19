function mat_in_transform = wavelet_2D_transform_translation_invariant(mat_in,L_coarsest_scale_level,QMF)
% FWT_TI -- 2-D translation invariant forward wavelet transform
%  Usage
%    TIWT = FWT2_TI(x,L,qmf)
%  Inputs
%    x        2-d image (n by n real array, n dyadic)
%    L        degree of coarsest scale
%    qmf      orthonormal quadrature mirror filter
%  Outputs
%    TIWT     translation-invariant wavelet transform table, (3*(J-L)+1)*n by n
%
%  See Also
%    IWT2_TI, IWT2_TIMedian
%

[signal_length,J_dyadic_length] = quadlength(mat_in);
D_number_of_levels_to_compute = J_dyadic_length-L_coarsest_scale_level;

mat_in_transform = zeros( (3*D_number_of_levels_to_compute+1)*signal_length , signal_length);
lastx = (3*D_number_of_levels_to_compute*signal_length+1) : (3*D_number_of_levels_to_compute*signal_length+signal_length); 
lasty = 1 : signal_length;
mat_in_transform(lastx,lasty) = mat_in;
%

%Loop over the different scales:
for d_current_depth = 0 : (D_number_of_levels_to_compute-1),
    
    %Get current level parameters:
    l_current_coarsest_level = J_dyadic_length-d_current_depth-1; 
    current_number_of_samples = 2^(J_dyadic_length-d_current_depth);
    
    
    for b_block_index1 = 0:(2^d_current_depth-1)
        for b_block_index2 = 0:(2^d_current_depth-1)
            
            current_samples = mat_in_transform(3*D_number_of_levels_to_compute*signal_length ...
                                 + wavelet_packet_table_indexing(d_current_depth,b_block_index1,signal_length), ...
                                     wavelet_packet_table_indexing(d_current_depth,b_block_index2,signal_length) );
            
            %Get wavelet transform of all needed samples (with a shift in each dimension):
            wc00 = wavelet_2D_transform_orthogonal(current_samples,l_current_coarsest_level,QMF);
            wc01 = wavelet_2D_transform_orthogonal(circular_shift_2D(current_samples,0,1),l_current_coarsest_level,QMF);
            wc10 = wavelet_2D_transform_orthogonal(circular_shift_2D(current_samples,1,0),l_current_coarsest_level,QMF);
            wc11 = wavelet_2D_transform_orthogonal(circular_shift_2D(current_samples,1,1),l_current_coarsest_level,QMF);
            
            %Get appropriate indices:
            index10 = wavelet_packet_table_indexing(d_current_depth+1, 2*b_block_index1, signal_length); 
            index20 = wavelet_packet_table_indexing(d_current_depth+1, 2*b_block_index2, signal_length);
            index11 = wavelet_packet_table_indexing(d_current_depth+1, 2*b_block_index1+1, signal_length); 
            index21 = wavelet_packet_table_indexing(d_current_depth+1, 2*b_block_index2+1, signal_length);
            
            %horizontal stuff:
            mat_in_transform(3*d_current_depth*signal_length + index10 , index20) = ...
                wc00(1:(current_number_of_samples/2),(current_number_of_samples/2+1):current_number_of_samples);
            
            mat_in_transform(3*d_current_depth*signal_length + index11,  index20) = ...
                wc01(1:(current_number_of_samples/2),(current_number_of_samples/2+1):current_number_of_samples);
            
            mat_in_transform(3*d_current_depth*signal_length + index10 , index21) = ...
                wc10(1:(current_number_of_samples/2),(current_number_of_samples/2+1):current_number_of_samples);
            
            mat_in_transform(3*d_current_depth*signal_length + index11 , index21) = ...
                wc11(1:(current_number_of_samples/2),(current_number_of_samples/2+1):current_number_of_samples);
            
            %vertical stuff:
            mat_in_transform((3*d_current_depth+1)*signal_length + index10 , index20) = ...
                wc00((current_number_of_samples/2+1):current_number_of_samples,1:(current_number_of_samples/2));
            
            mat_in_transform((3*d_current_depth+1)*signal_length + index11,  index20) = ...
                wc01((current_number_of_samples/2+1):current_number_of_samples,1:(current_number_of_samples/2));
            
            mat_in_transform((3*d_current_depth+1)*signal_length + index10 , index21) = ...
                wc10((current_number_of_samples/2+1):current_number_of_samples,1:(current_number_of_samples/2));
            
            mat_in_transform((3*d_current_depth+1)*signal_length + index11 , index21) = ...
                wc11((current_number_of_samples/2+1):current_number_of_samples,1:(current_number_of_samples/2));
           
            %diagonal stuff:
            mat_in_transform((3*d_current_depth+2)*signal_length + index10 , index20) = ...
                wc00((current_number_of_samples/2+1):current_number_of_samples, (current_number_of_samples/2+1):current_number_of_samples);
           
            mat_in_transform((3*d_current_depth+2)*signal_length + index11,  index20) = ...
                wc01((current_number_of_samples/2+1):current_number_of_samples, (current_number_of_samples/2+1):current_number_of_samples);
            
            mat_in_transform((3*d_current_depth+2)*signal_length + index10 , index21) = ...
                wc10((current_number_of_samples/2+1):current_number_of_samples, (current_number_of_samples/2+1):current_number_of_samples);
            
            mat_in_transform((3*d_current_depth+2)*signal_length + index11 , index21) = ...
                wc11((current_number_of_samples/2+1):current_number_of_samples, (current_number_of_samples/2+1):current_number_of_samples);
            
            %low freq stuff:
            mat_in_transform(3*D_number_of_levels_to_compute*signal_length + index10 , index20) = ...
                wc00(1:(current_number_of_samples/2), 1:(current_number_of_samples/2));
           
            mat_in_transform(3*D_number_of_levels_to_compute*signal_length + index11,  index20) = ...
                wc01(1:(current_number_of_samples/2), 1:(current_number_of_samples/2));
            
            mat_in_transform(3*D_number_of_levels_to_compute*signal_length + index10 , index21) = ...
                wc10(1:(current_number_of_samples/2), 1:(current_number_of_samples/2));
           
            mat_in_transform(3*D_number_of_levels_to_compute*signal_length + index11 , index21) = ...
                wc11(1:(current_number_of_samples/2), 1:(current_number_of_samples/2));
        end
    end
end




