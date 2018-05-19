function reconstructed_mat = wavelet_2D_inverse_transform_translation_invariant(wavelet_transform,L_coarsest_level,QMF)
% IWT2_TI -- Invert 2-d translation invariant wavelet transform
%  Usage
%    x = IWT2_TI(TIWT,qmf)
%  Inputs
%    TIWT     translation-invariant wavelet transform table, (3*(J-L)+1)*n by n
%    L        degree of coarsest scale
%    qmf      quadrature mirror filter
%  Outputs
%    x        2-d image reconstructed from translation-invariant transform TIWT
%
%  See Also
%    FWT2_TI, IWT2_TIMedian
%

%Get wavelet transform parameters:
[D1,signal_length] = size(wavelet_transform);
J_dyadic_length = log2(signal_length);
D_number_of_levels = J_dyadic_length-L_coarsest_level;
%

%Get appropriate wavelet transform samples:
lastx = (3*D_number_of_levels*signal_length+1):(3*D_number_of_levels*signal_length+signal_length); 
lasty = 1:signal_length;
reconstructed_mat = wavelet_transform(lastx,lasty);

for d = (D_number_of_levels-1):-1:0,
    
    l_current_coarsest_level = J_dyadic_length-d-1; 
    current_number_of_samples = 2^(J_dyadic_length-d);
    
    for b1=0:(2^d-1), 
        for b2=0:(2^d-1),
            
            index10 = wavelet_packet_table_indexing(d+1,2*b1,signal_length); 
            index20 = wavelet_packet_table_indexing(d+1,2*b2,signal_length);
            index11 = wavelet_packet_table_indexing(d+1,2*b1+1,signal_length); 
            index21 = wavelet_packet_table_indexing(d+1,2*b2+1,signal_length);

            wc00 = [reconstructed_mat(index10,index20) , wavelet_transform(3*d*signal_length+index10,index20) ; ...
                wavelet_transform((3*d+1)*signal_length+index10,index20) , wavelet_transform((3*d+2)*signal_length+index10,index20)];
            
            wc01 = [reconstructed_mat(index11,index20) , wavelet_transform(3*d*signal_length+index11,index20) ; ...
                wavelet_transform((3*d+1)*signal_length+index11,index20) , wavelet_transform((3*d+2)*signal_length+index11,index20)];
            
            wc10 = [reconstructed_mat(index10,index21) , wavelet_transform(3*d*signal_length+index10,index21) ; ...
                wavelet_transform((3*d+1)*signal_length+index10,index21) , wavelet_transform((3*d+2)*signal_length+index10,index21)];
            
            wc11 = [reconstructed_mat(index11,index21) , wavelet_transform(3*d*signal_length+index11,index21) ; ...
                wavelet_transform((3*d+1)*signal_length+index11,index21) , wavelet_transform((3*d+2)*signal_length+index11,index21)];

            reconstructed_mat(packet(d,b1,signal_length), wavelet_packet_table_indexing(d,b2,signal_length)) = ...
               1/4 * (      wavelet_2D_inverse_transform_orthogonal(wc00,l_current_coarsest_level,QMF) + ....
          circular_shift_2D(wavelet_2D_inverse_transform_orthogonal(wc01,l_current_coarsest_level,QMF),0,-1) + ...
          circular_shift_2D(wavelet_2D_inverse_transform_orthogonal(wc10,l_current_coarsest_level,QMF),-1,0) + ...
          circular_shift_2D(wavelet_2D_inverse_transform_orthogonal(wc11,l_current_coarsest_level,QMF),-1,-1) );
        end
    end
    
end
