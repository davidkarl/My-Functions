function reconstructed = reconstruct_riesz_pyramid(riesz_pyramid, index_matrix)
% Collapases a multi-scale pyramid of and returns the reconstructed image.
% pyr is a column vector, in which each level of the pyramid is
% concatenated, pind is the size of each level. 
%
% Copyright, Neal Wadhwa, August 2014
%

% Get the filter taps
% Because we won't be simultaneously lowpassing/highpassing anything and 
% most of the computational savings comes from the simultaneous application 
% of the filters, we use the direct form of the filters rather the 
% McClellan transform form
[ filter_1D_lowpass_coefficients, filter_1D_highpass_coefficients, chebychev_polynomial_lowpass, ...
           chebychev_polynomial_highpass, McClellan_transform_matrix, ...
           filter_2D_lowpass_direct, filter_2D_highpass_direct ] = get_filters_for_riesz_pyramid();
filter_2D_lowpass_direct = 2*filter_2D_lowpass_direct; % To make up for the energy lost during downsampling

 
number_of_subbands = size(index_matrix,1);
lowpass_previous = get_pyramid_subband(riesz_pyramid, index_matrix, number_of_subbands);
for k = number_of_subbands:-1:2
    
    %Initialize upsampled image:
    upsampled_image_size = index_matrix(k-1,:);
    
    %Upsample the lowest level:
    lowest = zeros(upsampled_image_size, 'single');
    lowest(1:2:end,1:2:end) = lowpass_previous;
     
    %Lowpass it with reflective boundary conditions:
    lowest = [lowest(5:-1:2,5:-1:2),         lowest(5:-1:2,:),         lowest(5:-1:2, end-1:-1:end-4); ...
              lowest(:,5:-1:2),              lowest,                   lowest(:,end-1:-1:end-4); ...
              lowest(end-1:-1:end-4,5:-1:2), lowest(end-1:-1:end-4,:), lowest(end-1:-1:end-4,end-1:-1:end-4)];
    lowest = conv2(lowest, filter_2D_lowpass_direct, 'valid');
     
    %Get the next level:
    nextLevel = get_pyramid_subband(riesz_pyramid, index_matrix, k-1);
    nextLevel = [nextLevel(5:-1:2,5:-1:2),         nextLevel(5:-1:2,:),         nextLevel(5:-1:2, end-1:-1:end-4); ...
                 nextLevel(:,5:-1:2),              nextLevel,                   nextLevel(:,end-1:-1:end-4); ...
                 nextLevel(end-1:-1:end-4,5:-1:2), nextLevel(end-1:-1:end-4,:), nextLevel(end-1:-1:end-4,end-1:-1:end-4)];
    
    %Highpass the level and add it to lowest level to form a new lowest level:
    lowest = lowest + conv2(nextLevel, filter_2D_highpass_direct, 'valid');
    lowpass_previous = lowest;
end
reconstructed = lowpass_previous;
end

