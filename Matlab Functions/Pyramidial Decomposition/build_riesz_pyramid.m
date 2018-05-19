function [riesz_pyramid, index_matrix] = build_riesz_pyramid(mat_in)
% Returns a multi-scale pyramid of im. pyr is the pyramid concatenated as a
% column vector while pind is the size of each level. im is expected to be
% a grayscale two dimenionsal image in either single floating
% point precision.
%
% Copyright, Neal Wadhwa, August 2014
%
% Part of the Supplementary Material to:
%
% Riesz Pyramids for Fast Phase-Based Video Magnification
% Neal Wadhwa, Michael Rubinstein, Fredo Durand and William T. Freeman
% Computational Photography (ICCP), 2014 IEEE International Conference on

%Get the filter taps:
[ filter_1D_lowpass, filter_1D_highpass, chebychev_polynomial_lowpass, ...
           chebychev_polynomial_highpass, McClellan_transform_matrix, ...
           filter_2D_lowpass_direct, filter_2D_highpass_direct ] = get_filters_for_riesz_pyramid();
chebychev_polynomial_lowpass = reshape(chebychev_polynomial_lowpass, [1 1 numel(chebychev_polynomial_lowpass)]);
chebychev_polynomial_lowpass = 2 * chebychev_polynomial_lowpass; % To make up for the energy lost during downsampling
chebychev_polynomial_highpass = reshape(chebychev_polynomial_highpass, [1 1 numel(chebychev_polynomial_highpass)]);

%Initialize pyramid: 
riesz_pyramid = [];
index_matrix = zeros(0,2);
mat_in_size = size(mat_in);

%Build pyramid:
minimum_mat_size_for_pyramid = 10;
mat_in_lowpass_previous = mat_in;
while any(any(mat_in_size>minimum_mat_size_for_pyramid)) % Stop building the pyramid when the image is too small
    
    Y = zeros(mat_in_size(1), mat_in_size(2), numel(chebychev_polynomial_lowpass),'single'); % 
    Y(:,:,1) = mat_in_lowpass_previous; 
    
    %We apply the McClellan transform repeated to the image:
    for k = 2:numel(chebychev_polynomial_lowpass)
       previousFiltered = Y(:,:,k-1);
       %Reflective boundary conditions:
       previousFiltered = [previousFiltered(2,2), previousFiltered(2,:), previousFiltered(2,end-1); ...
                           previousFiltered(:,2), previousFiltered, previousFiltered(:,end-1); ...
                           previousFiltered(end-1,2), previousFiltered(end-1,:), previousFiltered(end-1,end-1)];            
       Y(:,:,k) = conv2(previousFiltered, McClellan_transform_matrix, 'valid');
    end
    
    %Use Y to compute lo and highpass filtered image:
    mat_in_lowpass_current = sum(bsxfun(@times, Y, chebychev_polynomial_lowpass),3);
    mat_in_highpass_current = sum(bsxfun(@times, Y, chebychev_polynomial_highpass),3);
    
    %Add highpassed image to the pyramid:
    riesz_pyramid = [riesz_pyramid; mat_in_highpass_current(:)];
    index_matrix = [index_matrix; mat_in_size];
    
    %Downsample lowpassed image:
    mat_in_lowpass_current = mat_in_lowpass_current(1:2:end,1:2:end);
    
    %Recurse on the lowpassed image:
    mat_in_size = size(mat_in_lowpass_current);
    mat_in_lowpass_previous = mat_in_lowpass_current;
end
 
%Add a residual level for the remaining low frequencies:
riesz_pyramid =  [riesz_pyramid; mat_in_lowpass_previous(:)];
index_matrix = [index_matrix; mat_in_size];
end

