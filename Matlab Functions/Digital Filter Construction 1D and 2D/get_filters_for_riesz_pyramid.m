function [ filter_1D_lowpass_coefficients, filter_1D_highpass_coefficients, chebychev_polynomial_lowpass, ...
           chebychev_polynomial_highpass, McClellan_transform_matrix, ...
           filter_2D_lowpass_direct, filter_2D_highpass_direct ] = get_filters_for_riesz_pyramid()
% Returns the lowpass and highpass filters specified in the supplementary
% materials of "Riesz Pyramid for Fast Phase-Based Video Magnification"
%
% Copyright Neal Wadhwa, August 2014
%
% Part of the Supplementary Material to:
%
% Riesz Pyramids for Fast Phase-Based Video Magnification
% Neal Wadhwa, Michael Rubinstein, Fredo Durand and William T. Freeman
% Computational Photography (ICCP), 2014 IEEE International Conference on
%
% hL and hH are the one dimenionsal filters designed by our optimization
% bL and bH are the corresponding Chebysheve polynomials
% t is the 3x3 McClellan transform matrix
% directL and directH are the direct forms of the 2d filters

 

filter_1D_lowpass_coefficients =  [-0.0209 -0.0219 0.0900 0.2723 0.3611 0.2723 0.09 -0.0219 -0.0209];
filter_1D_highpass_coefficients =  [0.0099 0.0492 0.1230 0.2020 -0.7633 0.2020 0.1230 0.0492 0.0099];

%These are computed using Chebyshev polynomials, see filterToChebyCoeff for more details:
chebychev_polynomial_lowpass = filter_coefficients_to_chebyshev_polynomial_coefficients(filter_1D_lowpass_coefficients);
chebychev_polynomial_highpass = filter_coefficients_to_chebyshev_polynomial_coefficients(filter_1D_highpass_coefficients);

%McClellan Transform:
McClellan_transform_matrix = [1/8,  1/4, 1/8; ...
                              1/4, -1/2, 1/4; ...
                              1/8,  1/4, 1/8];


%Transform chebyshev polynomial to 2D filter:
filter_2D_lowpass_direct = transform_filter_1D_to_2D(chebychev_polynomial_lowpass, McClellan_transform_matrix);
filter_2D_highpass_direct = transform_filter_1D_to_2D(chebychev_polynomial_highpass, McClellan_transform_matrix);

end

 
%%
%Returns the Chebyshev polynomial coefficients corresponding to a symmetric 1D filter
function chebyshev_polynomial_coefficients = filter_coefficients_to_chebyshev_polynomial_coefficients(filter_1D_coefficients)
    %taps should be an odd symmetric filter:
    filter_length = numel(filter_1D_coefficients);
    filter_center = (filter_length+1)/2; % Number of unique entries
    
    %Compute frequency response:
    % g(1) + g(2)*cos(\omega) + g(3) \cos(2\omega) + ...
    g(1) = filter_1D_coefficients(filter_center);
    g(2:filter_center) = filter_1D_coefficients(filter_center+1:end)*2;
    
    %Only need five polynomials for our filters:
    ChebyshevPolynomial{1} = [0, 0, 0, 0, 1];
    ChebyshevPolynomial{2} = [0, 0, 0, 1, 0];
    ChebyshevPolynomial{3} = [0, 0, 2, 0, -1];
    ChebyshevPolynomial{4} = [0, 4, 0, -3, 0];
    ChebyshevPolynomial{5} = [8, 0, -8, 0, 1];
    
    %-->
    %Chebyshev polynomials are used to transform from cos(n*omega) to cos(omega)^n polynomials:
    %-->
    
    %Now, convert frequency response to polynomials form
    % b(1) + b(2)\cos(\omega) + b(3) \cos(\omega)^2 + ...
    b = zeros(1,filter_center);
    for k = 1:filter_center
       p = ChebyshevPolynomial{k};       
       b = b + g(k)*p;
    end
    chebyshev_polynomial_coefficients = fliplr(b);


end


function filter_2D_impulse_response = transform_filter_1D_to_2D(chebyshev_polynomial_coefficients, McClellan_transform_matrix)
    number_of_chebyshev_polynomial_coefficients = numel(chebyshev_polynomial_coefficients);
    N = 2*number_of_chebyshev_polynomial_coefficients-1;
    
    %Initial an impulse and then filter it:
    impulse_2D = zeros(N,N);
    impulse_2D(number_of_chebyshev_polynomial_coefficients, number_of_chebyshev_polynomial_coefficients)= 1;
    
     
    Y(:,:,1) = impulse_2D;
    for k = 2:numel(chebyshev_polynomial_coefficients);
        % Filter delta function repeatedly with the McClellan transform
        % Size of X is chosen so boundary conditions don't matter
        Y(:,:,k) = conv2(Y(:,:,k-1), McClellan_transform_matrix,'same'); 
    end
    %Take a linear combination of these to get the full 2D response:
    chebyshev_polynomial_coefficients = reshape(chebyshev_polynomial_coefficients, [1 1 numel(chebyshev_polynomial_coefficients)]);    
    filter_2D_impulse_response = sum(bsxfun(@times, Y, chebyshev_polynomial_coefficients),3);
    
end




