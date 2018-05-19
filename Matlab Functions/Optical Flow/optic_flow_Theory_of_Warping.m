function [u, v] = optic_flow_Theory_of_Warping( mat1, mat2 )

%Lagrangian parameters:
alpha_global = 10.0;
alpha_local = 15.0;

%Get mat size and number of levels (for what?):
[M_original, N_original] = size(mat1);
number_of_levels = round( log10(30/min(M_original,N_original)) / log10(0.9) ); %Doesn't make sense, change to constant like 20

%Initialize flow field final solution and current change, respectively:
M_new = fix( power(0.9,number_of_levels) * M_original );
N_new = fix( power(0.9,number_of_levels) * N_original );
u = zeros(M_new, N_new) ;
v = zeros(M_new, N_new) ;
du = zeros(M_new, N_new) ;
dv = zeros(M_new, N_new) ;


%Set up Gaussian derivative filter (both smoothes and derives):
gaussian_derivative_filter_sigma = 1;
gaussian_derivative_filter_minimum_value  = 10^-6;
gaussian_derivative_filter_max_size = 10000; %arbitrarily high to allow filters of any size
gaussian_derivative_filter_axis = linspace(-gaussian_derivative_filter_max_size,gaussian_derivative_filter_max_size,2*gaussian_derivative_filter_max_size+1);
gaussian_derivative_filter = 1/sqrt(2*pi*gaussian_derivative_filter_sigma^2)*exp(-gaussian_derivative_filter_axis.^2/(2*gaussian_derivative_filter_sigma^2));
gaussian_derivative_filter = -gaussian_derivative_filter .* (gaussian_derivative_filter_axis/gaussian_derivative_filter_sigma^2);
gaussian_derivative_filter = gaussian_derivative_filter(abs(gaussian_derivative_filter)>gaussian_derivative_filter_minimum_value);

%Set up Second Gaussian smoothing filter (already smoothed before rescaling):
max_grid_size = 100;
minimum_gaussian_filter_value = 10^-9;
gaussian_smoothing_filter_sigma = 1;
gaussian_smoothing_filter_axis = linspace(-max_grid_size, max_grid_size, 2*max_grid_size+1);
gaussian_second_smoothing_filter = 1/sqrt(2*pi*gaussian_smoothing_filter_sigma^2) * exp(-gaussian_smoothing_filter_axis.^2 / (2*gaussian_smoothing_filter_sigma^2));
gaussian_second_smoothing_filter = gaussian_second_smoothing_filter(abs(gaussian_second_smoothing_filter) > minimum_gaussian_filter_value);
gaussian_second_smoothing_filter = gaussian_second_smoothing_filter / sum(gaussian_second_smoothing_filter);


%Go over the different pyramid levels:
for level_counter = number_of_levels : -1 : 8
	
    %%%%%%%% SMOOTH AND RESCALE IMAGES TO CURRENT PYRAMID DIMENSIONS:
    %(if this takes too long maybe i can use the same smoothing for all levels!?!?!?!)
    %Set up parameters for gaussian smoothing for current level:
    resize_scaling_factor = power(0.9,level_counter);
    %Set Up gaussian filter for image smoothing:
    max_grid_size = 100;
    minimum_gaussian_filter_value = 10^-3;
    gaussian_smoothing_filter_sigma = 1/resize_scaling_factor;
    gaussian_smoothing_filter_axis = linspace(-max_grid_size, max_grid_size, 2*max_grid_size+1);
    gaussian_smoothing_filter_before_rescaling = 1/sqrt(2*pi*gaussian_smoothing_filter_sigma^2) * exp(-gaussian_smoothing_filter_axis.^2 / (2*gaussian_smoothing_filter_sigma^2));
    gaussian_smoothing_filter_before_rescaling = gaussian_smoothing_filter_before_rescaling(abs(gaussian_smoothing_filter_before_rescaling) > minimum_gaussian_filter_value);
    gaussian_smoothing_filter_before_rescaling = gaussian_smoothing_filter_before_rescaling / sum(gaussian_smoothing_filter_before_rescaling);
    %Smooth images:
    if size(mat1,3) == 3
       %if color image then smooth all three colors:
       mat1_smoothed = mat1;
       mat2_smoothed = mat2;
       mat1_smoothed(:,:,1) = conv2(gaussian_smoothing_filter_before_rescaling,gaussian_smoothing_filter_before_rescaling,mat1(:,:,1),'same');
       mat1_smoothed(:,:,2) = conv2(gaussian_smoothing_filter_before_rescaling,gaussian_smoothing_filter_before_rescaling,mat1(:,:,2),'same');
       mat1_smoothed(:,:,3) = conv2(gaussian_smoothing_filter_before_rescaling,gaussian_smoothing_filter_before_rescaling,mat1(:,:,3),'same');
       mat2_smoothed(:,:,1) = conv2(gaussian_smoothing_filter_before_rescaling,gaussian_smoothing_filter_before_rescaling,mat2(:,:,1),'same');
       mat2_smoothed(:,:,2) = conv2(gaussian_smoothing_filter_before_rescaling,gaussian_smoothing_filter_before_rescaling,mat2(:,:,2),'same');
       mat2_smoothed(:,:,3) = conv2(gaussian_smoothing_filter_before_rescaling,gaussian_smoothing_filter_before_rescaling,mat2(:,:,3),'same');
    else
       mat1_smoothed = conv2(gaussian_smoothing_filter_before_rescaling,gaussian_smoothing_filter_before_rescaling,mat1,'same'); 
       mat2_smoothed = conv2(gaussian_smoothing_filter_before_rescaling,gaussian_smoothing_filter_before_rescaling,mat2,'same'); 
    end
    %Rescale images using imresize:
    mat1_smoothed_rescaled = imresize(mat1_smoothed, resize_scaling_factor, 'bilinear', 0);
    mat2_smoothed_rescaled = imresize(mat2_smoothed, resize_scaling_factor, 'bilinear', 0);
	[M_new, N_new] = size(mat1_smoothed_rescaled) ;
    
    %Shift (Warp) mat2 to former solution position:
    [X_mat2,Y_mat2] = meshgrid(1:N_new,1:M_new);
    X_mat2_new = X_mat2 + u;
    Y_mat2_new = Y_mat2 + v;
    mat2_smoothed_rescaled_shifted = zeros(M_new,N_new,d);
    mat2_smoothed_rescaled_shifted(:,:,1) = interp2(X_mat2,Y_mat2,squeeze(mat2_smoothed_rescaled(:,:,1)),X_mat2_new,Y_mat2_new);
    mat2_smoothed_rescaled_shifted(:,:,2) = interp2(X_mat2,Y_mat2,squeeze(mat2_smoothed_rescaled(:,:,2)),X_mat2_new,Y_mat2_new);
    mat2_smoothed_rescaled_shifted(:,:,3) = interp2(X_mat2,Y_mat2,squeeze(mat2_smoothed_rescaled(:,:,3)),X_mat2_new,Y_mat2_new);
    
    
    %%%%%%%% CALCULATE GRADIENT USING GAUSSIAN DERIVATIVE FILTER (SMOOTH AND DERIVE):
    %Set up gradient gaussian term for the lagrangian:
    mat2_smoothed_rescaled_gray = rgb2gray(mat2_smoothed_rescaled);
    Ix = convn( mat2_smoothed_rescaled_gray, gaussian_derivative_filter, 'same' ) ;
    Iy = convn( mat2_smoothed_rescaled_gray, gaussian_derivative_filter', 'same' ) ;
    
    %Get LAGRANGIAN E_smoothness term coefficient:
    lagrangian_gradient_gaussian_term_sigma = 2;
    gradient_argument_gaussian_function_for_lagrangian = 1/sqrt(2*pi*lagrangian_gradient_gaussian_term_sigma^2) * ...
                                                    exp( -(Ix.^2+Iy.^2)/(2*lagrangian_gradient_gaussian_term_sigma^2));
    alpha_smoothness_term = alpha_global*ones(M_new,N_new) + alpha_local*gradient_argument_gaussian_function_for_lagrangian;
    
    
    %Smooth images AGAIN (IS THIS REALLY NECESSARY?):
    if size(mat1,3) == 3
       %if color image then smooth all three colors:
       mat1_smoothed_rescaled_smoothed(:,:,1) = conv2(gaussian_second_smoothing_filter,gaussian_second_smoothing_filter,mat1_smoothed_rescaled(:,:,1),'same');
       mat1_smoothed_rescaled_smoothed(:,:,2) = conv2(gaussian_second_smoothing_filter,gaussian_second_smoothing_filter,mat1_smoothed_rescaled(:,:,2),'same');
       mat1_smoothed_rescaled_smoothed(:,:,3) = conv2(gaussian_second_smoothing_filter,gaussian_second_smoothing_filter,mat1_smoothed_rescaled(:,:,3),'same');
       mat2_smoothed_rescaled_shifted_smoothed(:,:,1) = conv2(gaussian_second_smoothing_filter,gaussian_second_smoothing_filter,mat2_smoothed_rescaled_shifted(:,:,1),'same');
       mat2_smoothed_rescaled_shifted_smoothed(:,:,2) = conv2(gaussian_second_smoothing_filter,gaussian_second_smoothing_filter,mat2_smoothed_rescaled_shifted(:,:,2),'same');
       mat2_smoothed_rescaled_shifted_smoothed(:,:,3) = conv2(gaussian_second_smoothing_filter,gaussian_second_smoothing_filter,mat2_smoothed_rescaled_shifted(:,:,3),'same');
    else
       mat1_smoothed_rescaled_smoothed = conv2(gaussian_second_smoothing_filter,gaussian_second_smoothing_filter,mat1,'same'); 
       mat2_smoothed_rescaled_shifted_smoothed = conv2(gaussian_second_smoothing_filter,gaussian_second_smoothing_filter,mat2,'same'); 
    end
    
    
    %Get the different channels:
    %This function takes an image and returns the different channels that need to be made out of it.
    %Currently the channels are:
    % 1) Grayscale intensity.
    % 2) Green - Red
    % 3) Green - Blue
    % 4) x-derivative of intensity
    % 5) y-derivative of intensity
    % 2nd and 3rd components are scaled to increase numerical stability
    % (WHY NOT SIMPLY USE {RED,GREEN,BLUE} DIRECTLY????????)
    %%%%%%%%%%% Get the different Gradient (Derivatives) Components:
    %Set up scaling factor for Green-Red & Green-Blue mats and Initialize I1 and I2:
    color_difference_channels_scaling = 0.25;
    number_of_channels = 5;
    I1 = zeros(M_new,N_new,number_of_channels);
    %First component is the grayscale intensity:
    I1(:, :, 1) = rgb2gray(mat1_smoothed_rescaled_smoothed);
    I2(:, :, 1) = rgb2gray(mat2_smoothed_rescaled_shifted_smoothed);
    %Second and Third components are the Green-Red and Green-Blue difference, respectively, SCALED:
    I1(:, :, 2) = mat1_smoothed_rescaled_smoothed(:, :, 2) - mat1_smoothed_rescaled_smoothed(:, :, 1);
    I1(:, :, 3) = mat1_smoothed_rescaled_smoothed(:, :, 2) - mat1_smoothed_rescaled_smoothed(:, :, 3);
    I1(:, :, 2) = I1(:, :, 2) * color_difference_channels_scaling ;
    I1(:, :, 3) = I1(:, :, 3) * color_difference_channels_scaling ;
    I2(:, :, 2) = mat2_smoothed_rescaled_shifted_smoothed(:, :, 2) - mat2_smoothed_rescaled_shifted_smoothed(:, :, 1);
    I2(:, :, 3) = mat2_smoothed_rescaled_shifted_smoothed(:, :, 2) - mat2_smoothed_rescaled_shifted_smoothed(:, :, 3);
    I2(:, :, 2) = I2(:, :, 2) * color_difference_channels_scaling ;
    I2(:, :, 3) = I2(:, :, 3) * color_difference_channels_scaling ;
    %Fourth and Fifth components are the grayscale intensity gradients (again smoothing plus derivative):
    I1(:, :, 4) = convn( I1(:,:,1), gaussian_derivative_filter, 'same') ;
    I1(:, :, 5) = convn( I1(:,:,1), gaussian_derivative_filter', 'same') ;
    I2(:, :, 4) = convn( I2(:,:,1), gaussian_derivative_filter, 'same') ;
    I2(:, :, 5) = convn( I2(:,:,1), gaussian_derivative_filter', 'same') ;
    
    
    %Calculate spatial and temporal gradients for the different channels:
    Ikx = convn( I2, gd, 'same');
    Iky = convn( I2, gd', 'same');
    Ikt = I2-I1;
    [M_new, N_new, number_of_channels] = size(Ikt);
    
    
    %%%%%%% USE CALCULATED GRADIENTS TO GET FLOW FIELD SOLUTIONS (du,dv):
    %We need to do 3 fixed point steps with 500 iterations in each step.
    number_of_inner_loop_iterations = 500;
    number_of_outer_loop_iterations = 3;
    %Initialize du,dv,duv:
    du = zeros(M_new, N_new);
    dv = zeros(M_new, N_new);
    %Now for the 3 outer iterations.
    for outer_loop_iterations_counter = 1:number_of_outer_loop_iterations
        %First compute the values of the data term:
        % psi = sqrt(x+eps) -->  psi' = 1/(2*sqrt(x+eps))
        epsilon = 10^-3;
        psi_data_argument = (Ikx.*repmat(du,[1,1,number_of_channels]) + Iky.*repmat(dv,[1,1,number_of_channels]) + Ikt).^2;
        psi_derivative_data = 1./(2*sqrt(psi_data_argument+epsilon));
        
        
        %Second Compute the values of the smoothness term:
        %(1).Initializd psi_derivative_smoothness:
        [Mu,Nu] = size(u);
        psi_derivative_smoothness = zeros(2*Mu+1, 2*Nu+1);
        %(2).Resize alphImg by interpolation:
        alpha_smoothness_rescaled = imresize(alpha_smoothness_term, [2*Mu+1,2*Nu+1], 'bilinear');
        %(3).Compute Derivatives of flow field:
        ux = convn(u, [1,-1]);
        uy = convn(u, [1,-1]');
        vx = convn(v, [1,-1]);
        vy = convn(v, [1,-1]');
        %(3).Smooth Derivatives of flow field to add to gradient magnitude term (IS THIS NECESSARY????):
        ux_double_averaged = convn(convn(ux,[1,1]/2,'valid'),[1,1]'/2);
        uy_double_averaged = convn(convn(uy,[1,1]'/2,'valid'),[1,1]/2);
        vx_double_averaged = convn(convn(vx,[1,1]/2,'valid'),[1,1]'/2);
        vy_double_averaged = convn(convn(vy,[1,1]'/2,'valid'),[1,1]/2);
        %(4).Compute psi_derivative of flow field derivative vector magnitude squared as argument:
        psi_derivative_smoothness(1:2:end, 2:2:end) = 1./(2*sqrt(uy.^2+vy.^2 + ux_double_averaged.^2+vx_double_averaged.^2));
        psi_derivative_smoothness(2:2:end, 1:2:end) = 1./(2*sqrt(ux.^2+vx.^2 + uy_double_averaged.^2+vy_double_averaged.^2));
        psi_derivative_smoothness = alpha_smoothness_rescaled .* psi_derivative_smoothness;
        %(5). small adjustment. remove if necessary. respecting the boundary conditions for all the images:
        psi_derivative_smoothness(1,:) = 0;
        psi_derivative_smoothness(:,1) = 0;
        psi_derivative_smoothness(end,:) = 0;
        psi_derivative_smoothness(:,end) = 0;
        %(6). Get the sum over the different channels of the psi_derivative_smoothness:
        psi_derivative_smoothness_sum = psi_derivative_smoothness(1:2:2*Mu, 2:2:end) + psi_derivative_smoothness(3:2:end, 2:2:end) + ...
            psi_derivative_smoothness(2:2:end, 1:2:2*Nu) + psi_derivative_smoothness(2:2:end, 3:2:end);
        
        %Construt linear matrix equations parameters A & b:
        %(1). First construct design matrix A:
        %Get ordinary axis from 1 to twice the number of pixels (the 2 factor comes from the two axes ux,uy):
        ordinary_pixel_number_axis = repmat(1:2*Mu*Nu, 6, 1);
        rows_sparse = ordinary_pixel_number_axis(:); 
        column_sparse = rows_sparse; 
        column_sparse(1:6:end) = rows_sparse(1:6:end) - 2*Mu;	% x-1
        column_sparse(2:6:end) = rows_sparse(2:6:end) - 2;			% y-1
        column_sparse(9:12:end) = rows_sparse(9:12:end) - 1;		% v
        column_sparse(4:12:end) = rows_sparse(4:12:end) + 1;		% u
        column_sparse(5:6:end) = rows_sparse(5:6:end) + 2;			% y+1
        column_sparse(6:6:end) = rows_sparse(6:6:end) + 2*Mu;	% x+1
        
        %Get values for A design matrix: 
        uapp = sum(psi_derivative_data.*(Ikx.^2), 3) + psi_derivative_smoothness_sum ;
        vapp = sum(psi_derivative_data.*(Iky.^2), 3) + psi_derivative_smoothness_sum ;
        uvapp = sum(psi_derivative_data.*(Ikx.*Iky), 3) ;
        vuapp = sum(psi_derivative_data.*(Ikx.*Iky), 3) ;
        values_sparse = zeros(size(rows_sparse));
        values_sparse(3:12:end) = uapp(:);
        values_sparse(10:12:end) = vapp(:);
        values_sparse(4:12:end) = uvapp(:);
        values_sparse(9:12:end) = vuapp(:);
        psi_derivative_smoothness_ux_sparse_values = psi_derivative_smoothness(2:2:end, 1:2:2*Nu);
        values_sparse(1:12:end) = -psi_derivative_smoothness_ux_sparse_values(:);
        values_sparse(7:12:end) = -psi_derivative_smoothness_ux_sparse_values(:);
        psi_derivative_smoothness_ux_sparse_values = psi_derivative_smoothness(2:2:end, 3:2:end);
        values_sparse(6:12:end) = -psi_derivative_smoothness_ux_sparse_values(:);
        values_sparse(12:12:end) = -psi_derivative_smoothness_ux_sparse_values(:);
        psi_derivative_smoothness_uy_sparse_values = psi_derivative_smoothness(1:2:2*Mu, 2:2:end);
        values_sparse(2:12:end) = -psi_derivative_smoothness_uy_sparse_values(:);
        values_sparse(8:12:end) = -psi_derivative_smoothness_uy_sparse_values(:);
        psi_derivative_smoothness_uy_sparse_values = psi_derivative_smoothness(3:2:end, 2:2:end);
        values_sparse(5:12:end) = -psi_derivative_smoothness_uy_sparse_values(:);
        values_sparse(11:12:end) = -psi_derivative_smoothness_uy_sparse_values(:);
        %Now trim:
        ind = find(column_sparse>0);
        rows_sparse = rows_sparse(ind);
        column_sparse = column_sparse(ind);
        values_sparse = values_sparse(ind);
        ind = find(column_sparse < 2*Mu*Nu+1);
        rows_sparse = rows_sparse(ind);
        column_sparse = column_sparse(ind);
        values_sparse = values_sparse(ind);
        A_design_matrix = sparse(rows_sparse,column_sparse,values_sparse);
        
        %Construct the b vector:
        u_padded = padarray(u, [1,1]);
        v_padded = padarray(v, [1,1]);
        pdfaltsumu = psi_derivative_smoothness(2:2:end, 1:2:2*Nu) .* (u_padded(2:Mu+1, 1:Nu) - u_padded(2:Mu+1, 2:Nu+1)) + ...
            psi_derivative_smoothness(2:2:end, 3:2:end) .* (u_padded(2:Mu+1, 3:end)-u_padded(2:Mu+1, 2:Nu+1)) + ...
            psi_derivative_smoothness(1:2:2*Mu, 2:2:end) .* (u_padded(1:Mu, 2:Nu+1)-u_padded(2:Mu+1, 2:Nu+1)) + ...
            psi_derivative_smoothness(3:2:end, 2:2:end) .* (u_padded(3:end, 2:Nu+1)-u_padded(2:Mu+1, 2:Nu+1));
        pdfaltsumv = psi_derivative_smoothness(2:2:end, 1:2:2*Nu) .* (v_padded(2:Mu+1, 1:Nu)-v_padded(2:Mu+1, 2:Nu+1)) + ...
            psi_derivative_smoothness(2:2:end, 3:2:end) .* (v_padded(2:Mu+1, 3:end)-v_padded(2:Mu+1, 2:Nu+1)) + ...
            psi_derivative_smoothness(1:2:2*Mu, 2:2:end) .* (v_padded(1:Mu, 2:Nu+1)-v_padded(2:Mu+1, 2:Nu+1)) + ...
            psi_derivative_smoothness(3:2:end, 2:2:end) .* (v_padded(3:end, 2:Nu+1)-v_padded(2:Mu+1, 2:Nu+1));
        constu = sum(psi_derivative_data.*(Ikx.*Ikt), 3) - pdfaltsumu;
        constv = sum(psi_derivative_data.*(Iky.*Ikt), 3) - pdfaltsumv;
        b = zeros(2*Mu*Nu, 1);
        b(1:2:end) = -constu(:);
        b(2:2:end) = -constv(:);
        
        
        
        %Successive Over-Relaxation Method (SOR, Gauss-Seidel method when omega = 1) to solve linear equation A*x=b:
        %Check other optimization methods to see which is the fastest and most appropriate 
        %(also try more accurate methods)
        duv = zeros(2*M_new*N_new,1);
        x = duv;
        omega = 1.9;
        tolerance = 10^-5 * ones(2*M_new*N_new,1);
        [duv, err, it, flag] = sor(A_design_matrix, x, b, omega, number_of_inner_loop_iterations, tolerance);
   
        %Now convert duv into du and dv:
        du(:) = duv(1:2:end);
        dv(:) = duv(2:2:end);
        
    end %END OF OUTER ITERATIONS LOOP
    

    %Get dimensions of the next level:
	M_new = fix(power(0.9, level_counter-1)*M_original);
	N_new = fix(power(0.9, level_counter-1)*N_original);

	u = imresize(u+du, [M_new N_new], 'bilinear');
	v = imresize(v+dv, [M_new N_new], 'bilinear');

end %END OF PYRAMID LEVELS LOOP
