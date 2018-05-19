function [u, v] = optic_flow_Theory_of_Warping2(mat1, mat2)

alpha_smoothness = 30.0; %Global smoothness variable.
gamma_derivative_smoothness = 80.0; %Global weight for derivatives.

%Get mat size and number of levels (for what?):
[M_original, N_original, dt] = size(mat1);
number_of_levels = 40; %for face

%Build gaussian smoothing filter for initial smoothing:
max_grid_size = 100;
resize_scaling_factor = power(0.95,number_of_levels);
gaussian_smoothing_filter_sigma = 1/resize_scaling_factor;
minimum_gaussian_filter_value = 10^-3;
gaussian_second_smoothing_axis = linspace(-max_grid_size, max_grid_size, 2*max_grid_size+1);
gaussian_second_smoothing_filter = 1/(sqrt(2*pi)*gaussian_smoothing_filter_sigma) * exp(-gaussian_second_smoothing_axis.^2/(2*gaussian_smoothing_filter_sigma^2));
gaussian_second_smoothing_filter = gaussian_second_smoothing_filter( abs(gaussian_second_smoothing_filter) > abs(minimum_gaussian_filter_value) );
gaussian_second_smoothing_filter = gaussian_second_smoothing_filter / sum(gaussian_second_smoothing_filter);

%Smooth both images:
if (size(mat1,3) == 3)
	%if color image then smooth all three colors:
	mat1_smoothed = mat1;
    mat2_smoothed = mat2;
	mat1_smoothed(:,:,1) = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat1(:,:,1), 'same');
	mat1_smoothed(:,:,2) = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat1(:,:,2), 'same');
	mat1_smoothed(:,:,3) = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat1(:,:,3), 'same');
    mat2_smoothed(:,:,1) = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat2(:,:,1), 'same');
	mat2_smoothed(:,:,2) = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat2(:,:,2), 'same');
	mat2_smoothed(:,:,3) = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat2(:,:,3), 'same');
else
	mat1_smoothed = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat1, 'same');
    mat2_smoothed = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat2, 'same');
end	

%Resize images:
mat1_smoothed_resized = imresize(mat1_smoothed, resize_scaling_factor, 'bilinear', 0);
mat2_smoothed_resized = imresize(mat2_smoothed, resize_scaling_factor, 'bilinear', 0);

%Initialize flow field final solution:
u = zeros(size(rgb2gray(mat1_smoothed_resized)));
v = zeros(size(rgb2gray(mat1_smoothed_resized)));

%Create gaussian derivative filter:
gaussian_derivative_filter_sigma = 1;
gaussian_derivative_filter_minimum_value  = 10^-6;
gaussian_derivative_filter_max_size = 10000; %arbitrarily high to allow filters of any size
gaussian_derivative_filter_axis = linspace(-gaussian_derivative_filter_max_size,gaussian_derivative_filter_max_size,2*gaussian_derivative_filter_max_size+1);
gaussian_derivative_filter = 1/sqrt(2*pi*gaussian_derivative_filter_sigma^2)*exp(-gaussian_derivative_filter_axis.^2/(2*gaussian_derivative_filter_sigma^2));
gaussian_derivative_filter = -gaussian_derivative_filter .* (gaussian_derivative_filter_axis/gaussian_derivative_filter_sigma^2);
gaussian_derivative_filter = gaussian_derivative_filter(abs(gaussian_derivative_filter)>gaussian_derivative_filter_minimum_value);
    
    
    
%Go over the different pyramid levels:
for level_counter = number_of_levels-1 : -1 : 1
    
    %Transform rgb image to gray (LARGE FUNCTION, UNDERSTAND IF CAN BE AVOIDED):
	I1 = rgb2gray(mat1_smoothed_resized);
	I2 = rgb2gray(mat2_smoothed_resized);

	%Compute images derivatives:
    Ikx1 = convn(I1,gaussian_derivative_filter,'same');
    Iky1 = convn(I1,gaussian_derivative_filter','same');
    Ikx2 = convn(I2,gaussian_derivative_filter,'same');
    Iky2 = convn(I2,gaussian_derivative_filter','same');
     
    %Get differential images:
	Ikz = double(I2) - double(I1);
	Ixz = double(Ikx1) - double(Ikx2);
	Iyz = double(Iky1) - double(Iky2);
	
    
    %%%%%%% USE CALCULATED GRADIENTS TO GET FLOW FIELD SOLUTIONS (du,dv):
    %We need to do 3 fixed point steps with 500 iterations in each step.
    number_of_inner_loop_iterations = 10;
    number_of_outer_loop_iterations = 3;
    
    
	% Calling the processing for a particular resolution.
	% Last two arguments are the outer and inner iterations, respectively.
	% 1.8 is the omega value for the SOR iteration.
    alpha_smoothness = 1;
    gamma_derivative_smoothness = 1;
    omega = 1.8;
    [ht, wt, dt] = size(Ikz); 
    du = zeros(ht,wt);
    dv = zeros(ht,wt);
    tol = 1e-8 * ones(2*ht*wt,1);
    duv = zeros( 2 * ht *wt,1);
    
    %Compute second derivatives:
    Ixx = convn(Ikx1,gaussian_derivative_filter,'same');
    Ixy = convn(Ikx1,gaussian_derivative_filter','same');
    Iyx = convn(Iky1,gaussian_derivative_filter,'same');
    Iyy = convn(Iky1,gaussian_derivative_filter','same');
        
    
    %Now for outer_iter iterations:
    for outer_loop_iterations_counter = 1 : number_of_outer_loop_iterations
        %First compute the values of the data and smoothness terms:
        psi_derivative_argument = (Ikz + Ikx1.*du + Iky1.*dv).^2 + ...
            gamma_derivative_smoothness*((Ixz + Ixx.*du + Ixy.*dv).^2 + (Iyz + Ixy.*du + Iyy.*dv).^2);
        psi_derivative_epsilon = 10^-3;
        psi_derivative = 1./(2*sqrt(psi_derivative_argument+psi_derivative_epsilon));

        
        %Compute new psi_derivative_smoothness:.
        %(1).Initializd psi_derivative_smoothness:
        [h,w] = size(u);
        psi_derivative_smoothness = zeros(2*h+1,2*w+1);
        %(2).Compute Derivatives of flow field:
        ux = convn(u,[1,-1]);
        uy = convn(u,[1,-1]');
        vx = convn(v,[1,-1]);
        vy = convn(v,[1,-1]');
        %(3).Smooth Derivatives of flow field to add to gradient magnitude term (IS THIS NECESSARY????):
        uy_double_averaged = convn(convn(uy,[1,1]'/2,'valid'), [1,1]/2); % Computes the delta u(i+1/2, j) and delta u(i-1/2, j).
        ux_double_averaged = convn(convn(ux,[1,1]/2,'valid'), [1,1]'/2); % Computes the delta u(i, j+1/2) and delta u(i, j-1/2).
        vy_double_averaged = convn(convn(vy,[1,1]'/2,'valid'), [1,1]/2); % Computes the delta v(i+1/2, j) and delta v(i-1/2, j).
        vx_double_averaged = convn(convn(vx,[1,1]/2,'valid'), [1,1]'/2); % Computes the delta v(i+1/2, j) and delta v(i-1/2, j).
        %(4).Finally g( delta u ) (i+1/2, j) and (i-1/2, j) and (i, j+1/2) and (i, j-1/2):
        psi_derivative_smoothness( 1:2:end, 2:2:end ) = 1./(2*sqrt(uy.^2 + vy.^2 + ux_double_averaged.^2 + vx_double_averaged.^2 + psi_derivative_epsilon));
        psi_derivative_smoothness( 2:2:end, 1:2:end ) = 1./(2*sqrt(ux.^2 + vx.^2 + uy_double_averaged.^2 + vy_double_averaged.^2 + psi_derivative_epsilon));
        psi_derivative_smoothness = alpha_smoothness * psi_derivative_smoothness;
        
        %Constructing the new matrix:
        [ht, wt] = size(u);
        %small adjustment. remove if necessary. respecting the boundary conditions for all the images:
        psi_derivative_smoothness(1,:) = 0;
        psi_derivative_smoothness(:,1) = 0;
        psi_derivative_smoothness(end,:) = 0;
        psi_derivative_smoothness(:,end) = 0;
        %sigma (j belongs N(i)) (psi dash) (i~j) (k,l):
        psi_derivative_sum = psi_derivative_smoothness(1:2:2*ht, 2:2:end) + ...
                             psi_derivative_smoothness(3:2:end, 2:2:end ) + ...
                             psi_derivative_smoothness(2:2:end, 1:2:2*wt) + ...
                             psi_derivative_smoothness(2:2:end, 3:2:end) ;
        
        %Then construct the A and b matrices:
        %(1). First construct design matrix A:
        %Get ordinary axis from 1 to twice the number of pixels (the 2 factor comes from the two axes ux,uy):
        ordinary_pixel_number_axis = repmat(1:2*ht*wt, 6, 1);
        rows_sparse =  ordinary_pixel_number_axis(:);	% Rows
        columns_sparse = rows_sparse; % Cols
        columns_sparse(1:6:end) = rows_sparse(1:6:end) - 2*ht;	% x-1
        columns_sparse(2:6:end) = rows_sparse(2:6:end) - 2;			% y-1
        columns_sparse(9:12:end) = rows_sparse(9:12:end) - 1;		% v
        columns_sparse(4:12:end) = rows_sparse(4:12:end) + 1;		% u
        columns_sparse(5:6:end) = rows_sparse(5:6:end) + 2;			% y+1
        columns_sparse(6:6:end) = rows_sparse(6:6:end) + 2*ht;	% x+1
        
        
        %Get values for A design matrix: 
        %argument to u(i) in the first of the 2 linear Euler Lagrange equations:
        uapp = psi_derivative.*(Ikx1.^2 + gamma_derivative_smoothness*(Ixx.^2 + Ixy.^2)) + psi_derivative_sum;
        %argument to v(i) in the second of the 2 linear Euler Lagrange equations:
        vapp = psi_derivative.*(Iky1.^2 + gamma_derivative_smoothness*(Iyy.^2 + Ixy.^2)) + psi_derivative_sum;
        %argument to v(i) in the first of the 2 linear Euler Lagrange equations:
        uvapp = psi_derivative.*(Ikx1.*Iky1 + gamma_derivative_smoothness*(Ixx.*Ixy + Iyy.*Ixy));
        %argument to u(i) in the second of the 2 linear Euler Lagrange equations:
        vuapp = psi_derivative.*(Ikx1.*Iky1 + gamma_derivative_smoothness*(Ixx.*Ixy + Iyy.*Ixy));
        %Initiailize values_sparse:
        values_sparse = zeros(size(rows_sparse));
        values_sparse(3:12:end) = uapp(:);
        values_sparse(10:12:end) = vapp(:);
        values_sparse(4:12:end) = uvapp(:);
        values_sparse(9:12:end) = vuapp(:);
        %arguments to u(j) in the linear Euler Lagrange equations:
        psi_derivative_smoothness_ux_sparse_values = psi_derivative_smoothness(2:2:end, 1:2:2*wt);
        values_sparse(1:12:end) = -psi_derivative_smoothness_ux_sparse_values(:);
        values_sparse(7:12:end) = -psi_derivative_smoothness_ux_sparse_values(:);
        psi_derivative_smoothness_ux_sparse_values = psi_derivative_smoothness(2:2:end, 3:2:end);
        values_sparse(6:12:end) = -psi_derivative_smoothness_ux_sparse_values(:);
        values_sparse(12:12:end) = -psi_derivative_smoothness_ux_sparse_values(:);
        %arguments to v(j) in the linear Euler Lagrange equations:
        psi_derivative_smoothness_uy_sparse_values = psi_derivative_smoothness(1:2:2*ht, 2:2:end);
        values_sparse(2:12:end) = -psi_derivative_smoothness_uy_sparse_values(:);
        values_sparse(8:12:end) = -psi_derivative_smoothness_uy_sparse_values(:);
        psi_derivative_smoothness_uy_sparse_values = psi_derivative_smoothness(3:2:end, 2:2:end);
        values_sparse(5:12:end) = -psi_derivative_smoothness_uy_sparse_values(:);
        values_sparse(11:12:end) = -psi_derivative_smoothness_uy_sparse_values(:);
        %Now trim:
        ind = (columns_sparse > 0);
        rows_sparse = rows_sparse(ind);
        columns_sparse = columns_sparse(ind);
        values_sparse = values_sparse(ind);
        ind = (columns_sparse < 2*ht*wt+1);
        rows_sparse = rows_sparse(ind);
        columns_sparse = columns_sparse(ind);
        values_sparse = values_sparse(ind);
        %Construct A design matrix:
        A = sparse(rows_sparse, columns_sparse, values_sparse);
        
        %Construct the b vector:
        u_padded = padarray(u, [1,1]);
        v_padded = padarray(v, [1,1]);
        %Computing the constant terms for the first of the Euler Lagrange equations:
        pdfaltsumu = psi_derivative_smoothness(2:2:end, 1:2:2*wt) .* ( u_padded(2:ht+1, 1:wt) - u_padded(2:ht+1, 2:wt+1) ) + ...
            psi_derivative_smoothness( 2:2:end, 3:2:end) .* ( u_padded(2:ht+1, 3:end) - u_padded(2:ht+1, 2:wt+1) ) + ...
            psi_derivative_smoothness( 1:2:2*ht, 2:2:end) .* ( u_padded(1:ht, 2:wt+1) - u_padded(2:ht+1, 2:wt+1) ) + ...
            psi_derivative_smoothness( 3:2:end, 2:2:end) .* ( u_padded(3:end, 2:wt+1) - u_padded(2:ht+1, 2:wt+1) ) ;
        %Computing the constant terms for the second of the Euler Lagrange equations:
        pdfaltsumv = psi_derivative_smoothness(2:2:end, 1:2:2*wt) .* ( v_padded(2:ht+1, 1:wt) - v_padded(2:ht+1, 2:wt+1) ) + ...
            psi_derivative_smoothness( 2:2:end, 3:2:end) .* ( v_padded(2:ht+1, 3:end) - v_padded(2:ht+1, 2:wt+1) ) + ...
            psi_derivative_smoothness( 1:2:2*ht, 2:2:end) .* ( v_padded(1:ht, 2:wt+1) - v_padded(2:ht+1, 2:wt+1) ) + ...
            psi_derivative_smoothness( 3:2:end, 2:2:end) .* ( v_padded(3:end, 2:wt+1) - v_padded(2:ht+1, 2:wt+1) ) ;
        constu = psi_derivative .* ( Ikx1 .* Ikz + gamma_derivative_smoothness * ( Ixx .* Ixz + Ixy .* Iyz ) ) - pdfaltsumu ;
        constv = psi_derivative .* ( Iky1 .* Ikz + gamma_derivative_smoothness * ( Ixy .* Ixz + Iyy .* Iyz ) ) - pdfaltsumv ;
        b = zeros(2*ht*wt,1);
        b(1:2:end) = -constu(:);
        b(2:2:end) = -constv(:);
        
        
        %Now call SOR for 500 iterations:
        [duv, err, it, flag] = sor(A, duv, b, omega, number_of_inner_loop_iterations, tol);
        
        %Now convert duv into du and dv:
        du(:) = duv(1:2:end);
        dv(:) = duv(2:2:end);
        
    end %END OF OUTER ITERATIONS LOOP
    
    
	%Adding up the optical flow:
	u = u + du;
	v = v + dv;
    
    %Resize images according to new level of pyramid:
    resize_scaling_factor = power(0.95,level_counter);
    gaussian_smoothing_filter_sigma = 1/resize_scaling_factor;
    gaussian_second_smoothing_axis = linspace(-max_grid_size, max_grid_size, 2*max_grid_size+1);
    gaussian_second_smoothing_filter = 1/(sqrt(2*pi)*gaussian_smoothing_filter_sigma) * exp(-gaussian_second_smoothing_axis.^2/(2*gaussian_smoothing_filter_sigma^2));
    gaussian_second_smoothing_filter = gaussian_second_smoothing_filter( abs(gaussian_second_smoothing_filter) > abs(minimum_gaussian_filter_value) );
    gaussian_second_smoothing_filter = gaussian_second_smoothing_filter / sum(gaussian_second_smoothing_filter);
    %Smooth both images:
    if (size(mat1,3) == 3)
        %if color image then smooth all three colors:
        mat1_smoothed = mat1;
        mat2_smoothed = mat2;
        mat1_smoothed(:,:,1) = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat1(:,:,1), 'same');
        mat1_smoothed(:,:,2) = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat1(:,:,2), 'same');
        mat1_smoothed(:,:,3) = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat1(:,:,3), 'same');
        mat2_smoothed(:,:,1) = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat2(:,:,1), 'same');
        mat2_smoothed(:,:,2) = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat2(:,:,2), 'same');
        mat2_smoothed(:,:,3) = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat2(:,:,3), 'same');
    else
        mat1_smoothed = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat1, 'same');
        mat2_smoothed = conv2(gaussian_second_smoothing_filter, gaussian_second_smoothing_filter, mat2, 'same');
    end
    
    %Resize smoothed images:
    mat1_smoothed_resized = imresize(mat1_smoothed, resize_scaling_factor, 'bilinear', 0);
    mat2_smoothed_resized = imresize(mat2_smoothed, resize_scaling_factor, 'bilinear', 0);
    
	%Resize optical flow to current resolution:
	u = imresize(u, [size(mat1_smoothed_resized,1), size(mat1_smoothed_resized,2)], 'bilinear');
	v = imresize(v, [size(mat1_smoothed_resized,1), size(mat1_smoothed_resized,2)], 'bilinear');

    %Warp mat2 closer to mat1:
    mat2_smoothed_resized = zeros(size(mat2_smoothed_resized));
    [h,w,d] = size(mat2_smoothed_resized);
    [uc,vc] = meshgrid(1:w,1:h);
    uc1 = uc + u;
    vc1 = vc + v;
    tmp = zeros(h,w);
    tmp(:) = interp2(uc, vc, double(squeeze(mat2_smoothed_resized(:, :, 1))), uc1(:), vc1(:), 'bilinear') ;
    mat2_smoothed_resized(:, :, 1) = tmp ;
    tmp(:) = interp2(uc, vc, double(squeeze(mat2_smoothed_resized(:, :, 2))), uc1(:), vc1(:), 'bilinear') ;
    mat2_smoothed_resized(:, :, 2) = tmp ;
    tmp(:) = interp2(uc, vc, double(squeeze(mat2_smoothed_resized(:, :, 3))), uc1(:), vc1(:), 'bilinear') ;
    mat2_smoothed_resized(:, :, 3) = tmp ;
    

end %END OF PYRAMID LEVELS LOOP
