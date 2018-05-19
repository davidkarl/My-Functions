function [Optical_flow_over_time,point_where_gaussian_evelope_is_10_percent] = ...
                                                optical_flow_phase_based(Image_sequence, ...
                                                                         number_of_velocity_vectors_along_x_axis, ...
                                                                         linearity_threshold, ...
                                                                         min_number_of_valid_velocity_components)

% Usage: O = optical_flow (II, gx, thres_lin, nc_min)
%	II [sy sx st]	Image Sequence (Y-X-t)
%	gx		Number of velocity vectors along X-axis (0=all)
%	thres_lin	Linearity threshold [.05]
%	nc_min		Minimal number of valid component velocities for
%				computation of full velocity [5]

[Nx,Ny,Nz] = size(Image_sequence);
 
if (nargin<2)
	number_of_velocity_vectors_along_x_axis = 0;
end
if (nargin<3)
	linearity_threshold = 0.05;
end
if (nargin<4)
	min_number_of_valid_velocity_components = 5;
end


if (number_of_velocity_vectors_along_x_axis == 0)
	pixel_intervals = 1;
else
	pixel_intervals = floor(Ny/number_of_velocity_vectors_along_x_axis);
	pixel_intervals = pixel_intervals + (pixel_intervals==0);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Filterbank Parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gabor_filter_frequencies = [	0.02156825 -0.08049382; ...
                                0.05892557 -0.05892557; ...
                                0.08049382 -0.02156825; ...
                                0.08049382 0.02156825; ...
                                0.05892557 0.05892557; ...
                                0.02156825 0.08049382; ...
                                0.06315486 -0.10938742; ...
                                0.10938742 -0.06315486; ...
                                0.12630971 0.00000000; ...
                                0.10938742 0.06315486; ...
                                0.06315486 0.10938742];
gabor_filter_gaussian_envelope_sigma = [9.31648319 9.31648319 9.31648319 9.31648319 9.31648319 9.31648319 ...
	6.14658664 6.14658664 6.14658664 6.14658664 6.14658664]';
number_of_frequencies = size(gabor_filter_frequencies,1);

%%%%%%%
% Aux %
%%%%%%%
temporal_axis = (1:Nz);
temporal_axis_3D = zeros(1,1,Nz);
temporal_axis_3D(1:Nz) = 1:Nz;
Sum_of_t_squared = sum(temporal_axis.^2);
Sum_of_t = sum(temporal_axis);
den = (Nz.*Sum_of_t_squared-Sum_of_t.^2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Filter Outputs & Component Velocities %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tclock1 = clock;
AC = zeros(Nx,Ny,Nz);
AS = zeros(Nx,Ny,Nz);
filter_component_velocity = zeros(number_of_frequencies,Nx,Ny);	% Filter Component Velocity
regression_MSE = zeros(number_of_frequencies,Nx,Ny);	% MSE of Regression
point_where_gaussian_evelope_is_10_percent = ceil(max(gabor_filter_gaussian_envelope_sigma)*sqrt(log(100)));	% Point where Gaussian envelope drops to 10%


for frequency_counter = 1:number_of_frequencies
    
	%Generate 1D kernels:
	window_length = floor(6*gabor_filter_gaussian_envelope_sigma(frequency_counter));
	centered_window_axis = (0:window_length-1)' - window_length/2 + 0.5;
	Gaussian_envelope = exp(-centered_window_axis.^2./(2*gabor_filter_gaussian_envelope_sigma(frequency_counter)^2)) ./ (sqrt(2*pi)*gabor_filter_gaussian_envelope_sigma(frequency_counter));
	FCx = Gaussian_envelope .* cos(2*pi*gabor_filter_frequencies(frequency_counter,1).*centered_window_axis);
	FCy = Gaussian_envelope .* cos(2*pi*gabor_filter_frequencies(frequency_counter,2).*centered_window_axis);
	FSx = Gaussian_envelope .* sin(2*pi*gabor_filter_frequencies(frequency_counter,1).*centered_window_axis);
	FSy = Gaussian_envelope .* sin(2*pi*gabor_filter_frequencies(frequency_counter,2).*centered_window_axis);

	%Exceptions for null frequencies:
	if (sum(FCx.^2)==0)
		FCx = ones(window_length,1);
	end
	if (sum(FCy.^2)==0)
		FCy = ones(window_length,1);
	end
	if (sum(FSx.^2)==0)
		FSx = ones(window_length,1);
	end
	if (sum(FSy.^2)==0)
		FSy = ones(window_length,1);
	end

	%Perform Convolutions, room for improvement (subsampling):
	for image_counter = 1:Nz
		current_image = Image_sequence(:,:,image_counter)';

		%Sine Filter:
		current_image_FSx_smoothed = conv2(current_image, FSx, 'same')';
		current_image_FSx_FCy_smoothed = conv2(current_image_FSx_smoothed, FCy, 'same');
		current_image_FCx_smoothed = conv2(current_image, FCx, 'same')';
		current_image_FCx_FSy_smoothed = conv2(current_image_FCx_smoothed, FSy, 'same');
		AS(:,:,image_counter) = current_image_FSx_FCy_smoothed + current_image_FCx_FSy_smoothed;

		%Cosine Filter:
		%Tcx = conv2(IIp, FCx, 'same')';
		current_image_FSx_FCy_smoothed = conv2(current_image_FCx_smoothed, FCy, 'same');
		%Tsx = conv2(IIp, FSx, 'same')';
		current_image_FCx_FSy_smoothed = conv2(current_image_FSx_smoothed, FSy, 'same');
		AC(:,:,image_counter) = current_image_FSx_FCy_smoothed - current_image_FCx_FSy_smoothed;
	end

	%Compute and Unwrap Phase:
	Mcos = (AC==0);
	Phase_unwrapped = atan(AS./(AC+Mcos)) + pi.*(AC<0); %the second term is because atan is no single valued
	Phase_unwrapped(Mcos) = NaN;
	k = 2;
	while (k<=Nz)
		Phase_temporal_difference = Phase_unwrapped(:,:,k) - Phase_unwrapped(:,:,k-1); 
		Phase_temporal_difference_above_pi_logical_mask = abs(Phase_temporal_difference)>pi;
		Phase_unwrapped(:,:,k:Nz) = Phase_unwrapped(:,:,k:Nz) - ...
              repmat(2*pi*sign(Phase_temporal_difference).*Phase_temporal_difference_above_pi_logical_mask,[1,1,Nz-k+1]);
		k = k + (sum(sum(Phase_temporal_difference_above_pi_logical_mask))==0);
	end

	%Use Regression to find a straight line fit over time for each pixel phase:
	Sxy = sum( repmat(temporal_axis_3D,[Nx,Ny,1]).*Phase_unwrapped, 3);
	Sy = sum(Phase_unwrapped,3);
	phase_temporal_gradient_intercept = (Sum_of_t_squared.*Sy-Sum_of_t.*Sxy) ./ (Nz.*Sum_of_t_squared-Sum_of_t.^2);
	phase_temporal_gradient_slope = (Nz.*Sxy-Sum_of_t.*Sy) ./ (Nz.*Sum_of_t_squared-Sum_of_t.^2);
	Reg = repmat(phase_temporal_gradient_intercept,[1,1,Nz]) + repmat(phase_temporal_gradient_slope,[1,1,Nz]).*repmat(temporal_axis_3D,[Nx,Ny,1]);
	regression_MSE(frequency_counter,:,:) = mean((Reg-Phase_unwrapped).^2,3) ./ abs(phase_temporal_gradient_slope+(phase_temporal_gradient_slope==0));
    
    %Compute Filter Component Velocity:
	filter_component_velocity(frequency_counter,:,:) = -phase_temporal_gradient_slope .* (gabor_filter_frequencies(frequency_counter,1)+1i*gabor_filter_frequencies(frequency_counter,2)) ...
                                                        ./ (2*pi*sum(gabor_filter_frequencies(frequency_counter,:).^2));

end
tclock2 = clock;
time1 = etime(tclock2,tclock1);



%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Full Velocity %
%%%%%%%%%%%%%%%%%%%%%%%%%
Optical_flow_over_time = repmat(NaN, [Nx Ny 2]);
for i = point_where_gaussian_evelope_is_10_percent+1 : pixel_intervals : (Nx-point_where_gaussian_evelope_is_10_percent)
  for j = point_where_gaussian_evelope_is_10_percent+1 : pixel_intervals : (Ny-point_where_gaussian_evelope_is_10_percent)

	% Linearity Check
	valid_linear_phase_indices = find(regression_MSE(:,i,j)<linearity_threshold);
	V_current_pixel_velocity_valid_frequencies = filter_component_velocity(valid_linear_phase_indices,i,j);
	number_of_valid_linear_phase_indices = length(valid_linear_phase_indices);

	if (number_of_valid_linear_phase_indices>=min_number_of_valid_velocity_components)
		V_magnitude = V_current_pixel_velocity_valid_frequencies.*conj(V_current_pixel_velocity_valid_frequencies);
		Vx = real(V_current_pixel_velocity_valid_frequencies);
		Vy = imag(V_current_pixel_velocity_valid_frequencies);
		sumX = sum(Vx);
		sumY = sum(Vy);
		sumXY_normalized = sum(Vx.*Vy./V_magnitude);
		sumX_squared_normalized = sum(Vx.^2./V_magnitude);
		sumY_squared_normalized = sum(Vy.^2./V_magnitude);
		den = sumXY_normalized^2 - sumX_squared_normalized*sumY_squared_normalized;
		xr = -(sumX*sumY_squared_normalized-sumY*sumXY_normalized) / den;
		yr = (sumX*sumXY_normalized-sumY*sumX_squared_normalized) / den;
		Optical_flow_over_time(i,j,:) = [xr,yr];
    end
    
  end
end
tclock3 = clock;
time2 = etime(tclock3, tclock2);
% fprintf ('\tElapsed time: %.2f + %.2f = %.2f [sec]\n', time1, time2, time1+time2);

%Assign vx & vy:
vx_mat = Optical_flow_over_time(:,:,1);
vy_mat = Optical_flow_over_time(:,:,2);



