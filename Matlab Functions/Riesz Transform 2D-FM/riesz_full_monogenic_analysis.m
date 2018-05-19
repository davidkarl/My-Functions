function [rotation_angles_cell_array, ...
          coherency_cell_array, ...
          amplitude_cell_array, ...
          phase_cell_array, ...
          wave_number_cell_array, ...
          mat_in_riesz_wavelet_cell_array, ...
          mat_in_wavelet_cell_array, ...
          riesz_transform_configurations_object] = riesz_full_monogenic_analysis(mat_in, ...
                                                                                 number_of_scales, ...
                                                                                 smoothing_filter_sigma, ...
                                                                                 flag_restrict_angle_values)
% FULLMONOGENICANALYSIS perform a complete multiscale monogenic analysis of an image
% 
% --------------------------------------------------------------------------
% Input arguments:
%
% A an image to analyze
% 
% J the number of scales for the primary wavelet transform
%
% SIGMA regularization parameter. It is the standard deviation of the regularizing
% Gaussian kernel.
%
% FULL Specifies if angles should be restricted to values in [-pi/2,pi/2].
% Optional, default is 0.
%
% --------------------------------------------------------------------------
% Output arguments:
%
% ANG angles estimated pointwise in the wavelet bands. It consists in a cell
% of matrices. Each element of the cell corresponds to the matrix of angles
% for a wavelet band.
%
% COHERENCY coherency values estimated pointwise in the wavelet bands.
% It consists in a cell of matrices. Each element of the cell corresponds to
% the matrix of coherency values for a wavelet band.
%
% AMPLITUDE monogenic amplitude values estimated pointwise in the wavelet bands.
% It consists in a cell of matrices. Each element of the cell corresponds to
% the matrix of amplitude values for a wavelet band.
%
% PHASE monogenic phase values estimated pointwise in the wavelet bands.
% It consists in a cell of matrices. Each element of the cell corresponds to
% the matrix of phase values for a wavelet band.
%
% WAVENUMBER monogenic wave number values estimated pointwise in the wavelet bands.
% It consists in a cell of matrices. Each element of the cell corresponds to
% the matrix of wave nummber values for a wavelet band.
%
% QA Riesz-wavelet coefficients used for the monogenic analysis
%
% QWAV wavelet coefficients used for the monogenic analysis
%
% RIESZCONFIG RieszConfig2D object that specifies the Riesz-wavelet transform.
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

if nargin < 4,
    flag_restrict_angle_values = 0;
end

%configure the Riesz transform of order 1 (Initialization by calling riesz_transform_object constructor):
riesz_transform_configurations_object = riesz_transform_object(size(mat_in), 1, number_of_scales, 1);
wavelet_transform_configurations_object = riesz_transform_object(size(mat_in), 0, number_of_scales, 1);
riesz_transform_order = riesz_transform_configurations_object.riesz_transform_order;
number_of_scales = riesz_transform_configurations_object.number_of_scales;
mat_in_size = riesz_transform_configurations_object.mat_size;

%compute Riesz-wavelet coefficients:
mat_in_riesz_wavelet_cell_array = multiscale_riesz_analysis(mat_in, riesz_transform_configurations_object);

%% monogenic analysis:
[rotation_angles_cell_array , coherency_cell_array] = ...
                 riesz_monogenic_analysis_of_riesz_coefficients(mat_in_riesz_wavelet_cell_array, ...
                                                                riesz_transform_configurations_object, ...
                                                                smoothing_filter_sigma, ...
                                                                flag_restrict_angle_values, ...
                                                                mat_in);

%% rotate Riesz coefficients:
rotated_riesz_wavelet_cell_array = ...
                        riesz_rotate_2D_riesz_coefficients(mat_in_riesz_wavelet_cell_array, ...
                                                            riesz_transform_order, ...
                                                            rotation_angles_cell_array);
%% compute wavelet coefficients:  
mat_in_wavelet_cell_array = wavelet_analysis_riesz_matrix(mat_in, wavelet_transform_configurations_object);

%% compute normalization:
noise = zeros(mat_in_size);
noise(1) = 1;
noise_riesz_wavelet_cell_array = multiscale_riesz_analysis(noise, riesz_transform_configurations_object);
stdNoiseRiesz = ones(length(noise_riesz_wavelet_cell_array), 1);
for scale_counter = 1:length(noise_riesz_wavelet_cell_array)
    tmp = noise_riesz_wavelet_cell_array{scale_counter}(:,:,1);
    stdNoiseRiesz(scale_counter) = std(tmp(:));
end
clear Qnoise

%compute wavelet coefficients for noise:
noise_wavelet_cell_array = multiscale_riesz_analysis(noise, wavelet_transform_configurations_object);
stdNoiseWav = ones(length(noise_wavelet_cell_array), 1);
for scale_counter = 1:length(noise_wavelet_cell_array)
    stdNoiseWav(scale_counter) = std(noise_wavelet_cell_array{scale_counter}(:));
end
clear Qnoise


%% compute phase and amplitude:
phase_cell_array = cell(1, number_of_scales);
amplitude_cell_array = cell(1, number_of_scales);
for scale_counter = 1:number_of_scales,
    %phase{j} = atan(stdNoiseWav(j)*Qr{j}(:,:,1)./(Qwav{j}*stdNoiseRiesz(j)));
    phase_cell_array{scale_counter} = angle(mat_in_wavelet_cell_array{scale_counter}/stdNoiseWav(scale_counter) + ...
                                    1j*rotated_riesz_wavelet_cell_array{scale_counter}(:,:,1)/stdNoiseRiesz(scale_counter));
    
    %amplitude{j} = sqrt(QA{j}(:,:,1).^2 + QA{j}(:,:,2).^2 + Q{j}.^2*stdNoiseRiesz{j}(1)/stdNoiseWav(j));
    amplitude_cell_array{scale_counter} = sqrt((rotated_riesz_wavelet_cell_array{scale_counter}(:,:,1)/stdNoiseRiesz(scale_counter)).^2 + ...
                                    (mat_in_wavelet_cell_array{scale_counter}/stdNoiseWav(scale_counter)).^2);
end


%% compute the wavenumber:
wave_number_cell_array = cell(1, number_of_scales);

%compute the Riesz transform of the gradient image:
mat_in_laplacian = laplacian_in_the_frequency_domain(mat_in); %compute the laplacian of the image
mat_in_laplacian_riesz_wavelet_matrices_cell_array = ...
                        multiscale_riesz_analysis(mat_in_laplacian, riesz_transform_configurations_object);

%compute wavelet coefficients:
mat_in_laplacian_wavelet_cell_array = wavelet_analysis_riesz_matrix(mat_in_laplacian, wavelet_transform_configurations_object);
clear gA;

%form the code of Virginie Ulhmann:
correction_frequency = 1;
for scale_counter = 1:number_of_scales,
    %	<Rf, psi'_i> = w_(x,i) + jw(y,i).
    current_laplacian_riesz = mat_in_laplacian_riesz_wavelet_matrices_cell_array{scale_counter};
    current_laplacian_riesz = current_laplacian_riesz/stdNoiseRiesz(scale_counter); %normalize
    
    current_laplacian_wavelet = mat_in_laplacian_wavelet_cell_array{scale_counter};
    current_laplacian_wavelet = current_laplacian_wavelet/stdNoiseWav(scale_counter);
    
    current_wavelet = mat_in_wavelet_cell_array{scale_counter};
    current_wavelet = current_wavelet/stdNoiseWav(scale_counter);
    
    theta = rotation_angles_cell_array{scale_counter};
    current_riesz = mat_in_riesz_wavelet_cell_array{scale_counter}(:,:,1).*cos(theta) + ...
                          mat_in_riesz_wavelet_cell_array{scale_counter}(:,:,2).*sin(theta);
    current_riesz = current_riesz/stdNoiseRiesz(scale_counter);
    % Numerator of eq. (34) of [1]: -q_i*(w_(x,i)*cos(theta) + w(y_i)*sin(theta)) + w_i*(r_(1x,i) + r_(2y,i)),
    % With the following definitions:
    %		<Rf, psi'_i> = w_(x,i) + jw(y,i),
    %		<f, psi'_i> = -(r_(1x,i) + r(2y_i)),
    % And w_i are the wavelet coefficients.
    nu = current_riesz.*(current_laplacian_riesz(:,:,1).*cos(theta) + current_laplacian_riesz(:,:,2).*sin(theta)) + ...
         current_wavelet.*current_laplacian_wavelet;
    wave_number_cell_array{scale_counter} = (nu * correction_frequency )./(abs(current_wavelet).^2 + abs(current_riesz).^2);
end
end

% laplacian like filtering with frequency response ||w||
function gA = laplacian_in_the_frequency_domain(A)
[sizeY sizeX] = size(A);
gridX = -(floor(sizeX/2)):(floor(sizeX/2)-1);
gridX = 2*pi*repmat(gridX, sizeY, 1)/sizeX;
gridY = -(floor(sizeY/2)):(floor(sizeY/2)-1);
gridY = 2*pi*repmat(gridY', 1, sizeX)/sizeY;
dist = fftshift(sqrt(gridX.^2 + gridY.^2));
gA = ifft2(dist.*fft2(A));
end