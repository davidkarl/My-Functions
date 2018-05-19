function [rotation_angles_cell_array, ...
          coherency_cell_array, ...
          amplitude_cell_array, ...
          phase_cell_array, ...
          wave_number_cell_array] = ...
                            riesz_full_monogenic_analysis_of_riesz_coefficients(riesz_wavelet_cell_array, ...
                                                                                wavelet_cell_array, ...
                                                                                riesz_transform_object, ...
                                                                                smoothing_filter_sigma, ...
                                                                                flag_restrict_angle_values, ...
                                                                                mat_in)
% FULLMONOGENICANALYSISOFRIESZCOEFFS perform a complete multiscale monogenic
% analysis of given sets of Riesz-wavelet and wavelet coefficients
%
% --------------------------------------------------------------------------
% Input arguments: 
%
%
% QA Riesz-wavelet coefficients used for the monogenic analysis
% 
% QWAV wavelet coefficients used for the monogenic analysis
%
% RIESZCONFIG RieszConfig2D object that specifies the Riesz-wavelet
% transform.
%
% SIGMA regularization parameter. It is the standard deviation of the regularizing
% Gaussian kernel.
%
% FULL Specifies if angles should be restricted to values in [-pi/2,pi/2].
% Optional, default is 0.
%
% A an image to analyze. It is required only if full == 1.
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
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

if (nargin <5)
    flag_restrict_angle_values = 0;
end
if ~exist('A')
    mat_in = zeros(size(wavelet_cell_array{1},1), size(wavelet_cell_array{1},2));
end

%% monogenic analysis
[rotation_angles_cell_array , coherency_cell_array] = riesz_monogenic_analysis_of_riesz_coefficients(...
                                                                        riesz_wavelet_cell_array, ...
                                                                        riesz_transform_object, ...
                                                                        smoothing_filter_sigma, ...
                                                                        flag_restrict_angle_values, ...
                                                                        mat_in);

if (nargout>2)
    % compute phase
    %rotate Riesz coefficients
    riesz_wavelet_rotated_cell_array = ...
                        riesz_rotate_2D_riesz_coefficients(riesz_wavelet_cell_array, ...
                                                           riesz_transform_object.riesz_transform_order, ...
                                                           rotation_angles_cell_array);
    mat_in = riesz_prefilter(mat_in,riesz_transform_object); % prefilter image
    wavelet_cell_array = wavelet_analysis_riesz_matrix(mat_in, riesz_transform_object);
    
    %compute normalization
    noise = zeros(riesz_transform_object.mat_size);
    noise(1) = 1;
    Pnoise = riesz_prefilter(noise, riesz_transform_object);
    %compute Riesz-wavelet coefficients for noise
    Qnoise = riesz_transform_analysis(Pnoise, riesz_transform_object); %apply Riesz transform
    for j=1:length(Qnoise)
        tmp = Qnoise{j}(:,:,1);
        stdNoiseRiesz(j) = std(tmp(:));
    end
    clear Qnoise
    %compute wavelet coefficients for noise
    Qnoise = wavelet_analysis_riesz_matrix(Pnoise, riesz_transform_object);
    for j = 1:length(Qnoise)
        stdNoiseWav(j) = std(Qnoise{j}(:));
    end
    clear Qnoise
    
    %compute phase and amplitude
    phase_cell_array = cell(1, riesz_transform_object.number_of_scales);
    amplitude_cell_array = cell(1, riesz_transform_object.number_of_scales);
    for j = 1:riesz_transform_object.number_of_scales
        %phase{j} = atan(stdNoiseWav(j)*Qr{j}(:,:,1)./(Qwav{j}*stdNoiseRiesz(j)));
        phase_cell_array{j} = angle(wavelet_cell_array{j}/stdNoiseWav(j) + ...
                                                1j * riesz_wavelet_rotated_cell_array{j}(:,:,1)/stdNoiseRiesz(j));
        %amplitude{j} = sqrt(QA{j}(:,:,1).^2 + QA{j}(:,:,2).^2 + Q{j}.^2*stdNoiseRiesz{j}(1)/stdNoiseWav(j));
        amplitude_cell_array{j} = sqrt((riesz_wavelet_rotated_cell_array{j}(:,:,1)/stdNoiseRiesz(j)).^2 + ...
                                                        (wavelet_cell_array{j}/stdNoiseWav(j)).^2);
    end
end

%% compute the wavenumber
if (nargout>4)
    wave_number_cell_array = cell(1, riesz_transform_object.number_of_scales);
    
    % compute the Riesz transform of the gradient image
    gA = laplacian(mat_in); %compute the laplacian of the image
    gA = riesz_prefilter(gA,riesz_transform_object); % prefilter image
    QgA = riesz_transform_analysis(gA,riesz_transform_object); %apply Riesz transform
    % compute wavelet coefficients
    QwgA = wavelet_analysis_riesz_matrix(gA, riesz_transform_object);
    clear gA;
    
    %from the code of Virginie Ulhmann
    correctionFreq = 1;
    for j=1:J,
        %	<Rf, psi'_i> = w_(x,i) + jw(y,i).
        w = QgA{j};
        %normalize
        w = w/stdNoiseRiesz(j);
        
        % <f, psi'_i> = -(r_(1x,i) + r(2y_i)).
        r = QwgA{j};
        r = r/stdNoiseWav(j);
        
        p = wavelet_cell_array{j};
        p = p/stdNoiseWav(j);
        
        %q = Qr{j}(:,:,1);
        theta = rotation_angles_cell_array{j};
        q = riesz_wavelet_cell_array{j}(:,:,1).*cos(theta) + riesz_wavelet_cell_array{j}(:,:,2).*sin(theta);
        q = q/stdNoiseRiesz(j);
        % Numerator of eq. (34) of [1]: -q_i*(w_(x,i)*cos(theta) + w(y_i)*sin(theta)) + w_i*(r_(1x,i) + r_(2y,i)),
        % With the following definitions:
        %		<Rf, psi'_i> = w_(x,i) + jw(y,i),
        %		<f, psi'_i> = -(r_(1x,i) + r(2y_i)),
        % And w_i are the wavelet coefficients.
        nu = q.*(w(:,:,1).*cos(theta) + w(:,:,2).*sin(theta)) + p.*r;
        wave_number_cell_array{j} = (nu * correctionFreq )./(abs(p).^2 + abs(q).^2);
    end
end
end