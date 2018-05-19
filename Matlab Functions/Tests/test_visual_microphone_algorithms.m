%test visual microphone algorithms:

%Initialize parameters:
image_file_name = 'barbara.tif';
low_cutoff_frequency = 200;
high_cutoff_frequency = 2000;
Fs = 5000;
number_of_seconds_to_check = 1;
phase_magnification_factor = 15;

%Get frame parameters: 
mat_in = imread(image_file_name);
[frame_height,frame_width,number_of_channels] = size(mat_in);
dyadic_partition_height = get_dyadic_partition_of_nondyadic_signals(frame_height);
dyadic_partition_width = get_dyadic_partition_of_nondyadic_signals(frame_width);
dyadic_size_vecs_cell = cell(number_of_levels,1);
number_of_scales = 4; %number of scales to actually use

%Tmpoeral Filter parameters:
low_cutoff_frequency_normalized = low_cutoff_frequency/(Fs/2);
high_cutoff_Frequency_normalized = high_cutoff_frequency(Fs/2);
%(1). FIR:
temporal_filter_function = @FIRWindowBP; %SWITCH!
%(2). IIR (IMPLEMENT FUNCTION):
temporal_filter_order = 1;
[B,A] = get_butterworth_filter_coefficients(temporal_filter_order,low_cutoff_frequency_normalized,high_cutoff_frequency_normalized);
for scale_counter = 1:number_of_levels
    %Initialize current value of quaternionic phase. each coefficient has a
    %two element quaternionic phase that is defined as phaseX(cos(orientation),sin(orientation)):
    dyadic_size_vecs_cell{k} = [dyadic_partition_width,dyadic_partition_height];
    phase_cos{k} = zeros(dyadic_size_vecs_cell{k});
    phase_sin{k} = zeros(dyadic_size_vecs_cell{k});
    %Initializes IIR temporal filter values. these values are used during temporal filtering.
    register0_cos{k} = zeros(dyadic_size_vecs_cell{k});
    register0_sin{k} = zeros(dyadic_size_vecs_cell{k});
    register1_cos{k} = zeros(dyadic_size_vecs_cell{k});
    register1_sin{k} = zeros(dyadic_size_vecs_cell{k});
end

%Get sound parameters:
number_of_samples = round(Fs*number_of_seconds_to_check);
flag_sine_or_audio = 1;
if flag_sine_or_audio == 1
    fc = 200;
    t_vec = my_linspace(0,1/Fs,number_of_samples);
    audio_vec = sin(2*pi*fc*t_vec);
elseif flag_sine_or_audio == 2
    audio_file_name = 'bla.wav';
    [audio_vec,Fs_audio] = wavread(audio_file_name);
    audio_vec = resample(audio_vec,Fs,Fs_audio);
end


%% Get Complex Steerable Pyramid parameters:
pyramid_types_strings = {'octave','halfOctave','smoothHalfOctave','quarterOctave'};
pyramid_type_chose_string = 'octave';
flag_attenuate_other_frequencies = false;
smoothing_filter_sigma = 1.5; %seems much

%Get Complex Steerable Pyramid Filters:
complex_steerable_pyramid_height = floor(log2(min(frame_height,frame_width))); %ALL SCALES, CHANGE THIS
pyramid_height = get_max_complex_steerable_pyramid_height(zeros(frame_height,frame_width));
switch pyramid_type
    case 'octave'
        complex_steerable_pyramid_filters_fourier = ...
            get_complex_steerable_pyramid_filters([frame_height frame_width], 2.^[0:-1:-pyramid_height], 4);
    case 'halfOctave'
        complex_steerable_pyramid_filters_fourier = ...
            get_complex_steerable_pyramid_filters([frame_height frame_width], 2.^[0:-0.5:-pyramid_height], 8,'twidth', 0.75);
    case 'smoothHalfOctave'
        complex_steerable_pyramid_filters_fourier = ...
            get_complex_steerable_pyramid_filters_smooth([frame_height frame_width], 8, 'filtersPerOctave', 2);
    case 'quarterOctave'
        complex_steerable_pyramid_filters_fourier = ...
            get_complex_steerable_pyramid_filters_smooth([frame_height frame_width], 8, 'filtersPerOctave', 4);
    otherwise
        error('Invalid Filter Types');
end
[cropped_complex_steerable_pyramid_filters,filters_non_zero_indices_cell_mat] = ...
                        get_filters_and_indices_which_are_non_zero(complex_steerable_pyramid_filters_fourier);

%For easier reading define filters_non_zero_indices:
number_of_levels = numel(complex_steerable_pyramid_filters_fourier);
for scale_counter = 1:number_of_levels
    filters_non_zero_indices{k} = [filters_non_zero_indices_cell_mat{k,1},filters_non_zero_indices_cell_mat{k,2}];
end
                    
%Define build_level and reconstruct_level function (CHANGE cropped_filters and functions to be WITHOUT fftshift):
build_level_function = @(mat_in_fft,k) ifft2(ifftshift(cropped_complex_steerable_pyramid_filters{k} .* mat_in_fft(filters_non_zero_indices{k})));
reconstruct_level_function = @(mat_in_fft,k) 2*(cropped_complex_steerable_pyramid_filters{k}.*fftshift(fft2(mat_in_fft)));


%% Get Riesz Pyramid Parameters:
%%%%%% CHOOSE WAY TO IMPLEMENT THE RIESZ TRANSFORM %%%%% :
%(1). get image power spectrum to construct an ed-hok wavelet transform:
flag_use_real_space_or_fourier_space_convolution = 1; %1 or 2
max_real_space_support_for_fourier_defined_filters = 7; %[pixels]
mat_in_fft = fft2(mat_in);
mat_in_power_spectrum = abs(mat_in_fft).^2;
[ed_hok_riesz_wavelet_lowpass_impulse_response , ed_hok_riesz_wavelet_highpass_impulse_response] = ...
                                    get_ed_hok_time_domain_riesz_wavelet_filters(mat_in_power_spectrum);


%(2). use Rubinstein's riesz pyramid transform:
[ filter_1D_lowpass_coefficients_rubinstein, ...
  filter_1D_highpass_coefficients_rubinstein, ...
  chebychev_polynomial_lowpass_rubinstein, ...
  chebychev_polynomial_highpass_rubinstein, ...
  McClellan_transform_matrix_rubinstein, ...
  filter_2D_lowpass_direct_rubinstein, ...
  filter_2D_highpass_direct_rubinstein ] = get_filters_for_riesz_pyramid();
filter_2D_lowpass_direct_rubinstein_fft = fft2(filter_2D_lowpass_direct_rubinstein);
filter_2D_highpass_direct_rubinstein_fft = fft2(filter_2D_highpass_direct_rubinstein);

%(3). use Generalized Riesz-Transform stuff:
isotropic_wavelet_types = {'simoncelli','shannon','aldroubi','papadakis','meyer','ward'};
prefilter_types = {'shannon','simoncelli','none'};
riesz_transform_order = 1;
riesz_transform_configurations_object = riesz_transform_object(size(mat_in), 1, number_of_scales, 1);
wavelet_transform_configurations_object = riesz_transform_object(size(mat_in), 0, number_of_scales, 1);
flag_restrict_angle_values = 0;
number_of_riesz_channels = riesz_transform_configurations_object.number_of_riesz_channels;
flag_use_perfilter = 1; %1 use prefilter, 0 don't
%%%% 
isotropic_wavelet_type = riesz_transform_configurations_object.isotropic_wavelet_type;
wavelet_type = riesz_transform_configurations_object.wavelet_type;
isotropic_wavelet_type = 'simoncelli';
wavelet_type = 'isotropic'; %As opposed to 'spline', read paper to see why better
prefilter_type = 'simoncelli';
%%%%
mat_in_number_of_dimensions = 2;
flag_coefficients_in_fourier_0_or_spatial_domain_1 = 1;
flag_downsample_or_not = 1;
riesz_transform_fourier_filter_x = riesz_transform_configurations_object.riesz_transform_filters{1};
riesz_transform_fourier_filter_y = riesz_transform_configurations_object.riesz_transform_filters{2};
riesz_transform_prefilter_lowpass = riesz_transform_configurations_object.prefilter.filterLow;
riesz_transform_prefilter_highpass = riesz_transform_configurations_object.prefilter.filterHigh;


%COMPUTE NORMALIZATION COEFFICIENTS:
%(*)compute riesz-wavelet coefficients for normalization:
%** ADD THIS CALCULATION IN THE ABOVE CALCULATION OF THE REFERENCE CELL ARRAY
delta_image = zeros(mat_in_size);
delta_image(1) = 1;
delta_image_fft = fft2(delta_image_fft);
noise_riesz_wavelet_cell_array = cell(1, number_of_scales+1);
noise_riesz_wavelet_cell_array = multiscale_riesz_analysis(delta_image, riesz_transform_configurations_object);
stdNoiseRiesz = ones(length(noise_riesz_wavelet_cell_array), 1);
for scale_counter = 1:length(noise_riesz_wavelet_cell_array)
    tmp = noise_riesz_wavelet_cell_array{scale_counter}(:,:,1); %normalization only acording to first channel
    stdNoiseRiesz(scale_counter) = std(tmp(:));
end
%(*)compute wavelet coefficients for normalization:
noise_wavelet_cell_array = multiscale_riesz_analysis(delta_image, wavelet_transform_configurations_object);
stdNoiseWav = ones(length(noise_wavelet_cell_array), 1);
for scale_counter = 1:length(noise_wavelet_cell_array)
    stdNoiseWav(scale_counter) = std(noise_wavelet_cell_array{scale_counter}(:));
end


%PRE-CALCULATE LOWPASS AND HIGHPASS MASKS:
%the reason it was the in the downsampling loop is that the loop is
%intended for fourier space filtering, and so for each mat size a different
%mask is needed to be created. 
rows_dyadic_partition = get_dyadic_partition_of_nondyadic_signals(size(mat_in,1));
columns_dyadic_partition = get_dyadic_partition_of_nondyadic_signals(size(mat_in,2));
maskHP_fft = cell(1,number_of_scales);
maskLP_fft = cell(1,number_of_scales);
maskHP = cell(1,number_of_scales);
maskLP = cell(1,number_of_scales);
for scale_counter = 1:number_of_scales
    switch isotropic_wavelet_type,
        case IsotropicWaveletType.Meyer,
            [maskHP_fft{scale_counter} , maskLP_fft{scale_counter}] =  meyerMask(rows_dyadic_partition(scale_counter), columns_dyadic_partition(scale_counter), 2);
        case IsotropicWaveletType.Simoncelli,
            [maskHP_fft{scale_counter} , maskLP_fft{scale_counter}] =  simoncelliMask(rows_dyadic_partition(scale_counter), columns_dyadic_partition(scale_counter), 2);
        case IsotropicWaveletType.Papadakis,
            [maskHP_fft{scale_counter} , maskLP_fft{scale_counter}] =  papadakisMask(rows_dyadic_partition(scale_counter), columns_dyadic_partition(scale_counter), 2);
        case IsotropicWaveletType.Aldroubi,
            [maskHP_fft{scale_counter} , maskLP_fft{scale_counter}] =  aldroubiMask(rows_dyadic_partition(scale_counter), columns_dyadic_partition(scale_counter), 2);
        case IsotropicWaveletType.Shannon,
            [maskHP_fft{scale_counter} , maskLP_fft{scale_counter}] =  halfSizeEllipsoidalMask(rows_dyadic_partition(scale_counter), columns_dyadic_partition(scale_counter), 2);
        case IsotropicWaveletType.Ward,
            error('Wards wavelet function is not provided in this toolbox');
        otherwise
            error('unknown wavelet type. Valid options are: meyer, simoncelli, papadakis, aldroubi, shannon')
    end
    %IFFT to get real space impulse response:
    maskHP{scale_counter} = ifft2(maskHP_fft{scale_counter});
    maskLP{scale_counter} = ifft2(maskLP_fft{scale_counter});
end

%(1).get rotation angles using riesz-type structure tensor and using regular method
%(*) compute Riesz-wavelet coefficients:

%(A). PREFILTER IMAGES:
mat_in_fft = fft2(mat_in);
if ~isempty(riesz_transform_configurations_object.prefilter.filterLow),
    mat_in_prefiltered = ifft2(mat_in_fft .* riesz_transform_prefilter_lowpass);
    mat_in_prefiltered_highpassed = ifft2(mat_in_fft .* riesz_transform_prefilter_highpass);
    delta_image_prefiltered = ifft2(delta_image_fft .* riesz_transform_prefilter_lowpass);
    delta_image_prefiltered_highpassed = ifft2(delta_image_fft .* riesz_transform_prefilter_highpass); 
else
    mat_in_prefiltered = mat_in;
    mat_in_prefiltered_highpassed = zeros(size(mat_in));
    delta_image_prefiltered = delta_image;
    delta_image_prefiltered_highpassed = zeros(size(delta_image));
end
mat_in_prefiltered_fft = fft2(mat_in_prefiltered);
delta_image_prefiltered_fft = fft2(delta_image_prefiltered);


%(B). PERFORM RIESZ-TRANSFORM TO CURRENT IMAGE:
%(1). to mat_in:
%fourier space:
mat_in_riesz_coefficients_fft_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
mat_in_riesz_coefficients_fft_3D(:,:,1) = mat_in_prefiltered_fft.*riesz_transform_fourier_filter_x;
mat_in_riesz_coefficients_fft_3D(:,:,2) = mat_in_prefiltered_fft.*riesz_transform_fourier_filter_x;
%direct space:
mat_in_riesz_coefficients_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
mat_in_riesz_coefficients_3D(:,:,1) = real(ifft2(mat_in_riesz_coefficients_fft_3D(:,:,1)));
mat_in_riesz_coefficients_3D(:,:,2) = real(ifft2(mat_in_riesz_coefficients_fft_3D(:,:,2)));
%(2). to delta_image:
%fourier space:
delta_image_riesz_coefficients_fft_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
delta_image_riesz_coefficients_fft_3D(:,:,1) = delta_image_prefiltered_fft.*riesz_transform_fourier_filter_x;
delta_image_riesz_coefficients_fft_3D(:,:,2) = delta_image_prefiltered_fft.*riesz_transform_fourier_filter_x;
%direct space:
delta_image_riesz_coefficients_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
delta_image_riesz_coefficients_3D(:,:,1) = real(ifft2(delta_image_riesz_coefficients_fft_3D(:,:,1)));
delta_image_riesz_coefficients_3D(:,:,2) = real(ifft2(delta_image_riesz_coefficients_fft_3D(:,:,2)));

 
%(C). APPLY WAVELET DECOMPOSITION TO RIESZ COEFFICIENTS
mat_in_riesz_wavelet_cell_array_reference = cell(1, number_of_scales+1);
delta_image_riesz_wavelet_cell_array_reference = cell(1, number_of_scales+1);
for riesz_channel_counter = 1:number_of_riesz_channels,
    %Compute fft2 because before we used ifft2 in riesz_coefficients_matrix_of_lowpassed_image_3D calculation:
    highpass_wavelet_coefficients = cell(1, number_of_scales);
    for scale_counter = 1:number_of_scales
        
        %Lowpass and Highpass (add highpass to wavelet cell array and use lowpass for coarser scales):
        if flag_use_real_space_or_fourier_space_convolution == 2
            %(1) mat in riesz:
            mat_in_riesz_coefficients_fft_3D_HP = mat_in_riesz_coefficients_fft_3D.*maskHP_fft{scale_counter};
            mat_in_riesz_coefficients_fft_3D = mat_in_riesz_coefficients_fft_3D.*maskLP_fft{scale_counter};
            %(2) delta image riesz-wavelet:
            delta_image_riesz_coefficients_fft_3D_HP = delta_image_riesz_coefficients_fft_3D.*maskHP_fft{scale_counter};
            delta_image_riesz_coefficients_fft_3D = delta_image_riesz_coefficients_fft_3D.*maskLP_fft{scale_counter};            
        elseif flag_use_real_space_or_fourier_space_convolution == 1
            %SEE IF CONV2 CAN DO 2D CONVOLUTION FOR EACH MATRIX ALONG THE THIRD DIMENSION:
            %(1)
            mat_in_riesz_coefficients_3D_HP(:,:,1) = conv2(mat_in_riesz_coefficients_3D(:,:,1),maskHP{scale_counter}(:,:,1));
            mat_in_riesz_coefficients_3D(:,:,1) = conv2(mat_in_riesz_coefficients_3D(:,:,2),maskLP{scale_counter}(:,:,1));
            mat_in_riesz_coefficients_3D_HP(:,:,2) = conv2(mat_in_riesz_coefficients_3D(:,:,1),maskHP{scale_counter}(:,:,2));
            mat_in_riesz_coefficients_3D(:,:,2) = conv2(mat_in_riesz_coefficients_3D(:,:,2),maskLP{scale_counter}(:,:,2));
            %(2)
            delta_image_riesz_coefficients_3D_HP(:,:,1) = conv2(delta_image_riesz_coefficients_3D(:,:,1),maskHP{scale_counter}(:,:,1));
            delta_image_riesz_coefficients_3D(:,:,1) = conv2(delta_image_riesz_coefficients_3D(:,:,1),maskLP{scale_counter}(:,:,1));
            delta_image_riesz_coefficients_3D_HP(:,:,2) = conv2(delta_image_riesz_coefficients_3D(:,:,2),maskHP{scale_counter}(:,:,2));
            delta_image_riesz_coefficients_3D(:,:,1) = conv2(delta_image_riesz_coefficients_3D(:,:,2),maskLP{scale_counter}(:,:,2));
        end
        
        %Downsample:
        if flag_downsample_or_not == 1
            if flag_use_real_space_or_fourier_space_convolution == 2
                %fourier space downsampling == spectrum folding:
                %(1). mat in: 
                c2 = size(mat_in_riesz_coefficients_fft_3D,2)/2;
                c1 = size(mat_in_riesz_coefficients_fft_3D,1)/2;
                mat_in_riesz_coefficients_fft_3D = ...
                    0.25*( mat_in_riesz_coefficients_fft_3D(1:c1, 1:c2) + ...
                           mat_in_riesz_coefficients_fft_3D((1:c1)+c1, 1:c2) + ...
                           mat_in_riesz_coefficients_fft_3D((1:c1) + c1, (1:c2) +c2) + ...
                           mat_in_riesz_coefficients_fft_3D(1:c1, (1:c2) +c2)...
                         );
                %(2). delta image:
                delta_image_riesz_coefficients_fft_3D = ...
                    0.25*( delta_image_riesz_coefficients_fft_3D(1:c1, 1:c2) + ...
                           delta_image_riesz_coefficients_fft_3D((1:c1)+c1, 1:c2) + ...
                           delta_image_riesz_coefficients_fft_3D((1:c1) + c1, (1:c2) +c2) + ...
                           delta_image_riesz_coefficients_fft_3D(1:c1, (1:c2) +c2)...
                         ); 
            elseif flag_use_real_space_or_fourier_space_convolution == 1
                %direct space downsampling:
                mat_in_riesz_coefficients_3D = mat_in_riesz_coefficients_3D(1:2:end,1:2:end);
                delta_image_riesz_coefficients_3D = delta_image_riesz_coefficients_3D(1:2:end,1:2:end);
            end
        else
            %do NOT downsample (only filter again and again)
        end
        
        %Assign calculated highpass (detailed) coefficients to wavelet cell array:
        if flag_use_real_space_or_fourier_space_convolution == 2
            mat_in_riesz_wavelet_cell_array_reference{scale_counter} = ifft2(mat_in_riesz_coefficients_fft_3D_HP);
            delta_image_riesz_wavelet_cell_array_reference{scale_counter} = ifft2(delta_image_riesz_coefficients_fft_3D_HP);
        elseif flag_use_real_space_or_fourier_space_convolution == 1
            mat_in_riesz_wavelet_cell_array_reference{scale_counter} = mat_in_riesz_coefficients_3D_HP;
            delta_image_riesz_wavelet_cell_array_reference{scale_counter} = delta_image_riesz_coefficients_3D_HP;
        end

    end %END SCALES LOOP
    
    %Assign the appropriate lowpass residual:
    if flag_use_real_space_or_fourier_space_convolution == 2
        mat_in_riesz_wavelet_cell_array_reference{number_of_scales+1} = ifft2(mat_in_riesz_coefficients_fft_3D);
        delta_image_riesz_wavelet_cell_array_reference{number_of_scales+1} = ifft2(delta_image_riesz_coefficients_fft_3D);
    elseif flag_use_real_space_or_fourier_space_convolution == 1
        mat_in_riesz_wavelet_cell_array_reference{number_of_scales+1} = mat_in_riesz_coefficients_3D_HP;
        delta_image_riesz_wavelet_cell_array_reference{number_of_scales+1} = delta_image_riesz_coefficients_3D;
    end
    
end %riesz channels loop


%(D). GET ROTATION (ORIENTATION) ANGLES USING STRUCTURE TENSOR OR REGULAR METHOD:
flag_use_structure_tensor_or_regular_method_angles = 1; %1 or 2
if flag_use_structure_tensor_or_regular_method_angles == 1
    [rotation_angles_cell_array , coherency_cell_array] = ...
        riesz_monogenic_analysis_of_riesz_coefficients(mat_in_riesz_wavelet_cell_array_reference, ...
                                                       riesz_transform_configurations_object, ...
                                                       smoothing_filter_sigma, ...
                                                       flag_restrict_angle_values, ...
                                                       mat_in);
else
    rotation_angles_cell_array = atan(mat_in_riesz_wavelet_cell_array_reference{:}(:,:,2) ...
                                   ./ mat_in_riesz_wavelet_cell_array_reference{:}(:,:,1));
end


%(E). ROTATE RIESZ COEFFICIENTS:
flag_use_generalized_transform_method_simplified_method_or_none = 1; %1 or 2
rotated_riesz_wavelet_cell_array_reference = mat_in_riesz_wavelet_cell_array_reference;
if flag_use_generalized_transform_method_simplified_method_or_none == 1
    %(1). using generalized-riesz-transform toolbox notation
    for scale_counter = 1:length(rotation_angles_cell_array)
        %Get current scale coefficients:
        current_scale_3D_coefficients_vecotrized = ...
            reshape(mat_in_riesz_wavelet_cell_array_reference{scale_counter}, ...
            size(mat_in_riesz_wavelet_cell_array_reference{scale_counter},1) * size(mat_in_riesz_wavelet_cell_array_reference{scale_counter},2), ...
            size(mat_in_riesz_wavelet_cell_array_reference{scale_counter},3));
        %Get current scale rotation angles:
        current_scale_rotation_angles_vectorized = rotation_angles_cell_array{scale_counter}(:);
        %Get current scale rotation angles matrices:
        S = riesz_compute_multiple_rotation_matrices_for_2D_riesz_transform(current_scale_rotation_angles_vectorized, riesz_transform_order);
        for sample_counter = 1:size(current_scale_rotation_angles_vectorized, 1)
            current_scale_3D_coefficients_vecotrized(sample_counter, :) = ...
                (S(:,:, sample_counter)*current_scale_3D_coefficients_vecotrized(sample_counter, :)')';
        end
        %Reshape rotated vectors (1 vec per riesz channel) and turn then back to matrices:
        rotated_riesz_wavelet_cell_array_reference{scale_counter} = ...
            reshape(current_scale_3D_coefficients_vecotrized, size(mat_in_riesz_wavelet_cell_array_reference{scale_counter}, 1), ...
            size(mat_in_riesz_wavelet_cell_array_reference{scale_counter}, 2), ...
            size(mat_in_riesz_wavelet_cell_array_reference{scale_counter}, 3));
    end
elseif flag_use_generalized_transform_method_simplified_method_or_none == 2
    %(2). using what i understand as the simplified operation for order=1 transform:
    for scale_counter = 1:length(rotation_angles_cell_array)
        rotated_riesz_wavelet_cell_array_reference{scale_counter} = ...
            cos(rotation_angles_cell_array{scale_counter}).*mat_in_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1) ...
            + sin(rotation_angles_cell_array{scale_coutner}).*mat_in_riesz_wavelet_cell_array_reference{scale_counter}(:,:,2);
    end
else
   %(3). do not rotate:
end


%(F). COMPUTE WAVELET COEFFICIENTS:             
% mat_in_prefiltered_fft = fft2(mat_in); %THIS IS THE ORIGINAL AS FOUND IN riesz_full_monogenic_analysis, seems wrong!!!
mat_in_wavelet_cell_array_reference = cell(1, number_of_levels+1);     
delta_image_wavelet_cell_array_reference = cell(1, number_of_levels+1);
for scale_counter = 1:number_of_levels
      
    %high pass and lowpass image:
    if flag_use_real_space_or_fourier_space_convolution == 2
        %(1). mat in
        mat_in_prefiltered_fft_HP = mat_in_prefiltered_fft.*maskHP_fft{scale_counter};
        mat_in_prefiltered_fft = mat_in_prefiltered_fft.*maskLP_fft{scale_counter};
        %(2). delta image:
        delta_image_prefiltered_fft_HP = delta_image_prefiltered_fft.*maskHP_fft{scale_counter};
        delta_image_prefiltered_fft = delta_image_prefiltered_fft.*maskLP_fft{scale_counter};
    else
        %(1). mat in
        mat_in_prefiltered_HP = conv2(mat_in_prefiltered,maskHP_fft{scale_counter});
        mat_in_prefiltered = conv2(mat_in_prefiltered,maskLP_fft{scale_counter});
        %(2). delta image:
        delta_image_prefiltered_HP = conv2(delta_image_prefiltered,maskHP_fft{scale_counter});
        delta_image_prefiltered = conv2(delta_image_prefiltered,maskHP_fft{scale_counter});
    end
    
    %Downsample:
    if flag_downsample_or_not == 1
        if flag_use_real_space_or_fourier_space_convolution == 2
            c2 = size(mat_in_prefiltered_fft,2)/2;
            c1 = size(mat_in_prefiltered_fft,1)/2;
            %(1). mat in:
            mat_in_prefiltered_fft = 0.25*(mat_in_prefiltered_fft(1:c1, 1:c2) + mat_in_prefiltered_fft((1:c1)+c1, 1:c2) + ...
                mat_in_prefiltered_fft((1:c1) + c1, (1:c2) +c2) + mat_in_prefiltered_fft(1:c1, (1:c2) +c2));
            %(2). delta image:
            delta_image_prefiltered_fft = 0.25*(delta_image_prefiltered_fft(1:c1, 1:c2) + delta_image_prefiltered_fft((1:c1)+c1, 1:c2) + ...
                delta_image_prefiltered_fft((1:c1) + c1, (1:c2) +c2) + delta_image_prefiltered_fft(1:c1, (1:c2) +c2)); 
        elseif flag_use_real_space_or_fourier_space_convolution == 1
            %(1). mat in:
            mat_in_prefiltered = mat_in_prefiltered(1:2:end);
            %(2). delta image:
            delta_image_prefiltered = delta_image_prefiltered(1:2:end);
        end
                 
    else
        %do NOT downsample (only filter again and again)
    end
    
    %Assign proper coefficients (direct or fourier space) into the wavelet cell array:
    if flag_use_real_space_or_fourier_space_convolution == 2
        mat_in_wavelet_cell_array_reference{scale_counter} = ifft2(mat_in_prefiltered_fft_HP);
        delta_image_wavelet_cell_array_reference{scale_counter} = ifft2(delta_image_prefiltered_fft_HP);
    else
        mat_in_wavelet_cell_array_reference{scale_counter} = mat_in_prefiltered_HP;
        delta_image_wavelet_cell_array_reference{scale_counter} = delta_image_prefiltered_HP;
    end
    
end %LEVELS/SCALES LOOP
%Assign lowpass residual:
if flag_use_real_space_or_fourier_space_convolution == 2
    mat_in_wavelet_cell_array_reference{number_of_scales+1} = ifft2(mat_in_prefiltered_fft);
    delta_image_wavelet_cell_array_reference{number_of_scales+1} = ifft2(delta_image_prefiltered_fft);
else
    mat_in_wavelet_cell_array_reference{number_of_scales+1} = mat_in_prefiltered;
    delta_image_wavelet_cell_array_reference{number_of_scales+1} = delta_image_prefiltered;
end                                                                                                              


%(G). COMPUTE PHASE AND AMPLITUDE:
phase_cell_array_reference = cell(1, number_of_scales);
amplitude_cell_array_reference = cell(1, number_of_scales);
flag_get_phase_using_rotated_first_component_Rtate_or_regular = 1; %1,2 or 3
flag_amplitude_calculation_method = 1; %1,2 or 3
for scale_counter = 1:number_of_scales,
    R1_rotated = rotated_riesz_wavelet_cell_array_reference{:}(:,:,1);
    R2_rotated = rotated_riesz_wavelet_cell_array_reference{:}(:,:,2);
    R1 = mat_in_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1);
    R2 = mat_in_riesz_wavelet_cell_array_reference{scale_counter}(:,:,2);
    I = mat_in_wavelet_cell_array_reference{scale_counter};
    if flag_get_phase_using_rotated_first_component_or_regular == 1
        phase_cell_array_reference{scale_counter} = ...
            atan( R1_rotated./I...
                         * stdNoiseWav(scale_counter)/stdNoiseRiesz(scale_counter) );
    elseif flag_get_phase_using_rotated_first_component_or_regular == 2
        phase_cell_array_reference{scale_counter} = ...
            atan(sqrt(R1_rotated.^2+R2_rotated.^2)./I ...
                         * stdNoiseWav(scale_counter)/stdNoiseRiesz(scale_counter) );
    elseif flag_get_phase_using_rotated_first_component_or_regular == 3
        phase_cell_array_reference{scale_counter} = ...
            atan(sqrt(R1.^2+R2.^2)./I ...
                         * stdNoiseWav(scale_counter)/stdNoiseRiesz(scale_counter) );
    end
                                
    %HOW COME THE AMPLITUDE EXPRESSION DOESN'T INCLUDE ALL THREE TERMS AS BELOW????:
    %amplitude{j} = sqrt(QA{j}(:,:,1).^2 + QA{j}(:,:,2).^2 + Q{j}.^2*stdNoiseRiesz{j}(1)/stdNoiseWav(j));
    if flag_amplitude_calculation_method == 1
        amplitude_cell_array_reference{scale_counter} = ...
            sqrt( (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1)/stdNoiseRiesz(scale_counter)).^2 + ...
                  (mat_in_wavelet_cell_array_reference{scale_counter}/stdNoiseWav(scale_counter)).^2 );
    elseif flag_amplitude_calculation_method == 2
        amplitude_cell_array_reference{scale_counter} = ...    
            sqrt( (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1)/stdNoiseRiesz(scale_counter)).^2 + ...
                  (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,2)/stdNoiseRiesz(scale_counter)).^2 + ...
                  (mat_in_wavelet_cell_array_reference{scale_counter}/stdNoiseWav(scale_counter)).^2 );
    elseif flag_amplitude_calculation_method == 3
        amplitude_cell_array_reference{scale_counter} = ...
            sqrt( (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1)).^2 + ...
                  (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,2)).^2 + ...
                  (mat_in_wavelet_cell_array_reference{scale_counter}*stdNoiseRiesz(scale_counter)/stdNoiseWav(scale_counter)).^2 );
    elseif flag_amplitude_calculation_method == 4
        amplitude_cell_array_reference{scale_counter} = ...    
            sqrt( (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1)).^2 + ...
                  (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,2)).^2 + ...
                  (mat_in_wavelet_cell_array_reference{scale_counter}).^2 );
    end
end %END SCALE LOOP
 



 






%(2). rotation angles using regular:
quaternion_vec1 = [sin(phi)*cos(theta) , sin(phi)*sin(theta)];
quaternion_vec2 = [phi*cos(theta) , phi*sin(theta)];


%% PDTDFB decomposition phase:
cfg =  [2 2 2 2 2];
alpha = 0.15;
s = 512;
resi = false;
frequency_windows = get_PDTDFB_frequency_windows(s, alpha, length(cfg), resi);
y = PDTDFB_decomposition_FFT(im, cfg, frequency_windows, alpha, resi);
for scale = 1:length(cfg)
    for dir = 1:2^cfg(scale)
        Y_coef_real = y{scale+1}{1}{dir};
        % imaginary part
        Y_coef_imag = y{scale+1}{2}{dir};
        % Signal variance estimation
        Y_coef = Y_coef_real+1j*Y_coef_imag;
        current_scale_and_direction_phase = atan(Y_coef_imag./Y_coef_real);
    end
end



%% TRY ALGORITHMS:
%**AFTERWARDS SEE HOW TO FILTER SIGNAL, WHETHER TO USE PHASE DIRECTLY OR TO
%  USE THE SUGGESTION IN THE RUBINSTEIN PAPER OF USING
%  [phi*cos(theta),phi*sin(theta)] or [sin(phi)*cos(theta),sin(phi)*sin(theta)]
%noise parameters:
optical_SNR = inf;
mat_in_signal_per_pixel = sum(mat_in(:))/numel(mat_in);
noise_std = mat_in_signal_per_pixel*(1/optical_SNR);

%loop parameters:
number_of_shift_angles = 10;
number_of_noise_instances = 1;
shift_angles = my_linspace(0,pi/2,number_of_shift_angles);
for noise_counter = 1:number_of_noise_instances
    %Initialize noisy mats:
    noise_mat1 = noise_std * randn(size(mat_in));
    noise_mat2 = noise_std * randn(size(mat_in));
    noisy_mat1 = mat_in + noise_mat1;
    noisy_mat2 = mat_in + noise_mat2;
    
    for angle_counter = 1:number_of_shift_angles
        %Shift second mat:
        shift_angle = shift_angles(angle_counter);
        noisy_mat2 = shift_matrix(noist_mat2,1,shift_size*cos(shift_angle),shift_size*sin(shift_angle));
        
        %Crop mats:
        noisy_mat1 = noisy_mat1(1:end-1,1:end-1);
        noisy_mat2 = noisy_mat2(1:end-1,1:end-1);
        
        %% Get phase map
        %(1). 
        
        %
    end
end

















