%test visual microphone algorithms:

%Initialize parameters:
image_file_name = 'barbara.tif';
low_cutoff_frequency = 200;
high_cutoff_frequency = 2000;
Fs = 5000;
number_of_seconds_to_check = 1;
phase_magnification_factor = 15;
flag_attenuate_other_frequencies = false;
smoothing_filter_sigma = 1.5; %seems much

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

                
%Define build_level and reconstruct_level function (CHANGE cropped_filters and functions to be WITHOUT fftshift):
build_level_function = @(mat_in_fft,k) ifft2(ifftshift(cropped_complex_steerable_pyramid_filters{k} .* mat_in_fft(filters_non_zero_indices{k})));
reconstruct_level_function = @(mat_in_fft,k) 2*(cropped_complex_steerable_pyramid_filters{k}.*fftshift(fft2(mat_in_fft)));


%% Get Riesz Pyramid Parameters:
%%%%%% CHOOSE WAY TO IMPLEMENT THE RIESZ TRANSFORM %%%%% :
%(1). get image power spectrum to construct an ed-hok wavelet transform:
flag_use_real_space_or_fourier_space_convolution = 1; %1 or 2
max_real_space_support_for_fourier_defined_filters = 4; %[pixels] ONE SIDED (total size = 2*support+1)
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
filter_2D_lowpass_direct_rubinstein_fft = fft2(filter_2D_lowpass_direct_rubinstein,frame_height,frame_width);
filter_2D_highpass_direct_rubinstein_fft = fft2(filter_2D_highpass_direct_rubinstein,frame_height,frame_width);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(3). use Generalized Riesz-Transform stuff:
isotropic_wavelet_types = {'simoncelli','shannon','aldroubi','papadakis','meyer','ward',...
                           'radial_rubinstein', 'radial_rubinstein_smoothed'};
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%COMPUTE NORMALIZATION COEFFICIENTS:
%(*)compute riesz-wavelet coefficients for normalization:
%** ADD THIS CALCULATION IN THE ABOVE CALCULATION OF THE REFERENCE CELL ARRAY
delta_image = zeros(mat_in_size);
delta_image(1) = 1;
delta_image_fft = fft2(delta_image_fft);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    current_frame_height = rows_dyadic_partition(scale_counter);
    current_frame_width = columns_dyadic_partition(scale_counter);
    current_image_size = [current_frame_height,current_frame_width];
    switch isotropic_wavelet_type,
        case 'meyer' 
            [maskHP_fft{scale_counter} , maskLP_fft{scale_counter}] =  meyerMask(current_frame_height,current_frame_width, 2);
        case 'simoncelli',
            [maskHP_fft{scale_counter} , maskLP_fft{scale_counter}] =  simoncelliMask(current_frame_height,current_frame_width, 2);
        case 'papadakis',
            [maskHP_fft{scale_counter} , maskLP_fft{scale_counter}] =  papadakisMask(current_frame_height,current_frame_width, 2);
        case 'aldroubi',
            [maskHP_fft{scale_counter} , maskLP_fft{scale_counter}] =  aldroubiMask(current_frame_height,current_frame_width, 2);
        case 'shannon',
            [maskHP_fft{scale_counter} , maskLP_fft{scale_counter}] =  halfSizeEllipsoidalMask(current_frame_height,current_frame_width, 2);
        case 'ward',
            error('Wards wavelet function is not provided in this toolbox');
        case 'compact_rubinstein'
            maskHP{scale_counter} = filter_2D_highpass_direct_rubinstein;
            maskLP{scale_Counter} = filter_2D_lowpass_direct_rubinstein;
            maskHP_fft{scale_counter} = fft2(filter_2D_highpass_direct_rubinstein,current_frame_height,current_frame_width);
            maskLP_fft{scale_counter} = fft2(filter_2D_lowpass_direct_rubinstein,current_frame_height,current_frame_width);
        case 'radial_rubinstein'
            boundary_radiuses_between_adjacent_filters = 0.5; %half-band filter
            number_of_orientations = 8;
            transition_width = 0.75;
            [angle, log_rad] = get_polar_meshgrid([current_frame_height,current_frame_width]); 
            [maskHP_fft{scale_counter}, maskLP_fft{scale_counter}] = ...
                get_radial_mask_pair(boundary_radiuses_between_adjacent_filters, log_rad, transition_width);
            maskHP{scale_counter} = ifft2(maskHP_fft{scale_counter});
            maskLP{scale_counter} = ifft2(maskLP_fft{scale_counter});
            maskHP_fft{scale_counter} = fftshift(maskHP_fft{scale_counter});
            maskLP_fft{scale_counter} = fftshift(maskLP_fft{scale_counter});
        case 'radial_rubinstein_smoothed' %doesn't look like half-band filters!!!! 
            flag_complex_filter = true; % If true, only return filters in one half plane.
            cosine_order = 6;
            filters_per_octave = 6;
            pyramid_height = get_max_complex_steerable_pyramid_height(zeros([current_frame_height,current_frame_width]));
            [angle_meshgrid, radius_meshgrid] = get_polar_meshgrid([current_frame_height,current_frame_width]);
            radius_meshgrid = (log2(radius_meshgrid));
            radius_meshgrid = (pyramid_height+radius_meshgrid)/pyramid_height;
            number_of_filters = filters_per_octave*pyramid_height;
            radius_meshgrid = radius_meshgrid*(pi/2+pi/7*number_of_filters);
            angular_windowing_function = @(x, center) abs(x-center)<pi/2;
            radial_filters = {};
            count = 1;
            filters_squared_sum = zeros([current_frame_height,current_frame_width]);
            const = (2^(2*cosine_order))*(factorial(cosine_order)^2)/((cosine_order+1)*factorial(2*cosine_order));
            for k = number_of_filters:-1:1
                shift = pi/(cosine_order+1)*k+2*pi/7;
                radial_filters{count} = ...
                    sqrt(const)*cos(radius_meshgrid-shift).^cosine_order .* angular_windowing_function(radius_meshgrid,shift);
                filters_squared_sum = filters_squared_sum + radial_filters{count}.^2;
                count = count + 1;
            end
            %Compute lopass residual:
            center = ceil((current_image_size+0.5)/2);
            lodims = ceil((center+0.5)/4);
            %We crop the sum image so we don't also compute the high pass:
            filters_squared_sum_cropped = filters_squared_sum(center(1)-lodims(1):center(1)+lodims(1) , ...
                                                              center(2)-lodims(2):center(2)+lodims(2));
            low_pass = zeros(image_size);
            low_pass(center(1)-lodims(1):center(1)+lodims(1),center(2)-lodims(2):center(2)+lodims(2)) = ...
                                                                   abs(sqrt(1-filters_squared_sum_cropped));
            %Compute high pass residual:
            filters_squared_sum = filters_squared_sum + low_pass.^2;
            high_pass = abs(sqrt(1-filters_squared_sum));
            %If either dimension is even, this fixes some errors (UNDERSTAND WHY!!!):
            number_of_radial_filters = numel(radial_filters);
            if (mod(image_size(1),2) == 0) %even
                for k = 1:number_of_radial_filters
                    temp = radial_filters{k};
                    temp(1,:) = 0;
                    radial_filters{k} = temp;
                end
                high_pass(1,:) = 1;
                low_pass(1,:) = 0;
            end
            if (mod(image_size(2),2) == 0)
                for k = 1:number_of_radial_filters
                    temp = radial_filters{k};
                    temp(:,1) = 0;
                    radial_filters{k} = temp;
                end
                high_pass(:,1) = 1;
                low_pass(:,1) = 0;
            end
            maskLP_fft{scale_counter} = fftshift(low_pass);
            maskHP_fft{scale_counter} = fftshift(high_pass);
        otherwise
            error('unknown wavelet type. Valid options are: meyer, simoncelli, papadakis, aldroubi, shannon')
    end
    
    if ~strcmp(isotropic_wavelet_type,'compact_rubinstein')
        %IFFT to get real space impulse response:
        temp_highpass = ifft2(maskHP_fft{scale_counter});
        temp_lowpass = ifft2(maskLP_fft{scale_counter});
        proper_indices_rows = current_frame_height/2+1-max_real_space_support_for_fourier_defined_filters : current_frame_height/2+1+max_real_space_support_for_fourier_defined_filters;
        proper_indices_columns = current_frame_width/2+1-max_real_space_support_for_fourier_defined_filters : current_frame_width/2+1+max_real_space_support_for_fourier_defined_filters;
        maskHP{scale_counter} = temp_highpass(proper_indices_rows,proper_indices_columns);
        maskLP{scale_counter} = temp_lowpass(proper_indices_rows,proper_indices_columns);
    end
end %end of scale counter loop


%Create double (3D) copies of the masks to allow efficient multiplying of the 3D riesz coefficients:
maskHP_3D = cell(1,number_of_scales);
maskLP_3D = cell(1,number_of_scales);
maskHP_3D_fft = cell(1,number_of_scales);
maskLP_3D_fft = cell(1,number_of_scales);
for scale_counter = 1:number_of_scales
    maskHP_3D{scale_counter} = repmat(maskHP{scale_counter},1,1,2);
    maskLP_3D{sclae_counter} = repmat(maskLP{scale_counter},1,1,2);
    maskHP_3D_fft{scale_counter} = repmat(maskHP_fft{scale_counter},1,1,2);
    maskLP_3D_fft{scale_counter} = repmat(maskLP_fft{scale_counter},1,1,2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(*) compute Riesz-wavelet coefficients:
    % PERHAPSE LOOK INTO AN AVERAGED IMAGE POWER SPECTRUM AND DESIGN A FILTER
    % WHICH TRIES TO DOWNPLAY NOISY FREQUENCIES.
    % OR MAYBE CONTINUE SOMEHOW THE LINE OF THOUGHT OF USING LOTS OF LOCAL
    % FILTERS AND WEIGH THEM APPROPRIATELY.
%(A). PREFILTEROF DIFFERENT  IMAGES:
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(B). PERFORM RIESZ-TRANSFORM TO CURRENT (PREFILTERED) IMAGE:
%(1). to mat_in:
%fourier space:
mat_in_riesz_coefficients_fft_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
mat_in_riesz_coefficients_fft_3D(:,:,1) = mat_in_prefiltered_fft.*riesz_transform_fourier_filter_x;
mat_in_riesz_coefficients_fft_3D(:,:,2) = mat_in_prefiltered_fft.*riesz_transform_fourier_filter_y;
%direct space:
mat_in_riesz_coefficients_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
mat_in_riesz_coefficients_3D(:,:,1) = real(ifft2(mat_in_riesz_coefficients_fft_3D(:,:,1)));
mat_in_riesz_coefficients_3D(:,:,2) = real(ifft2(mat_in_riesz_coefficients_fft_3D(:,:,2)));
%(2). to delta_image:
%fourier space:
delta_image_riesz_coefficients_fft_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
delta_image_riesz_coefficients_fft_3D(:,:,1) = delta_image_prefiltered_fft.*riesz_transform_fourier_filter_x;
delta_image_riesz_coefficients_fft_3D(:,:,2) = delta_image_prefiltered_fft.*riesz_transform_fourier_filter_y;
%direct space:
delta_image_riesz_coefficients_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
delta_image_riesz_coefficients_3D(:,:,1) = real(ifft2(delta_image_riesz_coefficients_fft_3D(:,:,1)));
delta_image_riesz_coefficients_3D(:,:,2) = real(ifft2(delta_image_riesz_coefficients_fft_3D(:,:,2)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(C). APPLY WAVELET DECOMPOSITION TO RIESZ COEFFICIENTS
mat_in_riesz_wavelet_cell_array_reference = cell(1, number_of_scales+1);
delta_image_riesz_wavelet_cell_array_reference = cell(1, number_of_scales+1);
mat_in_riesz_wavelet_lowpass_cell_array_reference = cell(1,number_of_scales+2);
delta_image_riesz_wavelet_lowpass_cell_array_reference = cell(1,number_of_scales+2);
%(*) Assign raw prefiltered image to first cell of appropriate cell array:
mat_in_riesz_wavelet_lowpass_cell_array_reference{1} = mat_in_riesz_coefficients_3D;
delta_image_riesz_wavelet_lowpass_cell_array_reference{1} = delta_image_riesz_coefficients_3D;
for riesz_channel_counter = 1:number_of_riesz_channels,
    %Compute fft2 because before we used ifft2 in riesz_coefficients_matrix_of_lowpassed_image_3D calculation:
    highpass_wavelet_coefficients = cell(1, number_of_scales);
    for scale_counter = 1:number_of_scales
        
        %Lowpass and Highpass (add highpass to wavelet cell array and use lowpass for coarser scales):
        if flag_use_real_space_or_fourier_space_convolution == 2
            %(1) mat_in riesz-wavelet:
            mat_in_riesz_coefficients_fft_3D_HP = mat_in_riesz_coefficients_fft_3D.*maskHP_3D_fft{scale_counter};
            mat_in_riesz_coefficients_fft_3D = mat_in_riesz_coefficients_fft_3D.*maskLP_3D_fft{scale_counter};
            %(2) delta_image riesz-wavelet:
            delta_image_riesz_coefficients_fft_3D_HP = delta_image_riesz_coefficients_fft_3D.*maskHP_3D_fft{scale_counter};
            delta_image_riesz_coefficients_fft_3D = delta_image_riesz_coefficients_fft_3D.*maskLP_3D_fft{scale_counter};            
        elseif flag_use_real_space_or_fourier_space_convolution == 1
            
            %SEE IF CONV2 CAN DO 2D CONVOLUTION FOR EACH MATRIX ALONG THE THIRD DIMENSION:
            %(1) mat_in riesz-wavelet:
            mat_in_riesz_coefficients_3D_HP(:,:,1) = conv2(mat_in_riesz_coefficients_3D(:,:,1),maskHP{scale_counter});
            mat_in_riesz_coefficients_3D(:,:,1) = conv2(mat_in_riesz_coefficients_3D(:,:,1),maskLP{scale_counter});
            mat_in_riesz_coefficients_3D_HP(:,:,2) = conv2(mat_in_riesz_coefficients_3D(:,:,2),maskHP{scale_counter});
            mat_in_riesz_coefficients_3D(:,:,2) = conv2(mat_in_riesz_coefficients_3D(:,:,2),maskLP{scale_counter});
            %(2) delta_image riesz-wavelet:
            delta_image_riesz_coefficients_3D_HP(:,:,1) = conv2(delta_image_riesz_coefficients_3D(:,:,1),maskHP{scale_counter});
            delta_image_riesz_coefficients_3D(:,:,1) = conv2(delta_image_riesz_coefficients_3D(:,:,1),maskLP{scale_counter});
            delta_image_riesz_coefficients_3D_HP(:,:,2) = conv2(delta_image_riesz_coefficients_3D(:,:,2),maskHP{scale_counter});
            delta_image_riesz_coefficients_3D(:,:,2) = conv2(delta_image_riesz_coefficients_3D(:,:,2),maskLP{scale_counter});
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
            %DO NOT downsample (only filter again and again)
        end
        
        %Assign calculated HIGHPASS (detailed) coefficients to wavelet cell array:
        if flag_use_real_space_or_fourier_space_convolution == 2
            mat_in_riesz_wavelet_cell_array_reference{scale_counter} = ifft2(mat_in_riesz_coefficients_fft_3D_HP);
            delta_image_riesz_wavelet_cell_array_reference{scale_counter} = ifft2(delta_image_riesz_coefficients_fft_3D_HP);
        elseif flag_use_real_space_or_fourier_space_convolution == 1
            mat_in_riesz_wavelet_cell_array_reference{scale_counter} = mat_in_riesz_coefficients_3D_HP;
            delta_image_riesz_wavelet_cell_array_reference{scale_counter} = delta_image_riesz_coefficients_3D_HP;
        end
        
        %Keep track of LOWPASS part too (wavelet is a cascade of highpass parts):
        %(*) the first cell is the raw prefiltered image, the second is the
        %    lowpass filtered downsampled image etc'
        if flag_use_real_space_or_fourier_space_convolution == 2
            mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter+1} = ifft2(mat_in_riesz_coefficients_fft_3D);
            delta_image_riesz_wavelet_lowpass_cell_array_reference{scale_counter+1} = ifft2(delta_image_riesz_coefficients_fft_3D);
        elseif flag_use_real_space_or_fourier_space_convolution == 1
            mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter+1} = mat_in_riesz_coefficients_3D;
            delta_image_riesz_wavelet_lowpass_cell_array_reference{scale_counter+1} = delta_image_riesz_coefficients_3D;
        end
        
    end %END SCALES LOOP
    
    %Assign the appropriate (VERY MUCH DOWNSAMPLED) LOW_PASS RESIDUAL:
    if flag_use_real_space_or_fourier_space_convolution == 2
        mat_in_riesz_wavelet_cell_array_reference{number_of_scales+1} = ifft2(mat_in_riesz_coefficients_fft_3D);
        delta_image_riesz_wavelet_cell_array_reference{number_of_scales+1} = ifft2(delta_image_riesz_coefficients_fft_3D);
    elseif flag_use_real_space_or_fourier_space_convolution == 1
        mat_in_riesz_wavelet_cell_array_reference{number_of_scales+1} = mat_in_riesz_coefficients_3D_HP;
        delta_image_riesz_wavelet_cell_array_reference{number_of_scales+1} = delta_image_riesz_coefficients_3D;
    end
    
end %riesz channels loop

%Get normalization constants for riesz channels:
stdNoiseRiesz = ones(length(noise_riesz_wavelet_cell_array), 1);
stdNoiseRiesz_lowpass = ones(length(noise_riesz_wavelet_cell_array), 1);
for scale_counter = 1:length(delta_image_riesz_wavelet_cell_array_reference)
    stdNoiseRiesz(scale_counter) = std(delta_image_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1)); %normalization only acording to first channel
end
for scale_counter = 1:length(delta_image_riesz_wavelet_lowpass_cell_array_reference)
    stdNoiseRiesz_lowpass(scale_counter) = std(delta_image_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,1));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(D). COMPUTE WAVELET COEFFICIENTS:             
% mat_in_prefiltered_fft = fft2(mat_in); %THIS IS THE ORIGINAL AS FOUND IN riesz_full_monogenic_analysis, seems wrong!!!
mat_in_wavelet_cell_array_reference = cell(1, number_of_levels+1);     
delta_image_wavelet_cell_array_reference = cell(1, number_of_levels+1);
mat_in_wavelet_lowpass_cell_array_reference = cell(1,number_of_scales+2);
delta_image_wavelet_lowpass_cell_array_reference = cell(1,number_of_scales+2);
%(*) Assign raw prefiltered image to first cell of appropriate cell array:
mat_in_wavelet_lowpass_cell_array_reference{1} = mat_in_prefiltered;
delta_image_wavelet_lowpass_cell_array_reference{1} = delta_image_prefiltered;
for scale_counter = 1:number_of_levels
      
    %high pass and lowpass image:
    if flag_use_real_space_or_fourier_space_convolution == 2
        %(1). mat in
        mat_in_prefiltered_fft_HP = mat_in_prefiltered_fft.*maskHP_3D_fft{scale_counter};
        mat_in_prefiltered_fft = mat_in_prefiltered_fft.*maskLP_3D_fft{scale_counter};
        %(2). delta image:
        delta_image_prefiltered_fft_HP = delta_image_prefiltered_fft.*maskHP_3D_fft{scale_counter};
        delta_image_prefiltered_fft = delta_image_prefiltered_fft.*maskLP_3D_fft{scale_counter};
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
    
    %Assign proper HIGHPASS coefficients (direct or fourier space) into the wavelet cell array:
    if flag_use_real_space_or_fourier_space_convolution == 2
        mat_in_wavelet_cell_array_reference{scale_counter} = ifft2(mat_in_prefiltered_fft_HP);
        delta_image_wavelet_cell_array_reference{scale_counter} = ifft2(delta_image_prefiltered_fft_HP);
    else
        mat_in_wavelet_cell_array_reference{scale_counter} = mat_in_prefiltered_HP;
        delta_image_wavelet_cell_array_reference{scale_counter} = delta_image_prefiltered_HP;
    end
    
    %Assign proper LOWPASS coefficients into the wavelet cell array:
    if flag_use_real_space_or_fourier_space_convolution == 2
        mat_in_wavelet_lowpass_cell_array_reference{scale_counter+1} = ifft2(mat_in_prefiltered_fft);
        delta_image_wavelet_lowpass_cell_array_reference{scale_counter+1} = ifft2(delta_image_prefiltered_fft);
    else
        mat_in_wavelet_lowpass_cell_array_reference{scale_counter+1} = mat_in_prefiltered_fft;
        delta_image_wavelet_lowpass_cell_array_reference{scale_counter+1} = delta_image_prefiltered_fft;
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

%Get normalization constant for wavelet transform:
stdNoiseWav = ones(length(delta_image_wavelet_cell_array_reference), 1);
stdNoiseWav_lowpass = ones(length(delta_image_wavelet_cell_array_reference), 1);
for scale_counter = 1:length(delta_image_wavelet_cell_array_reference)
    stdNoiseWav(scale_counter) = std(delta_image_wavelet_cell_array_reference{scale_counter}(:));
end
for scale_counter = 1:length(delta_image_wavelet_lowpass_cell_array_reference)
    stdNoiseWav_lowpass(scale_counter) = std(delta_image_wavelet_lowpass_cell_array_reference{scale_counter}(:));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(E). GET ROTATION (ORIENTATION) ANGLES USING STRUCTURE TENSOR OR REGULAR METHOD:
flag_use_structure_tensor_or_regular_method_angles = 1; %1 or 2
rotation_angles_cell_array = cell(1, number_of_scales);
rotation_angles_lowpass_cell_array = cell(1, number_of_scales);
if flag_use_structure_tensor_or_regular_method_angles == 1
                                                   
     %riesz transform parameters:
     smoothing_filter_number_of_sigmas_to_cutoff = 4; 
     
     %compute the regularization kernel:
     smoothing_filter_axis_center = ceil(smoothing_filter_number_of_sigmas_to_cutoff*smoothing_filter_sigma) + 1;
     smoothing_filter_width = 2*ceil(smoothing_filter_number_of_sigmas_to_cutoff*smoothing_filter_sigma) + 1;
     smoothing_filter = zeros(smoothing_filter_width,smoothing_filter_width);
     for x1 = 1:smoothing_filter_width;
         for x2 = 1:smoothing_filter_width;
             smoothing_filter(x1, x2) = exp(-((x1-smoothing_filter_axis_center)^2 + (x2-smoothing_filter_axis_center)^2)...
                 / (2*smoothing_filter_sigma^2));
         end
     end
     %normalize:
     smoothing_filter = smoothing_filter/sum(smoothing_filter(:));
     
     % full range angle computation
     mat_in_wavelet_gradient_angle = mat_in_wavelet_cell_array_reference;
     mat_in_wavelet_lowpass_gradient_angle = mat_in_wavelet_lowpass_cell_array_reference;
     if flag_restrict_angle_value == 1, %compute sign of the direction thanks to the gradient of the wavelet coefficients
         
         %Compute gradient of wavelet coefficients for different scales:
         for scale_counter = 1:number_of_scales,
             %smooth current wavelet coefficients:
             %MAYBE ADD THAT INSTEAD OF IMFILTER USE A NEW FUNCTION WHICH DOES
             %convolve_without_end_effects BUT FOR 2D?!!?
             mat_in_wavelet_gradient_angle{scale_counter} = imfilter(mat_in_wavelet_gradient_angle{scale_counter}, ...
                                                                     smoothing_filter, 'symmetric');
             mat_in_wavelet_lowpass_gradient_angle{scale_counter} = imfilter(mat_in_wavelet_lowpass_gradient_angle{scale_counter}, ...
                                                                     smoothing_filter, 'symmetric');
             %Compute gradient for current scale wavelet:
             [FX,FY] = gradient(mat_in_wavelet_gradient_angle{scale_counter});
             [FX_lowpass,FY_lowpass] = gradient(mat_in_wavelet_lowpass_gradient_angle{scale_counter});
             
             %determine sign of the angle from the gradient:
             %(KIND OF WEIRD... WHY NOT ATAN2 AND THEN LOOK AT GRADIENT):
             mat_in_wavelet_gradient_angle{scale_counter} = atan2(FY, FX);
             mat_in_wavelet_lowpass_gradient_angle{scale_counter} = atan2(FY_lowpass, FX_lowpass);
         end
     end
     
     %loop over the scales:
     for scale_counter = 1:number_of_scales,
         %compute the 4 Jmn maps:
         if (size(mat_in_riesz_wavelet_cell_array_reference{scale_counter},3)==1) %ordinary wavelet transform (riesz order = 0)
             J11 = real(mat_in_riesz_wavelet_cell_array_reference{scale_counter}).^2;
             J12 = real(mat_in_riesz_wavelet_cell_array_reference{scale_counter}).*imag(mat_in_riesz_wavelet_cell_array_reference{scale_counter});
             J22 = imag(mat_in_riesz_wavelet_cell_array_reference{scale_counter}).^2;
             
             J11_lowpass = real(mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter}).^2;
             J12_lowpass = real(mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter}).*imag(mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter});
             J22_lowpass = imag(mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter}).^2;
         else
             J11 = mat_in_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1).^2;
             J12 = mat_in_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1).*mat_in_riesz_wavelet_cell_array_reference{scale_counter}(:,:,2);
             J22 = mat_in_riesz_wavelet_cell_array_reference{scale_counter}(:,:,2).^2;
             
             J11_lowpass = mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,1).^2;
             J12_lowpass = mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,1).*mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,2);
             J22_lowpass = mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,2).^2;
         end
         
         %convolve the maps with the regularization kernel:
         J11 = imfilter(J11, smoothing_filter, 'symmetric');
         J12 = imfilter(J12, smoothing_filter, 'symmetric');
         J22 = imfilter(J22, smoothing_filter, 'symmetric');
         J11_lowpass  = imfilter(J11_lowpass , smoothing_filter, 'symmetric');
         J12_lowpass  = imfilter(J12_lowpass , smoothing_filter, 'symmetric');
         J22_lowpass  = imfilter(J22_lowpass , smoothing_filter, 'symmetric');
         
         %compute the first eigenvalue table (UNDERSTAND WHY THIS ENABLES PHASE CALCULATION!!!):
         lambda1 = ( J22 + J11 + sqrt((J11-J22).^2 + 4*J12.^2) ) / 2;
         lambda1_lowpass = ( J22_lowpass + J11_lowpass + sqrt((J11_lowpass-J22_lowpass).^2 + 4*J12_lowpass.^2) ) / 2;
         
         if flag_restrict_angle_value, %use the gradient to discriminate angles shifted by pi:
             rotation_angles_cell_array{scale_counter} = atan((lambda1-J11)./J12) + pi*(mat_in_wavelet_gradient_angle{scale_counter}<0);
             rotation_angles_lowpass_cell_array{scale_counter} = atan((lambda1_lowpass-J11_lowpass)./J12_lowpass) + pi*(mat_in_wavelet_lowpass_gradient_angle{scale_counter}<0);
         else
             %compute the first eigen vector direction:
             rotation_angles_cell_array{scale_counter} = atan((lambda1-J11)./J12);
             rotation_angles_lowpass_cell_array{scale_counter} = atan((lambda1_lowpass-J11_lowpass)./J12_lowpass);
         end
     end %END OF SCALE LOOP
                                                   
else %CALCULATE REGULAR METHOD, NOT STRUCTURE TENSOR
    
    rotation_angles_cell_array = atan(mat_in_riesz_wavelet_cell_array_reference{:}(:,:,2) ...
                                   ./ mat_in_riesz_wavelet_cell_array_reference{:}(:,:,1));
    rotation_angles_lowpass_cell_array = atan(mat_in_riesz_wavelet_lowpass_cell_array_reference{:}(:,:,2) ...
                                   ./ mat_in_riesz_wavelet_lowpass_cell_array_reference{:}(:,:,1));                            
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(F). ROTATE RIESZ COEFFICIENTS:
flag_use_generalized_transform_method_simplified_method_or_none = 2; %1 or 2
rotated_riesz_wavelet_cell_array_reference = mat_in_riesz_wavelet_cell_array_reference;
rotated_riesz_wavelet_lowpass_cell_array_reference = mat_in_riesz_wavelet_lowpass_cell_array_reference;
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
    end %end of scales loop
elseif flag_use_generalized_transform_method_simplified_method_or_none == 2
    %(2). using what i understand as the simplified operation for order=1 transform:
    for scale_counter = 1:length(rotation_angles_cell_array)
        rotated_riesz_wavelet_cell_array_reference{scale_counter} = ...
            cos(rotation_angles_cell_array{scale_counter}).*mat_in_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1) ...
            + sin(rotation_angles_cell_array{scale_coutner}).*mat_in_riesz_wavelet_cell_array_reference{scale_counter}(:,:,2);
        
        rotated_riesz_wavelet_lowpass_cell_array_reference{scale_counter} = ...
            cos(rotation_angles_lowpass_cell_array{scale_counter}).*mat_in_riesz_wavelet_cell_lowpass_array_reference{scale_counter}(:,:,1) ...
            + sin(rotation_angles_lowpass_cell_array{scale_coutner}).*mat_in_riesz_wavelet_cell_lowpass_array_reference{scale_counter}(:,:,2);
    end
else
   %(3). do not rotate:
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




                                                                                                             
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(G). COMPUTE PHASE AND AMPLITUDE:
phase_cell_array_reference = cell(1, number_of_scales);
phase_lowpass_cell_array_reference = cell(1,number_of_scales);
amplitude_cell_array_reference = cell(1, number_of_scales);
amplitude_lowpass_cell_array_reference = cell(1, number_of_scales);
flag_get_phase_using_rotated_first_component_or_regular = 1; %1,2 or 3
flag_amplitude_calculation_method = 1; %1,2 or 3
for scale_counter = 1:number_of_scales,
    R1_rotated = rotated_riesz_wavelet_cell_array_reference{:}(:,:,1);
    R2_rotated = rotated_riesz_wavelet_cell_array_reference{:}(:,:,2);
    R1 = mat_in_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1);
    R2 = mat_in_riesz_wavelet_cell_array_reference{scale_counter}(:,:,2);
    I = mat_in_wavelet_cell_array_reference{scale_counter};
    
    R1_rotated_lowpass = rotated_riesz_wavelet_lowpass_cell_array_reference{:}(:,:,1);
    R2_rotated_lowpass = rotated_riesz_wavelet_lowpass_cell_array_reference{:}(:,:,2);
    R1_lowpass = mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,1);
    R2_lowpass = mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,2);
    I_lowpass = mat_in_wavelet_lowpass_cell_array_reference{scale_counter};
    
    
    if flag_get_phase_using_rotated_first_component_or_regular == 1
        phase_cell_array_reference{scale_counter} = ...
            atan( R1_rotated./I...
                         * stdNoiseWav(scale_counter)/stdNoiseRiesz(scale_counter) );
                     
        phase_lowpass_cell_array_reference{scale_counter} = ...
            atan( R1_rotated_lowpass./I_lowpass...
                         * stdNoiseWav_lowpass(scale_counter)/stdNoiseRiesz_lowpass(scale_counter) );             
    elseif flag_get_phase_using_rotated_first_component_or_regular == 2
        phase_cell_array_reference{scale_counter} = ...
            atan(sqrt(R1_rotated.^2+R2_rotated.^2)./I ...
                         * stdNoiseWav(scale_counter)/stdNoiseRiesz(scale_counter) );
                     
        phase_lowpass_cell_array_reference{scale_counter} = ...
            atan(sqrt(R1_rotated_lowpass.^2+R2_rotated_lowpass.^2)./I_lowpass ...
                         * stdNoiseWav_lowpass(scale_counter)/stdNoiseRiesz_lowpass(scale_counter) );              
    elseif flag_get_phase_using_rotated_first_component_or_regular == 3
        phase_cell_array_reference{scale_counter} = ...
            atan(sqrt(R1.^2+R2.^2)./I ...
                         * stdNoiseWav(scale_counter)/stdNoiseRiesz(scale_counter) );
                     
        phase_lowpass_cell_array_reference{scale_counter} = ...
            atan(sqrt(R1_lowpass.^2+R2_lowpass.^2)./I_lowpass ...
                         * stdNoiseWav_lowpass(scale_counter)/stdNoiseRiesz_lowpass(scale_counter) );              
    end
                                
    %HOW COME THE AMPLITUDE EXPRESSION DOESN'T INCLUDE ALL THREE TERMS AS BELOW????:
    %amplitude{j} = sqrt(QA{j}(:,:,1).^2 + QA{j}(:,:,2).^2 + Q{j}.^2*stdNoiseRiesz{j}(1)/stdNoiseWav(j));
    if flag_amplitude_calculation_method == 1
        amplitude_cell_array_reference{scale_counter} = ...
            sqrt( (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1)/stdNoiseRiesz(scale_counter)).^2 + ...
                  (mat_in_wavelet_cell_array_reference{scale_counter}/stdNoiseWav(scale_counter)).^2 );
        
        amplitude_lowpass_cell_array_reference{scale_counter} = ...
            sqrt( (rotated_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,1)/stdNoiseRiesz_lowpass(scale_counter)).^2 + ...
                  (mat_in_wavelet_lowpass_cell_array_reference{scale_counter}/stdNoiseWav_lowpass(scale_counter)).^2 );
    elseif flag_amplitude_calculation_method == 2
        amplitude_cell_array_reference{scale_counter} = ...    
            sqrt( (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1)/stdNoiseRiesz(scale_counter)).^2 + ...
                  (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,2)/stdNoiseRiesz(scale_counter)).^2 + ...
                  (mat_in_wavelet_cell_array_reference{scale_counter}/stdNoiseWav(scale_counter)).^2 );
              
        amplitude_lowpass_cell_array_reference{scale_counter} = ...    
            sqrt( (rotated_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,1)/stdNoiseRiesz_lowpass(scale_counter)).^2 + ...
                  (rotated_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,2)/stdNoiseRiesz_lowpass(scale_counter)).^2 + ...
                  (mat_in_wavelet_lowpass_cell_array_reference{scale_counter}/stdNoiseWav_lowpass(scale_counter)).^2 );
    elseif flag_amplitude_calculation_method == 3
        amplitude_cell_array_reference{scale_counter} = ...
            sqrt( (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1)).^2 + ...
                  (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,2)).^2 + ...
                  (mat_in_wavelet_cell_array_reference{scale_counter}*stdNoiseRiesz(scale_counter)/stdNoiseWav(scale_counter)).^2 );
              
        amplitude_lowpass_cell_array_reference{scale_counter} = ...
            sqrt( (rotated_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,1)).^2 + ...
                  (rotated_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,2)).^2 + ...
                  (mat_in_wavelet_lowpass_cell_array_reference{scale_counter}*stdNoiseRiesz_lowpass(scale_counter)/stdNoiseWav_lowpass(scale_counter)).^2 );
    elseif flag_amplitude_calculation_method == 4
        amplitude_cell_array_reference{scale_counter} = ...    
            sqrt( (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1)).^2 + ...
                  (rotated_riesz_wavelet_cell_array_reference{scale_counter}(:,:,2)).^2 + ...
                  (mat_in_wavelet_cell_array_reference{scale_counter}).^2 );
              
        amplitude_lowpass_cell_array_reference{scale_counter} = ...    
            sqrt( (rotated_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,1)).^2 + ...
                  (rotated_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,2)).^2 + ...
                  (mat_in_wavelet_lowpass_cell_array_reference{scale_counter}).^2 );
    end
end %END SCALE LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 




 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(H). UNWRAP PHASE:
%do some unwrapping algorithm if wanted...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(2). rotation angles using regular:
quaternion_vec1 = [sin(phi)*cos(theta) , sin(phi)*sin(theta)];
quaternion_vec2 = [phi*cos(theta) , phi*sin(theta)];




%% TRY ALGORITHMS:
%**AFTERWARDS SEE HOW TO FILTER SIGNAL, WHETHER TO USE PHASE DIRECTLY OR TO
%  USE THE SUGGESTION IN THE RUBINSTEIN PAPER OF USING
%  [phi*cos(theta),phi*sin(theta)] or [sin(phi)*cos(theta),sin(phi)*sin(theta)]
%noise parameters:
optical_SNR = inf;
mat_in_signal_per_pixel = sum(mat_in(:))/numel(mat_in);
noise_std = mat_in_signal_per_pixel*(1/optical_SNR);

%Initialize initial noisy mat:
%(1). Normalize audio_vec:
max_shift_size = 10^-1; %[pixels]
audio_vec = audio_vec/max(abs(audio_vec))*max_shift_size;
%(2). Shift matrix to sample 1 position:
shift_angle = pi/4;
shift_size = audio_vec(1);
mat_in = shift_matrix(mat_in,1,shift_size*cos(shift_angle),shift_size*sin(shift_angle));
%(3). Add Noise for Initial image:
mat_in_size = size(mat_in);
noise_mat = noise_std * randn(mat_in_size);
noisy_mat1 = mat_in + noise_mat;
for sample_counter = 2:number_of_samples
    
    %Get current sample:
    shift_size = audio_vec(sample_counter); 
    
    %Initialize noisy mats:
    noisy_mat2 = shift_matrix(mat_in,1,shift_size*cos(shift_angle),shift_size*sin(shift_angle));
    noise_mat = noise_std * randn(mat_in_size);
    noisy_mat2 = noisy_mat2 + noise_mat;

    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Register images:
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(A). PREFILTEROF DIFFERENT  IMAGES:
    mat_in = noisy_mat2;
    mat_in_fft = fft2(mat_in);
    if ~isempty(riesz_transform_configurations_object.prefilter.filterLow),
        mat_in_prefiltered = ifft2(mat_in_fft .* riesz_transform_prefilter_lowpass);
        mat_in_prefiltered_highpassed = ifft2(mat_in_fft .* riesz_transform_prefilter_highpass);
    else
        mat_in_prefiltered = mat_in;
        mat_in_prefiltered_highpassed = zeros(size(mat_in));
    end
    mat_in_prefiltered_fft = fft2(mat_in_prefiltered);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(B). PERFORM RIESZ-TRANSFORM TO CURRENT (PREFILTERED) IMAGE:
    %(1). to mat_in:
    %fourier space:
    mat_in_riesz_coefficients_fft_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
    mat_in_riesz_coefficients_fft_3D(:,:,1) = mat_in_prefiltered_fft.*riesz_transform_fourier_filter_x;
    mat_in_riesz_coefficients_fft_3D(:,:,2) = mat_in_prefiltered_fft.*riesz_transform_fourier_filter_y;
    %direct space:
    mat_in_riesz_coefficients_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
    mat_in_riesz_coefficients_3D(:,:,1) = real(ifft2(mat_in_riesz_coefficients_fft_3D(:,:,1)));
    mat_in_riesz_coefficients_3D(:,:,2) = real(ifft2(mat_in_riesz_coefficients_fft_3D(:,:,2)));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(C). APPLY WAVELET DECOMPOSITION TO RIESZ COEFFICIENTS
    mat_in_riesz_wavelet_cell_array = cell(1, number_of_scales+1);
    mat_in_riesz_wavelet_lowpass_cell_array = cell(1,number_of_scales+2);
    %(*) Assign raw prefiltered image to first cell of appropriate cell array:
    mat_in_riesz_wavelet_lowpass_cell_array{1} = mat_in_riesz_coefficients_3D;
    for riesz_channel_counter = 1:number_of_riesz_channels,
        %Compute fft2 because before we used ifft2 in riesz_coefficients_matrix_of_lowpassed_image_3D calculation:
        highpass_wavelet_coefficients = cell(1, number_of_scales);
        for scale_counter = 1:number_of_scales
            
            %Lowpass and Highpass (add highpass to wavelet cell array and use lowpass for coarser scales):
            if flag_use_real_space_or_fourier_space_convolution == 2
                %(1) mat_in riesz-wavelet:
                mat_in_riesz_coefficients_fft_3D_HP = mat_in_riesz_coefficients_fft_3D.*maskHP_3D_fft{scale_counter};
                mat_in_riesz_coefficients_fft_3D = mat_in_riesz_coefficients_fft_3D.*maskLP_3D_fft{scale_counter};
            elseif flag_use_real_space_or_fourier_space_convolution == 1
                %SEE IF CONV2 CAN DO 2D CONVOLUTION FOR EACH MATRIX ALONG THE THIRD DIMENSION:
                %(1) mat_in riesz-wavelet:
                mat_in_riesz_coefficients_3D_HP(:,:,1) = conv2(mat_in_riesz_coefficients_3D(:,:,1),maskHP{scale_counter});
                mat_in_riesz_coefficients_3D(:,:,1) = conv2(mat_in_riesz_coefficients_3D(:,:,1),maskLP{scale_counter});
                mat_in_riesz_coefficients_3D_HP(:,:,2) = conv2(mat_in_riesz_coefficients_3D(:,:,2),maskHP{scale_counter});
                mat_in_riesz_coefficients_3D(:,:,2) = conv2(mat_in_riesz_coefficients_3D(:,:,2),maskLP{scale_counter});
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
                elseif flag_use_real_space_or_fourier_space_convolution == 1
                    %direct space downsampling:
                    mat_in_riesz_coefficients_3D = mat_in_riesz_coefficients_3D(1:2:end,1:2:end);
                end
            else
                %DO NOT downsample (only filter again and again)
            end
            
            %Assign calculated HIGHPASS (detailed) coefficients to wavelet cell array:
            if flag_use_real_space_or_fourier_space_convolution == 2
                mat_in_riesz_wavelet_cell_array{scale_counter} = ifft2(mat_in_riesz_coefficients_fft_3D_HP);
            elseif flag_use_real_space_or_fourier_space_convolution == 1
                mat_in_riesz_wavelet_cell_array{scale_counter} = mat_in_riesz_coefficients_3D_HP;
            end
            
            %Keep track of LOWPASS part too (wavelet is a cascade of highpass parts):
            %(*) the first cell is the raw prefiltered image, the second is the
            %    lowpass filtered downsampled image etc'
            if flag_use_real_space_or_fourier_space_convolution == 2
                mat_in_riesz_wavelet_lowpass_cell_array{scale_counter+1} = ifft2(mat_in_riesz_coefficients_fft_3D);
            elseif flag_use_real_space_or_fourier_space_convolution == 1
                mat_in_riesz_wavelet_lowpass_cell_array{scale_counter+1} = mat_in_riesz_coefficients_3D;
            end
            
        end %END SCALES LOOP
        
        %Assign the appropriate (VERY MUCH DOWNSAMPLED) LOW_PASS RESIDUAL:
        if flag_use_real_space_or_fourier_space_convolution == 2
            mat_in_riesz_wavelet_cell_array{number_of_scales+1} = ifft2(mat_in_riesz_coefficients_fft_3D);
        elseif flag_use_real_space_or_fourier_space_convolution == 1
            mat_in_riesz_wavelet_cell_array{number_of_scales+1} = mat_in_riesz_coefficients_3D_HP;
        end
        
    end %riesz channels loop
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(D). COMPUTE WAVELET COEFFICIENTS:
    % mat_in_prefiltered_fft = fft2(mat_in); %THIS IS THE ORIGINAL AS FOUND IN riesz_full_monogenic_analysis, seems wrong!!!
    mat_in_wavelet_cell_array = cell(1, number_of_levels+1);
    mat_in_wavelet_lowpass_cell_array = cell(1,number_of_scales+2);
    %(*) Assign raw prefiltered image to first cell of appropriate cell array:
    mat_in_wavelet_lowpass_cell_array{1} = mat_in_prefiltered;
    for scale_counter = 1:number_of_levels
        
        %high pass and lowpass image:
        if flag_use_real_space_or_fourier_space_convolution == 2
            %(1). mat in
            mat_in_prefiltered_fft_HP = mat_in_prefiltered_fft.*maskHP_3D_fft{scale_counter};
            mat_in_prefiltered_fft = mat_in_prefiltered_fft.*maskLP_3D_fft{scale_counter};
        else
            %(1). mat in
            mat_in_prefiltered_HP = conv2(mat_in_prefiltered,maskHP_fft{scale_counter});
            mat_in_prefiltered = conv2(mat_in_prefiltered,maskLP_fft{scale_counter});
        end
        
        %Downsample:
        if flag_downsample_or_not == 1
            if flag_use_real_space_or_fourier_space_convolution == 2
                c2 = size(mat_in_prefiltered_fft,2)/2;
                c1 = size(mat_in_prefiltered_fft,1)/2;
                %(1). mat in:
                mat_in_prefiltered_fft = 0.25*(mat_in_prefiltered_fft(1:c1, 1:c2) + mat_in_prefiltered_fft((1:c1)+c1, 1:c2) + ...
                    mat_in_prefiltered_fft((1:c1) + c1, (1:c2) +c2) + mat_in_prefiltered_fft(1:c1, (1:c2) +c2));
            elseif flag_use_real_space_or_fourier_space_convolution == 1
                %(1). mat in:
                mat_in_prefiltered = mat_in_prefiltered(1:2:end);
            end
            
        else
            %do NOT downsample (only filter again and again)
        end
        
        %Assign proper HIGHPASS coefficients (direct or fourier space) into the wavelet cell array:
        if flag_use_real_space_or_fourier_space_convolution == 2
            mat_in_wavelet_cell_array{scale_counter} = ifft2(mat_in_prefiltered_fft_HP);
        else
            mat_in_wavelet_cell_array{scale_counter} = mat_in_prefiltered_HP;
        end
        
        %Assign proper LOWPASS coefficients into the wavelet cell array:
        if flag_use_real_space_or_fourier_space_convolution == 2
            mat_in_wavelet_lowpass_cell_array{scale_counter+1} = ifft2(mat_in_prefiltered_fft);
        else
            mat_in_wavelet_lowpass_cell_array{scale_counter+1} = mat_in_prefiltered_fft;
        end
        
    end %LEVELS/SCALES LOOP
    %Assign lowpass residual:
    if flag_use_real_space_or_fourier_space_convolution == 2
        mat_in_wavelet_cell_array{number_of_scales+1} = ifft2(mat_in_prefiltered_fft);
    else
        mat_in_wavelet_cell_array{number_of_scales+1} = mat_in_prefiltered;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(E). Rotate riesz transform:
    rotated_riesz_wavelet_cell_array = mat_in_riesz_wavelet_cell_array_reference;
    rotated_riesz_wavelet_lowpass_cell_array = mat_in_riesz_wavelet_lowpass_cell_array_reference;
    if flag_use_generalized_transform_method_simplified_method_or_none == 2
        %(2). using what i understand as the simplified operation for order=1 transform:
        for scale_counter = 1:length(rotation_angles_cell_array)
            rotated_riesz_wavelet_cell_array{scale_counter} = ...
                cos(rotation_angles_cell_array{scale_counter}).*mat_in_riesz_wavelet_cell_array{scale_counter}(:,:,1) ...
                + sin(rotation_angles_cell_array{scale_coutner}).*mat_in_riesz_wavelet_cell_array{scale_counter}(:,:,2);
            
            rotated_riesz_wavelet_lowpass_cell_array{scale_counter} = ...
                cos(rotation_angles_lowpass_cell_array{scale_counter}).*mat_in_riesz_wavelet_cell_lowpass_array{scale_counter}(:,:,1) ...
                + sin(rotation_angles_lowpass_cell_array{scale_coutner}).*mat_in_riesz_wavelet_cell_lowpass_array{scale_counter}(:,:,2);
        end
    else
        %(3). do not rotate:
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(F). COMPUTE PHASE AND AMPLITUDE OF CURRENT IMAGE:
    phase_cell_array = cell(1, number_of_scales);
    phase_lowpass_cell_array = cell(1,number_of_scales);
    amplitude_cell_array = cell(1, number_of_scales);
    amplitude_lowpass_cell_array = cell(1, number_of_scales);
    for scale_counter = 1:number_of_scales,
        R1_rotated = rotated_riesz_wavelet_cell_array{:}(:,:,1);
        R2_rotated = rotated_riesz_wavelet_cell_array{:}(:,:,2);
        R1 = mat_in_riesz_wavelet_cell_array{scale_counter}(:,:,1);
        R2 = mat_in_riesz_wavelet_cell_array{scale_counter}(:,:,2);
        I = mat_in_wavelet_cell_array{scale_counter};
        
        R1_rotated_lowpass = rotated_riesz_wavelet_lowpass_cell_array{:}(:,:,1);
        R2_rotated_lowpass = rotated_riesz_wavelet_lowpass_cell_array{:}(:,:,2);
        R1_lowpass = mat_in_riesz_wavelet_lowpass_cell_array{scale_counter}(:,:,1);
        R2_lowpass = mat_in_riesz_wavelet_lowpass_cell_array{scale_counter}(:,:,2);
        I_lowpass = mat_in_wavelet_lowpass_cell_array{scale_counter};
        
        
        if flag_get_phase_using_rotated_first_component_or_regular == 1
            phase_cell_array{scale_counter} = ...
                atan( R1_rotated./I...
                * stdNoiseWav(scale_counter)/stdNoiseRiesz(scale_counter) );
            
            phase_lowpass_cell_array{scale_counter} = ...
                atan( R1_rotated_lowpass./I_lowpass...
                * stdNoiseWav_lowpass(scale_counter)/stdNoiseRiesz_lowpass(scale_counter) );
        elseif flag_get_phase_using_rotated_first_component_or_regular == 2
            phase_cell_array{scale_counter} = ...
                atan(sqrt(R1_rotated.^2+R2_rotated.^2)./I ...
                * stdNoiseWav(scale_counter)/stdNoiseRiesz(scale_counter) );
            
            phase_lowpass_cell_array{scale_counter} = ...
                atan(sqrt(R1_rotated_lowpass.^2+R2_rotated_lowpass.^2)./I_lowpass ...
                * stdNoiseWav_lowpass(scale_counter)/stdNoiseRiesz_lowpass(scale_counter) );
        elseif flag_get_phase_using_rotated_first_component_or_regular == 3
            phase_cell_array{scale_counter} = ...
                atan(sqrt(R1.^2+R2.^2)./I ...
                * stdNoiseWav(scale_counter)/stdNoiseRiesz(scale_counter) );
            
            phase_lowpass_cell_array{scale_counter} = ...
                atan(sqrt(R1_lowpass.^2+R2_lowpass.^2)./I_lowpass ...
                * stdNoiseWav_lowpass(scale_counter)/stdNoiseRiesz_lowpass(scale_counter) );
        end
        
        %HOW COME THE AMPLITUDE EXPRESSION DOESN'T INCLUDE ALL THREE TERMS AS BELOW????:
        %amplitude{j} = sqrt(QA{j}(:,:,1).^2 + QA{j}(:,:,2).^2 + Q{j}.^2*stdNoiseRiesz{j}(1)/stdNoiseWav(j));
        if flag_amplitude_calculation_method == 1
            amplitude_cell_array_reference{scale_counter} = ...
                sqrt( (rotated_riesz_wavelet_cell_array{scale_counter}(:,:,1)/stdNoiseRiesz(scale_counter)).^2 + ...
                (mat_in_wavelet_cell_array{scale_counter}/stdNoiseWav(scale_counter)).^2 );
            
            amplitude_lowpass_cell_array{scale_counter} = ...
                sqrt( (rotated_riesz_wavelet_lowpass_cell_array{scale_counter}(:,:,1)/stdNoiseRiesz_lowpass(scale_counter)).^2 + ...
                (mat_in_wavelet_lowpass_cell_array{scale_counter}/stdNoiseWav_lowpass(scale_counter)).^2 );
        elseif flag_amplitude_calculation_method == 2
            amplitude_cell_array_reference{scale_counter} = ...
                sqrt( (rotated_riesz_wavelet_cell_array{scale_counter}(:,:,1)/stdNoiseRiesz(scale_counter)).^2 + ...
                (rotated_riesz_wavelet_cell_array{scale_counter}(:,:,2)/stdNoiseRiesz(scale_counter)).^2 + ...
                (mat_in_wavelet_cell_array{scale_counter}/stdNoiseWav(scale_counter)).^2 );
            
            amplitude_lowpass_cell_array{scale_counter} = ...
                sqrt( (rotated_riesz_wavelet_lowpass_cell_array{scale_counter}(:,:,1)/stdNoiseRiesz_lowpass(scale_counter)).^2 + ...
                (rotated_riesz_wavelet_lowpass_cell_array{scale_counter}(:,:,2)/stdNoiseRiesz_lowpass(scale_counter)).^2 + ...
                (mat_in_wavelet_lowpass_cell_array{scale_counter}/stdNoiseWav_lowpass(scale_counter)).^2 );
        elseif flag_amplitude_calculation_method == 3
            amplitude_cell_array_reference{scale_counter} = ...
                sqrt( (rotated_riesz_wavelet_cell_array{scale_counter}(:,:,1)).^2 + ...
                (rotated_riesz_wavelet_cell_array{scale_counter}(:,:,2)).^2 + ...
                (mat_in_wavelet_cell_array{scale_counter}*stdNoiseRiesz(scale_counter)/stdNoiseWav(scale_counter)).^2 );
            
            amplitude_lowpass_cell_array{scale_counter} = ...
                sqrt( (rotated_riesz_wavelet_lowpass_cell_array{scale_counter}(:,:,1)).^2 + ...
                (rotated_riesz_wavelet_lowpass_cell_array{scale_counter}(:,:,2)).^2 + ...
                (mat_in_wavelet_lowpass_cell_array{scale_counter}*stdNoiseRiesz_lowpass(scale_counter)/stdNoiseWav_lowpass(scale_counter)).^2 );
        elseif flag_amplitude_calculation_method == 4
            amplitude_cell_array_reference{scale_counter} = ...
                sqrt( (rotated_riesz_wavelet_cell_array{scale_counter}(:,:,1)).^2 + ...
                (rotated_riesz_wavelet_cell_array{scale_counter}(:,:,2)).^2 + ...
                (mat_in_wavelet_cell_array{scale_counter}).^2 );
            
            amplitude_lowpass_cell_array{scale_counter} = ...
                sqrt( (rotated_riesz_wavelet_lowpass_cell_array{scale_counter}(:,:,1)).^2 + ...
                (rotated_riesz_wavelet_lowpass_cell_array{scale_counter}(:,:,2)).^2 + ...
                (mat_in_wavelet_lowpass_cell_array{scale_counter}).^2 );
        end
    end %END SCALE LOOP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(G). CALCULATE PHASE DIFFERENCE BETWEEN CURRENT AND PREVIOUS FRAMES:
    [phase_difference_cos, phase_difference_sin, amplitude] = ...
                 compute_phase_difference_and_amplitude_of_riesz_bands(I,...
                                                                       R1,...
                                                                       R2,...
                                                                       I_previous,...
                                                                       R1_previous,...
                                                                       R2_previous);
    [phase_difference_cos, phase_difference_sin, amplitude] = ...
                 compute_phase_difference_and_amplitude_of_riesz_bands(I,...
                                                                       R1_rotated,...
                                                                       R2_rotated,...
                                                                       I_previous,...
                                                                       R1_rotated_previous,...
                                                                       R2_rotated_previous);                                                               
    
    [phase_difference_cos, phase_difference_sin, amplitude] = ...
                 compute_phase_difference_and_amplitude_of_riesz_bands(I,...
                                                                       R1_lowpass,...
                                                                       R2_lowpass,...
                                                                       I_lowpass,...
                                                                       R1_lowpass_previous,...
                                                                       R2_lowpass_previous);
                                                                   
    [phase_difference_cos, phase_difference_sin, amplitude] = ...
                 compute_phase_difference_and_amplitude_of_riesz_bands(I,...
                                                                       R1_rotated_lowpass,...
                                                                       R2_rotated_lowpass,...
                                                                       I_lowpass,...
                                                                       R1_rotated_lowpass_previous,...
                                                                       R2_rotated_lowpass_previous);                                                               
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %Assign current frame to be previous frame:
    noisy_mat1 = noisy_mat2;
    R1_rotated_previous = R1_rotated;
    R2_rotated_previous = R2_rotated;
    R1_previous = R1;
    R2_previous = R2;
    I_previous = I;
    R1_rotated_lowpass_previous = R1_rotated_lowpass;
    R2_rotated_lowpass_previous = R2_rotated_lowpass;
    R1_lowpass_previous = R1_lowpass;
    R2_lowpass_previous = R2_lowpass;
    I_lowpass_previous = I_lowpass;
    
end



















