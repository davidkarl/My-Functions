%VISUAL MICROPHONE ELBIT

%Threre are two basic ways to extract the phase and play with it:
%(1). Complex Steerable Pyramid:
    %(*) a pyramid decomposition (usually with filters in the fourier domain
    %    unless one uses the real space implementation which could be slow but is
    %    worth checking out at: Pyramidal Dual Tree Directional Filter Bank Decomposition)
    %(*) returns a complex (approximately analytic) version of the image
    %    where the hilbert real/imag pair was calculated with respect to a
    %    slice in fourier space. as implied, this decomposition uses many
    %    slices (a chosen variable) to slice fourier space into directional
    %    slices and radial slices (the radial slices are for the highpass and
    %    lowpass WAVELET decomposition). the slices are not symmetric cones
    %    with respect to the (0,0) frequency precisely to approximate the
    %    effect of a hilbert transform only in 2d (in 1d one simply zeros
    %    out the negative frequencies).
    %(*) when using this method all that is needed is to take the returned
    %    complex image decomposition and get the angle(filtered_image(angle_counter,scale_counter)).
    %    another way this is useful is when i want "FM", or phase differences
    %    between images - all i have to do is to take the "exponential" part of
    %    the returned image (image=R*exp(i*theta)) and element-divide it by
    %    the previous complex filtered image (just like i do when i use 1D FM
    %    using hilbert transform).
    %(*) another possible advantage of this method is that it effectively slices the fourier domain 
    %    which means it can effectively handle different frequencies
    %    different SNRs (also if one looks into it, it usually offers a
    %    pretty good denoising method because of that).
    
    
%(2). Riesz Transform:
    %(*) the riesz transform is the natural extention of the 1D hilbert transform into 2D.
    %(*) supposedly, with this method i can simply take ALL THE FREQUENCIES and
    %    use a riesz transform on them to extract the phase considering all
    %    frequenceis just as i would do with the 1D hilbert transform:
    %(*) what is done in Motion-Magnification is a decomposition of the frame
    %    into multiple scales just like any other wavelet, then a riesz
    %    transform is computed for each wavelet scale, which allows us to
    %    extract the phase and magnify it and then reconstruct everything
    %    nicely.
    %(*) as for the riesz-transform itself (the analogous of the Hilbert transform),
    %    there are two main ways to implement it. 
    %    ONE WAY is to implement it in the frequency domain using the matrix w_x/sqrt(w_x^2+w_y^2) and
    %    w_y/sqrt(w_x^2+w_y^2) as the transfer functions for each of the quatarions. 
    %    A SECOND WAY is to do what rubinstein and wadwah did, which is to
    %    constuct a specialized wavelet decomposition whose highpassed allowed
    %    frequencies enable a subsequent riesz transform which is very compact
    %    in REAL SPACE and which very closely resembles a derivative term
    %    ([-0.5,0,0.5]). 
    %(*). both ways are legitimate, however it is worth checking out which way
    %     is faster, the specialized real space APPROXIMATE riesz transform or
    %     the pseudo-exact fourier space riesz transform.
    %(*). the riesz transform can be generalized to multiple derivatives and
    %     directions as is done in the generalized riesz transform "toolbox".
    %     there are also works which explicitely generalize the riesz transform
    %     to multiple channels of a 2d matrix (like colored video). perhapse
    %     that would add something....i don't know...maybe it's just
    %     intellectual musterbation.
    %(*). in any case one must remember that, as i understand it, it is
    %     possible to DOWNSAMPLE IN FOURIER SPACE, which perhapse might give
    %     that precious extra boost in performance needed when taking the
    %     fourier space filtering approach.

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VIDEO PARAMETERS FOR ACTUAL VIDEO: 

%Get input variables:
video_file_name = 'Chips1-2200Hz-Mary_Had-input.avi';
video_file_name = 'image_warps.avi';
test_image_file_name = 'barbara.tif';
output_directory = 'C:\Users\master\Desktop\matlab';
low_cutoff_frequency = 100;
high_cutoff_frequency = 1000;
Fs = 2200;
phase_magnification_factor = 15;    
number_of_scales = 4; %number of scales to actually use
number_of_levels = number_of_scales;

%Read Video Parameters 
video_reader_object = VideoReader(video_file_name);
[~, writeTag, ~] = fileparts(video_file_name);
Fs = video_reader_object.FrameRate;
%Frequencies:
low_cutoff_frequency = 100;
high_cutoff_frequency = 1000;
Fs = 2200; %delete this when real data stream arrives
%
video_frames = video_reader_object.read();
[frame_height, frame_width, number_of_channels, number_of_frames] = size(video_frames);

%Get reference frame for forward modeling and also structure tensor modeling:
reference_frame = 1;
mat_in_total_channels = squeeze(video_frames(:,:,:,reference_frame));
mat_in = mat_in_total_channels(:,:,1); %just for modeling the filters and stuff for later
mat_in_fft = fft2(mat_in);
mat_in_power_spectrum = abs(mat_in_fft).^2;
mat_in_number_of_dimensions = 2;


% %Import test image to be able to clear script:
% mat_in_total_channels = double(imread(test_image_file_name)); %only 1 channel for barbara.tif
% mat_in = mat_in_total_channels;
% mat_in_fft = fft2(mat_in);
% mat_in_ps = abs(mat_in_fft).^2;
% mat_in_number_of_dimensions = 2;
% number_of_frames = 5; %doesn't mean anything right now
% number_of_channels = 1;
% [frame_height,frame_width] = size(mat_in_total_channels);

%Get Structure-Tensor for later:
%(*) flags for structure tensor feature map calculation
flag_derivative_method = 1;
flag_filter_before_derivative = 0; %0=don't, 1=1d, 2=2d
flag_filter_after_derivative = 0; %0=don't, 1=2d, 2=perpendicular to axis
flag_feature_map = 3; %1=Ixx+Iyy, 2=Ixx+Iyy+Ixy, 3=Ixx+Iyy+2*abs(Ixy), 4=coherence
flag_coherence_measure = 1;
flag_filter_feature_map = 1;
feature_map_regularizer = 5;
%(*) filters for structure tensor feature map calcuation
input_smoothing_filter_size = 5;
input_smoothing_filter_sigma = 2;
output_smoothing_filter_size = 5;
output_smoothing_filter_sigma = 2;
feature_map_filter_size = 5;
feature_map_filter_sigma = 2;
%(*) actually get structure tensor and subsequent feature map (heat map):
%I SHOULD DO IT EITHER FOR ALL THREE COLOR CHANNELS OR FOR INTENSITY!!!!
[feature_map_reference, Ixx, Iyy, Ixy] = get_structure_tensor(...
    mat_in, ...
    flag_derivative_method,...
    flag_filter_before_derivative,...
    flag_filter_after_derivative,...
    flag_feature_map,...
    flag_coherence_measure,...
    flag_filter_feature_map,...
    feature_map_regularizer,...
    input_smoothing_filter_size,...
    input_smoothing_filter_sigma, ...
    output_smoothing_filter_size,...
    output_smoothing_filter_sigma,...
    feature_map_filter_size,...
    feature_map_filter_sigma);
% figure(1); imagesc(mat_in); figure(2); imagesc(feature_map_reference);

%get dyadic partition with respect to height and width:
dyadic_partition_height = get_dyadic_partition_of_nondyadic_signals(frame_height);
dyadic_partition_width = get_dyadic_partition_of_nondyadic_signals(frame_width);
%flip dyadic partition vec to get correct order (lowering dimension the larger the vector index):
rows_dyadic_partition = flip(dyadic_partition_height);
columns_dyadic_partition = flip(dyadic_partition_width);
dyadic_size_vecs_cell = cell(number_of_levels,1);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Temporal Filter parameters (for phase temporal filtering):

low_cutoff_frequency_normalized = low_cutoff_frequency/(Fs/2);
high_cutoff_frequency_normalized = high_cutoff_frequency/(Fs/2);
%(1). FIR:
temporal_filter_function = @FIRWindowBP; %SWITCH!
%(2). IIR (IMPLEMENT FUNCTION):
temporal_filter_order = 1;
%IMPLEMENT get_butterworth_filter_coefficients (in order to filter many
%channels and levels i need to put the phases into a matrix form).
[b,a] = butter(temporal_filter_order,[low_cutoff_frequency_normalized,high_cutoff_frequency_normalized],'bandpass');
IIR_filter_object = dsp.IIRFilter;
IIR_filter_object.Numerator = b;
IIR_filter_object.Denominator = a;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% PHASE ANALYSIS METHOD 1: Complex Steerable Pyramid %%%%%%%%%%%%

%Parameters:
pyramid_type = 'octave';
flag_attenuate_other_frequencies = false; %?
smoothing_filter_sigma = 1.5; %seems much

%Get Complex Steerable Pyramid parameters (For reference):
pyramid_types_strings = {'octave','halfOctave','smoothHalfOctave','quarterOctave'};

%Get Complex Steerable Pyramid Filters:
complex_steerable_pyramid_max_thinkable_height = floor(log2(min(frame_height,frame_width))); %ALL SCALES, CHANGE THIS
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
%complex_steerable_pyramid_filters_fourier - already after fftshift and make sense! - i
%should do another fftshift to allow for efficiency.
[cropped_complex_steerable_pyramid_filters,filters_non_zero_indices_cell_mat] = ...
                        get_filters_and_indices_which_are_non_zero(complex_steerable_pyramid_filters_fourier);

%For easier reading define filters_non_zero_indices:
number_of_complex_steerable_pyramid_filters = numel(complex_steerable_pyramid_filters_fourier);
for filter_counter = 1:number_of_complex_steerable_pyramid_filters
    filters_non_zero_indices{filter_counter} = [filters_non_zero_indices_cell_mat{filter_counter,1}, ...
                                                filters_non_zero_indices_cell_mat{filter_counter,2}];
end
                    
%Define build_level and reconstruct_level function (CHANGE cropped_filters and functions to be WITHOUT fftshift):
% build_level_function = @(mat_in_fft,k) ifft2(ifftshift(cropped_complex_steerable_pyramid_filters{k} .* mat_in_fft(filters_non_zero_indices{k})));
% build_level_function = @(mat_in_fft,k) ifft2(ifftshift(cropped_complex_steerable_pyramid_filters{k} .* mat_in_fft(filters_non_zero_indices_cell_mat{k,1},filters_non_zero_indices_cell_mat{k,2}) ));
build_level_function = @(mat_in_fft,k) ifft2(ifftshift(complex_steerable_pyramid_filters_fourier{k} .* mat_in_fft));
reconstruct_level_function = @(mat_in_fft,k) 2*(complex_steerable_pyramid_filters_fourier{k}.*fftshift(fft2(mat_in_fft)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% PHASE ANALYSIS METHOD 2: Riesz Transform %%%%%%%%%%%%

%(1). use Rubinstein's riesz pyramid wavelet transform:
%filters currently need fftshift to make sense
[ filter_1D_lowpass_coefficients_rubinstein, ...
  filter_1D_highpass_coefficients_rubinstein, ...
  chebychev_polynomial_lowpass_rubinstein, ...
  chebychev_polynomial_highpass_rubinstein, ...
  McClellan_transform_matrix_rubinstein, ...
  filter_2D_lowpass_direct_rubinstein, ...
  filter_2D_highpass_direct_rubinstein ] = get_filters_for_riesz_pyramid_rubinstein();
filter_2D_lowpass_direct_rubinstein_fft = fft2(filter_2D_lowpass_direct_rubinstein,frame_height,frame_width);
filter_2D_highpass_direct_rubinstein_fft = fft2(filter_2D_highpass_direct_rubinstein,frame_height,frame_width);


%(2). use Generalized Riesz-Transform proposed wavelet decomposition pyramids stuff:
%(*) wavelet types and prefilter and stuff for reference:
wavelet_types = {'isotropic'};
isotropic_wavelet_types = {'simoncelli','shannon','aldroubi','papadakis','meyer','ward'};
prefilter_types = {'shannon','simoncelli','none'};
%(*) riesz and just wavelet transform objects:
riesz_transform_order = 1;
flag_restrict_angle_values = 0;
riesz_transform_configurations_object = riesz_transform_object(size(mat_in), 1, number_of_scales, 1);
wavelet_transform_configurations_object = riesz_transform_object(size(mat_in), 0, number_of_scales, 1);
number_of_riesz_channels = riesz_transform_configurations_object.number_of_riesz_channels; %2 because this is 1st order
%(*) riesz transform object transfer functions filters (these also need fftshift):
riesz_transform_fourier_filter_x = riesz_transform_configurations_object.riesz_transform_filters{1};
riesz_transform_fourier_filter_y = riesz_transform_configurations_object.riesz_transform_filters{2};
%(*) riesz transfrom object prefilters (these are scrambled! the need fftshift to make common sense)
riesz_transform_prefilter_lowpass = riesz_transform_configurations_object.prefilter.filterLow;
riesz_transform_prefilter_highpass = riesz_transform_configurations_object.prefilter.filterHigh;
%(*) get wavelet type from riesz transform object:
wavelet_type = riesz_transform_configurations_object.wavelet_type; %isotropic only one implemented in generalized riesz transform toolbox
isotropic_wavelet_type = riesz_transform_configurations_object.isotropic_wavelet_type;
prefilter_type = 'simoncelli';
%%%%
%(*) decide if coefficients are in fourier or spatial domain:
flag_coefficients_in_fourier_0_or_spatial_domain_1 = 1;
%(*) downsample or not:
flag_downsample_or_not = 1;




%(3). PRE-CALCULATE LOWPASS AND HIGHPASS MASKS:
%(*) Initialize filters cell arrays:
maskHP_fft = cell(1,number_of_scales);
maskLP_fft = cell(1,number_of_scales);
maskHP = cell(1,number_of_scales);
maskLP = cell(1,number_of_scales);
current_frame_height = frame_height;
current_frame_width = frame_width;
current_image_size = [current_frame_height,current_frame_width];
% scale_counter = 1; %scale_counter is always 1 because here there's only 1 scale for this function
%                    %if i want to have more scales i should put that in a loop
isotropic_wavelet_type = 'papadakis';
flag_use_real_space_or_fourier_space_convolution = 1; %1 or 2
max_real_space_support_for_fourier_defined_filters = 7; %[pixels]
% MAKE SURE ALL FILTERS ARE THE SAME SITUATION AS FAR AS FFTSHIFT.
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
            maskLP{scale_counter} = filter_2D_lowpass_direct_rubinstein;
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
            low_pass = zeros(current_image_size);
            low_pass(center(1)-lodims(1):center(1)+lodims(1),center(2)-lodims(2):center(2)+lodims(2)) = ...
                abs(sqrt(1-filters_squared_sum_cropped));
            %Compute high pass residual:
            filters_squared_sum = filters_squared_sum + low_pass.^2;
            high_pass = abs(sqrt(1-filters_squared_sum));
            %If either dimension is even, this fixes some errors (UNDERSTAND WHY!!!):
            number_of_radial_filters = numel(radial_filters);
            if (mod(current_image_size(1),2) == 0) %even
                for k = 1:number_of_radial_filters
                    temp = radial_filters{k};
                    temp(1,:) = 0;
                    radial_filters{k} = temp;
                end
                high_pass(1,:) = 1;
                low_pass(1,:) = 0;
            end
            if (mod(current_image_size(2),2) == 0)
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
    end %isotropic-wavelet type switch statement end
    
    %if isotropic_wavelet_type DOESN'T EQUAL compact_rubinstein:
    if strcmp(isotropic_wavelet_type,'compact_rubinstein')~=1
        %IFFT to get real space impulse response (and cut impulse response
        %up to most pixels you allowed in max_real_space_support_for_fourier_defined_filters:
        temp_highpass = ifft2(maskHP_fft{scale_counter});
        temp_lowpass = ifft2(maskLP_fft{scale_counter});
        proper_indices_rows = current_frame_height/2+1-max_real_space_support_for_fourier_defined_filters : current_frame_height/2+1+max_real_space_support_for_fourier_defined_filters;
        proper_indices_columns = current_frame_width/2+1-max_real_space_support_for_fourier_defined_filters : current_frame_width/2+1+max_real_space_support_for_fourier_defined_filters;
        maskHP{scale_counter} = temp_highpass(proper_indices_rows,proper_indices_columns);
        maskLP{scale_counter} = temp_lowpass(proper_indices_rows,proper_indices_columns);
    end
    
end

%Create double (3D) copies of the masks to allow efficient multiplying of the 3D riesz coefficients:
%again- the reason is that the riesz transform number of channels in MY CASE is 2 (R1,R2) so i simply 
%double copy the filters in the 3rd dimension to allow for efficient multiplication
maskHP_3D = cell(1,number_of_scales);
maskLP_3D = cell(1,number_of_scales);
maskHP_3D_fft = cell(1,number_of_scales);
maskLP_3D_fft = cell(1,number_of_scales);
for scale_counter = 1:number_of_scales
    maskHP_3D{scale_counter} = repmat(maskHP{scale_counter},1,1,2);
    maskLP_3D{scale_counter} = repmat(maskLP{scale_counter},1,1,2);
    maskHP_3D_fft{scale_counter} = repmat(maskHP_fft{scale_counter}(:,:,1),1,1,2);
    maskLP_3D_fft{scale_counter} = repmat(maskLP_fft{scale_counter}(:,:,1),1,1,2);
end

figure; 
imagesc(abs(maskLP_fft{1}(:,:,1)));
imagesc(abs(maskLP_fft{1}))
imagesc(fftshift(abs(maskLP_fft{4})))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%COMPUTE NORMALIZATION COEFFICIENTS USING DELTA FUNCTION:
%these are computed using the isotropic wavelet type in the riesz_transform_configurations_object
%if i am using one of the following instead: compact_rubinstein, radial_rubinstein, radial_rubinstein_smoothed
%then i need to analyze the delta image explicitly and not using multiscale_riesz_analysis
%multiscale_riesz_analysis also uses prefilters which i don't nesceserily want!
%(*) get delta image:
delta_image = zeros(mat_in_size);
delta_image(1) = 1;
delta_image_fft = fft2(delta_image);
% % % %(*) get multiscale Riesz analysis of the delta_image. i don't really
% % % %remember what multiscale_riesz_analysis absolutely does so keep an eye on it
% % % noise_riesz_wavelet_cell_array = multiscale_riesz_analysis(delta_image, riesz_transform_configurations_object);
% % % stdNoiseRiesz = ones(length(noise_riesz_wavelet_cell_array), 1);
% % % for scale_counter = 1:length(noise_riesz_wavelet_cell_array)
% % %     tmp = noise_riesz_wavelet_cell_array{scale_counter}(:,:,1); %normalization only acording to first channel
% % %     stdNoiseRiesz(scale_counter) = std(tmp(:));
% % % end
% % % %(*)get multiscale Wavelet analysis for normalization:
% % % noise_wavelet_cell_array = multiscale_riesz_analysis(delta_image, wavelet_transform_configurations_object);
% % % stdNoiseWav = ones(length(noise_wavelet_cell_array), 1);
% % % for scale_counter = 1:length(noise_wavelet_cell_array)
% % %     stdNoiseWav(scale_counter) = std(noise_wavelet_cell_array{scale_counter}(:));
% % % end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(A). PREFILTER:
flag_use_perfilter = 1; %1 use prefilter, 0 don't
mat_in_fft = fft2(mat_in);
if flag_use_perfilter == 1
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(B). PERFORM RIESZ-TRANSFORM (fourier space multiplication) TO CURRENT (PREFILTERED) IMAGE:
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
%(C). APPLY WAVELET DECOMPOSITION TO RIESZ COEFFICIENTS (R1,R2)!:
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
%(D). APPLY WAVELET DECOMPOSITION TO ORIGINAL IMAGE (I)!:         
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















