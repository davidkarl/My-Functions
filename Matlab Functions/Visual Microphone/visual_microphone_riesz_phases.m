%visual microphone riesz phase:


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
    
%(3). Wavelet Transform:
%(*). usually, the wavelet transform, even when applied to 2D images, is 1
%     in nature. usually one chooses a 1D wavelet and then in the 1D case one
%     gets [lowpass,highpass] and in the 2D case one gets 
%     [lowpass, horizontal_highpass,vertical_highpass,diagonal_highpass]
%(*). the good thing about the pyramidial (complex steerable or riesz)
%     decomposition is that the decomposition is radial and maybe angular. this
%     allows us to easily get a [lowpass,highpass] decomposition like the 1D
%     case but for 2D images.
%(*). The point is - how to choose those filters? - what is basically
%     implemented here is the cool trick i have used to make 1D filters into 2D
%     which is the McClellan Transform. what you do is take a 1D filter (or more specifically, a wavelet), which
%     is easy to design, and then use a known algorithm to approximate that 1D filter in 2D. 
%
    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VIDEO PARAMETERS FOR ACTUAL VIDEO: 

%Get input variables:
%Video and image files:
video_file_name = 'Chips1-2200Hz-Mary_Had-input.avi';
video_file_name = 'image_warps.avi';
test_image_file_name = 'barbara.tif';
output_directory = 'C:\Users\master\Desktop\matlab';
output_file_name = 'final_phase_amplified_movie';
%Get shifts fid:
shifts_file_name = 'xy_shifts.bin';
fid_shifts = fopen(shifts_file_name,'r');

%Phase magnification and Pyramidial Decomposition stuff:
phase_magnification_factor = 15*2;    
number_of_scales = 4; %number of scales to actually use
low_cutoff_frequency = 3;
high_cutoff_frequency = 10;

%coefficients calculation methods:
flag_coefficients_in_fourier_0_or_spatial_domain_1 = 1;
flag_downsample_or_not = 1;

%Individual Image analysis:
flag_analyze_every_channel_seperately_or_combine_or_single = 3; %1=seperataely, 2=combine, 3=single
flag_combine_method = 1; %1=average, 2=NTSC intensity
chrom_attenuation = 0;
reference_frame_index = 1;
transition_width = 1; 

%WAVELETS STUFF:
%(*) wavelet types (isotropic or "slices" like CSP):
wavelet_types = {'isotropic'};
%(*) wavelet types (those in the first row are all basically the same - radial brick wall):
isotropic_wavelet_types = {'simoncelli','shannon','aldroubi','papadakis','meyer','ward',...
                           'radial_rubinstein', 'radial_rubinstein_smoothed','compact_rubinstein'};
%(*) prefilter types:
prefilter_types = {'shannon','simoncelli','empirical','none'};
prefilter_type = 'none';


%Read Video Parameters: 
video_reader_object = VideoReader(video_file_name);
frame_height = video_reader_object.Height;
frame_width = video_reader_object.Width;
bits_per_pixels = video_reader_object.BitsPerPixel;
video_format = video_reader_object.VideoFormat;
[~, writeTag, ~] = fileparts(video_file_name);

%Frequencies[Hz]:
Fs = video_reader_object.FrameRate;
number_of_frames = floor(video_reader_object.Duration*Fs);

%Read frame:
mat_in = video_reader_object.readFrame();
mat_in = double(mat_in);
%Get only one Channel:
mat_in = mat_in(:,:,1);
mat_in_fft = fft2(mat_in);
mat_in_ps = abs(mat_in_fft).^2;
number_of_channels = size(mat_in,3);
% %Read ALL frames if you want:
% video_frames = video_reader_object.read();


%Get Output File stuff:
% video_writer_object = VideoWriter(output_file_name, 'Motion JPEG AVI');
video_writer_object = VideoWriter(output_file_name, 'Uncompressed AVI');
% video_writer_object.Quality = 90;
video_writer_object.FrameRate = Fs;
video_writer_object.open;


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
dyadic_size_vecs_cell = cell(number_of_scales,1);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Temporal Filter parameters (for phase temporal filtering):

low_cutoff_frequency_normalized = low_cutoff_frequency/(Fs/2);
high_cutoff_frequency_normalized = high_cutoff_frequency/(Fs/2);
%(1). FIR:
temporal_filter_function = @FIRWindowBP; %
%(2). IIR (IMPLEMENT FUNCTION):
temporal_filter_order = 2;
[b,a] = butter(temporal_filter_order,[low_cutoff_frequency_normalized,high_cutoff_frequency_normalized],'bandpass');
IIR_filter_object = dsp.IIRFilter;
IIR_filter_object.Numerator = b;
IIR_filter_object.Denominator = a;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% PHASE ANALYSIS %%%%%%%%%%%%


%(*) get prefilter:
if strcmp(prefilter_type,'none')
    prefilter_fft = ones(frame_height,frame_width);
end

%RIESZ TRANSFORM STUFF:
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

%Wavelet Filters:
[ filter_1D_lowpass_coefficients_rubinstein, ...
  filter_1D_highpass_coefficients_rubinstein, ...
  chebychev_polynomial_lowpass_rubinstein, ...
  chebychev_polynomial_highpass_rubinstein, ...
  McClellan_transform_matrix_rubinstein, ...
  filter_2D_lowpass_direct_rubinstein, ...
  filter_2D_highpass_direct_rubinstein ] = get_filters_for_riesz_pyramid();
filter_2D_lowpass_direct_rubinstein_fft = fft2(filter_2D_lowpass_direct_rubinstein,frame_height,frame_width);
filter_2D_highpass_direct_rubinstein_fft = fft2(filter_2D_highpass_direct_rubinstein,frame_height,frame_width);

%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%GET REFERENCE PYRAMID:
%(1). Get reference frame and convert it to ntc space:
reference_frame = mat_in;
if flag_analyze_every_channel_seperately_or_combine_or_single == 2
    if flag_combine_method == 1
        reference_frame = sum(reference_frame,3);
    elseif flag_combine_method == 2
        reference_frame = rgb2ntsc(reference_frame);
        reference_frame = squeeze(reference_frame(:,:,1));
    end
elseif flag_analyze_every_channel_seperately_or_combine_or_single == 3
    reference_frame = reference_frame(:,:,1);
end


%(2). Get Phase 
%(i will decide to use only fft domain from now on):
%(REMEMBER I CAN COMBINE! THE PREFILTER & RIESZ TRANSFORM):
%get fft:
previous_frame_fft = fft2(previous_frame);
%prefilter:
previous_frame_fft_filtered = previous_frame_fft .* prefilter_fft;
%get riesz transform (fft):
previous_frame_riesz_x_fft = previous_frame_fft.*riesz_transform_fourier_filter_x;
previous_frame_riesz_y_fft = previous_frame_fft.*riesz_transform_fourier_filter_y;
%get real space riesz channels:
previous_frame_riesz_x = ifft2(previous_frame_riesz_x_fft);
previous_frame_riesz_y = ifft2(previous_frame_riesz_y_fft);
%take real part (IS THIS LEGIT?):
previous_frame_riesz_x = real(previous_frame_riesz_x);
previous_frame_riesz_y = real(previous_frame_riesz_y);
%get phases and amplitude:
previous_phase = atan( sqrt(previous_frame_riesz_x.^2+previous_frame_riesz_y.^2) ./ previous_frame );
previous_amplitude = sqrt(previous_frame_riesz_x.^2 + previous_frame_riesz_y.^2+previous_frame.^2);


%(3). Temporal Filtering (see if it's better to filter before or after amplitude weighing):
phases_spread = previous_phase(:)';
temporally_filtered_phases = reshape(phases_spread(:),frame_height,frame_width);

%(3). Post Process Phases (Spatially linear/non-linear filter):
%probably i don't have to recalculate the filtered amplitude everytime and i can use an averaged quantity:
%get amplitude smoothing filter:
amplitude_smoothing_filter = fspecial('gaussian',5,5);
%spatially filter amplitude
spatially_filtered_previous_amplitude = conv2(previous_amplitude,amplitude_smoothing_filter,'same');
%spatially filter amplitude.*phases:
spatially_filtered_previous_amplitude_times_phases = conv2(temporally_filtered_phases,amplitude_smoothing_filter,'same');
%Amplitude Weigh Phases:
previous_spatially_temporally_filtered_phases = (spatially_filtered_previous_amplitude_times_phases)./(spatially_filtered_previous_amplitude);




%There's alot to figure out - 
%(1). at what order to do things?
%(2). when and how to filter exactly
%(3). how important is it to do pyramidially decompose through all the scales
%(4). how to get the phase? whether quanterion vectors or how i do it here
%(5). whether it's important to rotate the monogenic signal 
%(6). whether to and how to smooth the phases and how to use amplitue weighing if at all
%(7). which wavelets to choose for decomposition
%(8). which prefilter to use?
%(9). whether to unwrap? is it necesary when using differential phase from previous image
%(10). remember fft2 (i think) operates on every slice in the 3rd dimension in parallel


%phase difference
phase_difference_over_time = zeros(frame_height,frame_width,number_of_frames);
for frame_counter = reference_frame_index+1:number_of_frames
    tic
    
    %Read frame:
    current_frame = video_reader_object.readFrame();
    current_frame = double(current_frame);
    
    if flag_analyze_every_channel_seperately_or_combine_or_single == 2
        if flag_combine_method == 1
            current_frame = sum(current_frame,3);
        elseif flag_combine_method == 2
            current_frame = rgb2ntsc(current_frame);
            current_frame = squeeze(current_frame(:,:,1));
        end
    elseif flag_analyze_every_channel_seperately_or_combine_or_single == 3
        current_frame = current_frame(:,:,1);
    end
    
    
    %(2). Get Riesz TRIPLET:
    %(i will decide to use only fft domain from now on):
    %(REMEMBER I CAN COMBINE! THE PREFILTER & RIESZ TRANSFORM):
    %get fft:
    current_frame_fft = fft2(current_frame);
    %prefilter:
    current_frame_fft_filtered = current_frame_fft .* prefilter_fft;
    %get riesz transform (fft):
    current_frame_riesz_x_fft = current_frame_fft.*riesz_transform_fourier_filter_x;
    current_frame_riesz_y_fft = current_frame_fft.*riesz_transform_fourier_filter_y;
    %get real space riesz channels:
    current_frame_riesz_x = ifft2(current_frame_riesz_x_fft);
    current_frame_riesz_y = ifft2(current_frame_riesz_y_fft);
    %take real part (IS THIS LEGIT?):
    current_frame_riesz_x = real(current_frame_riesz_x);
    current_frame_riesz_y = real(current_frame_riesz_y);
    
    
    
    %(3). Get Phases:
    
    %(****) Straight Forward Way:
    current_phase = atan( sqrt(current_frame_riesz_x.^2+current_frame_riesz_y.^2) ./ current_frame );
    current_amplitude = sqrt(current_frame_riesz_x.^2 + current_frame_riesz_y.^2+current_frame.^2);
    
    %(3). Temporal Filtering (see if it's better to filter before or after amplitude weighing):
    current_phases_spread = current_phase(:)';
    %some temporal filtering (not for now)
    current_temporally_filtered_phases = reshape(current_phases_spread(:),frame_height,frame_width);
    
    %(3). Post Process Phases (Spatially linear/non-linear filter):
    %probably i don't have to recalculate the filtered amplitude everytime and i can use an averaged quantity:
    %spatially filter amplitude
    current_spatially_filtered_amplitude = conv2(current_amplitude,amplitude_smoothing_filter,'same');
    %spatially filter amplitude.*phases:
    current_spatially_filtered_phases = conv2(current_temporally_filtered_phases,amplitude_smoothing_filter,'same');
    current_spatially_filtered_amplitude_times_phases = current_spatially_filtered_phases .* current_spatially_filtered_amplitude;
    %Amplitude Weigh Phases:
    current_spatially_temporally_filtered_phases = (current_spatially_filtered_amplitude_times_phases)./(current_spatially_filtered_amplitude);
%     current_spatially_temporally_filtered_phases = current_temporally_filtered_phases;
%     current_spatially_temporally_filtered_phases = unwrap(current_spatially_temporally_filtered_phases);
   
    %phase difference
    phase_difference_over_time(:,:,frame_counter) = current_spatially_temporally_filtered_phases - previous_spatially_temporally_filtered_phases;

                                                                        
    %assign previous phase:
    previous_spatially_temporally_filtered_phases = current_spatially_temporally_filtered_phases;
    
    toc
    
    shiftx = fread(fid_shifts,frame_height*frame_width,'double');
    shifty = fread(fid_shifts,frame_height*frame_width,'double');
    shiftx = reshape(shiftx,frame_height,frame_width);
    shifty = reshape(shifty,frame_height,frame_width);
    shift_tot = shiftx.^2 + shifty.^2;
    
%     imagesc(phase_difference_over_time(10:end-10,10:end-10,frame_counter));

    subplot(2,1,1);
    imagesc(current_spatially_temporally_filtered_phases(10:end-10,10:end-10));
%     imagesc(phase_difference_over_time(:,:,frame_counter));
%     imagesc(phase_difference_over_time(20:end-20,20:end-20,frame_counter));
    subplot(2,1,2);
    imagesc(shift_tot);

%     imagesc(current_spatially_temporally_filtered_phases + 0.5*0.01*shift_tot);
    
    pause(0.3);
    
    
end





% function [spatially_smooth_temporally_filtered_phase] = ...
%                         amplitude_weighted_blur(temporally_filtered_phase,amplitude,blur_kernel)
%     %spatially blurs phase, weighted by amplitude:
%     
%     %MAYBE I CAN GET AMPLITUDE AND THEREFORE NOT CALCULATE AMPLITUDE
%     %CONVOLUTION IN DENOMINATOR IN EVERY FRAME???
%     
%     %MAYBE USE IRREGAULR SAMPLING LIFTING FILTERING STRCUTURE AND
%     %INTERPOLATE LOW AMPLITUDE PARTS OF THE IMAGE
%     denominator = conv2(amplitude, blur_kernel);
%     numerator = conv2(temporally_filtered_phase.*amplitude, blur_kernel);
%     spatiialy_smooth_temporally_filtered_phase = numerator./denominator;
% end








% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %(B). PERFORM RIESZ-TRANSFORM (fourier space multiplication) TO CURRENT (PREFILTERED) IMAGE:
% %(1). to mat_in:
% %fourier space:
% mat_in_riesz_coefficients_fft_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
% mat_in_riesz_coefficients_fft_3D(:,:,1) = mat_in_prefiltered_fft.*riesz_transform_fourier_filter_x;
% mat_in_riesz_coefficients_fft_3D(:,:,2) = mat_in_prefiltered_fft.*riesz_transform_fourier_filter_y;
% %direct space:
% mat_in_riesz_coefficients_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
% mat_in_riesz_coefficients_3D(:,:,1) = real(ifft2(mat_in_riesz_coefficients_fft_3D(:,:,1)));
% mat_in_riesz_coefficients_3D(:,:,2) = real(ifft2(mat_in_riesz_coefficients_fft_3D(:,:,2)));
% %(2). to delta_image:
% %fourier space:
% delta_image_riesz_coefficients_fft_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
% delta_image_riesz_coefficients_fft_3D(:,:,1) = delta_image_prefiltered_fft.*riesz_transform_fourier_filter_x;
% delta_image_riesz_coefficients_fft_3D(:,:,2) = delta_image_prefiltered_fft.*riesz_transform_fourier_filter_y;
% %direct space:
% delta_image_riesz_coefficients_3D = zeros(size(mat_in, 1), size(mat_in, 2), number_of_riesz_channels);
% delta_image_riesz_coefficients_3D(:,:,1) = real(ifft2(delta_image_riesz_coefficients_fft_3D(:,:,1)));
% delta_image_riesz_coefficients_3D(:,:,2) = real(ifft2(delta_image_riesz_coefficients_fft_3D(:,:,2)));
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %(C). APPLY WAVELET DECOMPOSITION TO RIESZ COEFFICIENTS (R1,R2)!:
% mat_in_riesz_wavelet_cell_array_reference = cell(1, number_of_scales+1);
% delta_image_riesz_wavelet_cell_array_reference = cell(1, number_of_scales+1);
% mat_in_riesz_wavelet_lowpass_cell_array_reference = cell(1,number_of_scales+2);
% delta_image_riesz_wavelet_lowpass_cell_array_reference = cell(1,number_of_scales+2);
% %(*) Assign raw prefiltered image to first cell of appropriate cell array:
% mat_in_riesz_wavelet_lowpass_cell_array_reference{1} = mat_in_riesz_coefficients_3D;
% delta_image_riesz_wavelet_lowpass_cell_array_reference{1} = delta_image_riesz_coefficients_3D;
% for riesz_channel_counter = 1:number_of_riesz_channels,
%     %Compute fft2 because before we used ifft2 in riesz_coefficients_matrix_of_lowpassed_image_3D calculation:
%     highpass_wavelet_coefficients = cell(1, number_of_scales);
%     for scale_counter = 1:number_of_scales
%         
%         %Lowpass and Highpass (add highpass to wavelet cell array and use lowpass for coarser scales):
%         if flag_use_real_space_or_fourier_space_convolution == 2
%             %(1) mat_in riesz-wavelet:
%             mat_in_riesz_coefficients_fft_3D_HP = mat_in_riesz_coefficients_fft_3D.*maskHP_3D_fft{scale_counter};
%             mat_in_riesz_coefficients_fft_3D = mat_in_riesz_coefficients_fft_3D.*maskLP_3D_fft{scale_counter};
%             %(2) delta_image riesz-wavelet:
%             delta_image_riesz_coefficients_fft_3D_HP = delta_image_riesz_coefficients_fft_3D.*maskHP_3D_fft{scale_counter};
%             delta_image_riesz_coefficients_fft_3D = delta_image_riesz_coefficients_fft_3D.*maskLP_3D_fft{scale_counter};            
%         elseif flag_use_real_space_or_fourier_space_convolution == 1
%             
%             %SEE IF CONV2 CAN DO 2D CONVOLUTION FOR EACH MATRIX ALONG THE THIRD DIMENSION:
%             %(1) mat_in riesz-wavelet:
%             mat_in_riesz_coefficients_3D_HP(:,:,1) = conv2(mat_in_riesz_coefficients_3D(:,:,1),maskHP{scale_counter});
%             mat_in_riesz_coefficients_3D(:,:,1) = conv2(mat_in_riesz_coefficients_3D(:,:,1),maskLP{scale_counter});
%             mat_in_riesz_coefficients_3D_HP(:,:,2) = conv2(mat_in_riesz_coefficients_3D(:,:,2),maskHP{scale_counter});
%             mat_in_riesz_coefficients_3D(:,:,2) = conv2(mat_in_riesz_coefficients_3D(:,:,2),maskLP{scale_counter});
%             %(2) delta_image riesz-wavelet:
%             delta_image_riesz_coefficients_3D_HP(:,:,1) = conv2(delta_image_riesz_coefficients_3D(:,:,1),maskHP{scale_counter});
%             delta_image_riesz_coefficients_3D(:,:,1) = conv2(delta_image_riesz_coefficients_3D(:,:,1),maskLP{scale_counter});
%             delta_image_riesz_coefficients_3D_HP(:,:,2) = conv2(delta_image_riesz_coefficients_3D(:,:,2),maskHP{scale_counter});
%             delta_image_riesz_coefficients_3D(:,:,2) = conv2(delta_image_riesz_coefficients_3D(:,:,2),maskLP{scale_counter});
%         end
%         
%         
%         %Downsample:
%         if flag_downsample_or_not == 1
%             if flag_use_real_space_or_fourier_space_convolution == 2
%                 %fourier space downsampling == spectrum folding:
%                 %(1). mat in: 
%                 c2 = size(mat_in_riesz_coefficients_fft_3D,2)/2;
%                 c1 = size(mat_in_riesz_coefficients_fft_3D,1)/2;
%                 mat_in_riesz_coefficients_fft_3D = ...
%                     0.25*( mat_in_riesz_coefficients_fft_3D(1:c1, 1:c2) + ...
%                            mat_in_riesz_coefficients_fft_3D((1:c1)+c1, 1:c2) + ...
%                            mat_in_riesz_coefficients_fft_3D((1:c1) + c1, (1:c2) +c2) + ...
%                            mat_in_riesz_coefficients_fft_3D(1:c1, (1:c2) +c2)...
%                          );
%                 %(2). delta image:
%                 delta_image_riesz_coefficients_fft_3D = ...
%                     0.25*( delta_image_riesz_coefficients_fft_3D(1:c1, 1:c2) + ...
%                            delta_image_riesz_coefficients_fft_3D((1:c1)+c1, 1:c2) + ...
%                            delta_image_riesz_coefficients_fft_3D((1:c1) + c1, (1:c2) +c2) + ...
%                            delta_image_riesz_coefficients_fft_3D(1:c1, (1:c2) +c2)...
%                          ); 
%             elseif flag_use_real_space_or_fourier_space_convolution == 1
%                 %direct space downsampling:
%                 mat_in_riesz_coefficients_3D = mat_in_riesz_coefficients_3D(1:2:end,1:2:end);
%                 delta_image_riesz_coefficients_3D = delta_image_riesz_coefficients_3D(1:2:end,1:2:end);
%             end
%         else
%             %DO NOT downsample (only filter again and again)
%         end
%         
%         %Assign calculated HIGHPASS (detailed) coefficients to wavelet cell array:
%         if flag_use_real_space_or_fourier_space_convolution == 2
%             mat_in_riesz_wavelet_cell_array_reference{scale_counter} = ifft2(mat_in_riesz_coefficients_fft_3D_HP);
%             delta_image_riesz_wavelet_cell_array_reference{scale_counter} = ifft2(delta_image_riesz_coefficients_fft_3D_HP);
%         elseif flag_use_real_space_or_fourier_space_convolution == 1
%             mat_in_riesz_wavelet_cell_array_reference{scale_counter} = mat_in_riesz_coefficients_3D_HP;
%             delta_image_riesz_wavelet_cell_array_reference{scale_counter} = delta_image_riesz_coefficients_3D_HP;
%         end
%         
%         %Keep track of LOWPASS part too (wavelet is a cascade of highpass parts):
%         %(*) the first cell is the raw prefiltered image, the second is the
%         %    lowpass filtered downsampled image etc'
%         if flag_use_real_space_or_fourier_space_convolution == 2
%             mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter+1} = ifft2(mat_in_riesz_coefficients_fft_3D);
%             delta_image_riesz_wavelet_lowpass_cell_array_reference{scale_counter+1} = ifft2(delta_image_riesz_coefficients_fft_3D);
%         elseif flag_use_real_space_or_fourier_space_convolution == 1
%             mat_in_riesz_wavelet_lowpass_cell_array_reference{scale_counter+1} = mat_in_riesz_coefficients_3D;
%             delta_image_riesz_wavelet_lowpass_cell_array_reference{scale_counter+1} = delta_image_riesz_coefficients_3D;
%         end
%         
%     end %END SCALES LOOP
%     
%     %Assign the appropriate (VERY MUCH DOWNSAMPLED) LOW_PASS RESIDUAL:
%     if flag_use_real_space_or_fourier_space_convolution == 2
%         mat_in_riesz_wavelet_cell_array_reference{number_of_scales+1} = ifft2(mat_in_riesz_coefficients_fft_3D);
%         delta_image_riesz_wavelet_cell_array_reference{number_of_scales+1} = ifft2(delta_image_riesz_coefficients_fft_3D);
%     elseif flag_use_real_space_or_fourier_space_convolution == 1
%         mat_in_riesz_wavelet_cell_array_reference{number_of_scales+1} = mat_in_riesz_coefficients_3D_HP;
%         delta_image_riesz_wavelet_cell_array_reference{number_of_scales+1} = delta_image_riesz_coefficients_3D;
%     end
%     
% end %riesz channels loop
% 
% %Get normalization constants for riesz channels:
% stdNoiseRiesz = ones(length(noise_riesz_wavelet_cell_array), 1);
% stdNoiseRiesz_lowpass = ones(length(noise_riesz_wavelet_cell_array), 1);
% for scale_counter = 1:length(delta_image_riesz_wavelet_cell_array_reference)
%     stdNoiseRiesz(scale_counter) = std(delta_image_riesz_wavelet_cell_array_reference{scale_counter}(:,:,1)); %normalization only acording to first channel
% end
% for scale_counter = 1:length(delta_image_riesz_wavelet_lowpass_cell_array_reference)
%     stdNoiseRiesz_lowpass(scale_counter) = std(delta_image_riesz_wavelet_lowpass_cell_array_reference{scale_counter}(:,:,1));
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %(D). APPLY WAVELET DECOMPOSITION TO ORIGINAL IMAGE (I)!:         
% % mat_in_prefiltered_fft = fft2(mat_in); %THIS IS THE ORIGINAL AS FOUND IN riesz_full_monogenic_analysis, seems wrong!!!
% mat_in_wavelet_cell_array_reference = cell(1, number_of_levels+1);     
% delta_image_wavelet_cell_array_reference = cell(1, number_of_levels+1);
% mat_in_wavelet_lowpass_cell_array_reference = cell(1,number_of_scales+2);
% delta_image_wavelet_lowpass_cell_array_reference = cell(1,number_of_scales+2);
% %(*) Assign raw prefiltered image to first cell of appropriate cell array:
% mat_in_wavelet_lowpass_cell_array_reference{1} = mat_in_prefiltered;
% delta_image_wavelet_lowpass_cell_array_reference{1} = delta_image_prefiltered;
% for scale_counter = 1:number_of_levels
%       
%     %high pass and lowpass image:
%     if flag_use_real_space_or_fourier_space_convolution == 2
%         %(1). mat in
%         mat_in_prefiltered_fft_HP = mat_in_prefiltered_fft.*maskHP_3D_fft{scale_counter};
%         mat_in_prefiltered_fft = mat_in_prefiltered_fft.*maskLP_3D_fft{scale_counter};
%         %(2). delta image:
%         delta_image_prefiltered_fft_HP = delta_image_prefiltered_fft.*maskHP_3D_fft{scale_counter};
%         delta_image_prefiltered_fft = delta_image_prefiltered_fft.*maskLP_3D_fft{scale_counter};
%     else
%         %(1). mat in
%         mat_in_prefiltered_HP = conv2(mat_in_prefiltered,maskHP_fft{scale_counter});
%         mat_in_prefiltered = conv2(mat_in_prefiltered,maskLP_fft{scale_counter});
%         %(2). delta image:
%         delta_image_prefiltered_HP = conv2(delta_image_prefiltered,maskHP_fft{scale_counter});
%         delta_image_prefiltered = conv2(delta_image_prefiltered,maskHP_fft{scale_counter});
%     end
%     
%     %Downsample:
%     if flag_downsample_or_not == 1
%         if flag_use_real_space_or_fourier_space_convolution == 2
%             c2 = size(mat_in_prefiltered_fft,2)/2;
%             c1 = size(mat_in_prefiltered_fft,1)/2;
%             %(1). mat in:
%             mat_in_prefiltered_fft = 0.25*(mat_in_prefiltered_fft(1:c1, 1:c2) + mat_in_prefiltered_fft((1:c1)+c1, 1:c2) + ...
%                 mat_in_prefiltered_fft((1:c1) + c1, (1:c2) +c2) + mat_in_prefiltered_fft(1:c1, (1:c2) +c2));
%             %(2). delta image:
%             delta_image_prefiltered_fft = 0.25*(delta_image_prefiltered_fft(1:c1, 1:c2) + delta_image_prefiltered_fft((1:c1)+c1, 1:c2) + ...
%                 delta_image_prefiltered_fft((1:c1) + c1, (1:c2) +c2) + delta_image_prefiltered_fft(1:c1, (1:c2) +c2)); 
%         elseif flag_use_real_space_or_fourier_space_convolution == 1
%             %(1). mat in:
%             mat_in_prefiltered = mat_in_prefiltered(1:2:end);
%             %(2). delta image:
%             delta_image_prefiltered = delta_image_prefiltered(1:2:end);
%         end
%                  
%     else
%         %do NOT downsample (only filter again and again)
%     end
%     
%     %Assign proper HIGHPASS coefficients (direct or fourier space) into the wavelet cell array:
%     if flag_use_real_space_or_fourier_space_convolution == 2
%         mat_in_wavelet_cell_array_reference{scale_counter} = ifft2(mat_in_prefiltered_fft_HP);
%         delta_image_wavelet_cell_array_reference{scale_counter} = ifft2(delta_image_prefiltered_fft_HP);
%     else
%         mat_in_wavelet_cell_array_reference{scale_counter} = mat_in_prefiltered_HP;
%         delta_image_wavelet_cell_array_reference{scale_counter} = delta_image_prefiltered_HP;
%     end
%     
%     %Assign proper LOWPASS coefficients into the wavelet cell array:
%     if flag_use_real_space_or_fourier_space_convolution == 2
%         mat_in_wavelet_lowpass_cell_array_reference{scale_counter+1} = ifft2(mat_in_prefiltered_fft);
%         delta_image_wavelet_lowpass_cell_array_reference{scale_counter+1} = ifft2(delta_image_prefiltered_fft);
%     else
%         mat_in_wavelet_lowpass_cell_array_reference{scale_counter+1} = mat_in_prefiltered_fft;
%         delta_image_wavelet_lowpass_cell_array_reference{scale_counter+1} = delta_image_prefiltered_fft;
%     end
%     
% end %LEVELS/SCALES LOOP
% %Assign lowpass residual:
% if flag_use_real_space_or_fourier_space_convolution == 2
%     mat_in_wavelet_cell_array_reference{number_of_scales+1} = ifft2(mat_in_prefiltered_fft);
%     delta_image_wavelet_cell_array_reference{number_of_scales+1} = ifft2(delta_image_prefiltered_fft);
% else
%     mat_in_wavelet_cell_array_reference{number_of_scales+1} = mat_in_prefiltered;
%     delta_image_wavelet_cell_array_reference{number_of_scales+1} = delta_image_prefiltered;
% end 
% 
% %Get normalization constant for wavelet transform:
% stdNoiseWav = ones(length(delta_image_wavelet_cell_array_reference), 1);
% stdNoiseWav_lowpass = ones(length(delta_image_wavelet_cell_array_reference), 1);
% for scale_counter = 1:length(delta_image_wavelet_cell_array_reference)
%     stdNoiseWav(scale_counter) = std(delta_image_wavelet_cell_array_reference{scale_counter}(:));
% end
% for scale_counter = 1:length(delta_image_wavelet_lowpass_cell_array_reference)
%     stdNoiseWav_lowpass(scale_counter) = std(delta_image_wavelet_lowpass_cell_array_reference{scale_counter}(:));
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
% 
% 
% 































