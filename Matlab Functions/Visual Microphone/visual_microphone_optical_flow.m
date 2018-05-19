%visual microphone optical flow


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
 
%Get shifts fid:
shifts_file_name = 'xy_shifts.bin';
fid_shifts = fopen(shifts_file_name,'r');

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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GO OVER FRAMES AND DO OPTICAL FLOW:
flag_optical_flow_method = 4;
%(1).Lucas-Kanade parameters:
block_size = 5;
%(2).Horn-Schunk:
u_initial = zeros(frame_height,frame_width);
v_initial = zeros(frame_height,frame_width);
flag_display_flow = 0;
flag_display_image = 0;
alpha_smoothness_parameter = 1;
number_of_iterations = 100;
%(3).Chan-Vo:
block_size = 10; %must be even for now
flag_search_algorithm = 2;
flag_check_bidirectional = 1;
flag_do_median_filtering = 1;
%(4).phase-based optical flow:
number_of_velocity_vectors_along_x_axis = 0;
linearity_threshold = 0.05;
min_number_of_valid_velocity_components = 2;
%(5).Theory of Warping:
1;
%(6).Sub Pixel Motion Estimation:
opts.BlockSize = 10;
opts.SearchLimit = 20;
%(7).Matlab opticalFlowLK
opticFlowLK = opticalFlowLK('NoiseThreshold',0.009);
%(8).Matlab opticalFlowFarneback
opticFlowFarnesback = opticalFlowFarneback;
%(9).Matlab opticalFlowHS
opticFlowHS = opticalFlowHS;
%(10).Matlab opticalFlowLKDoG
opticFlowLKDoG = opticalFlowLKDoG('NumFrames',3);



previous_frame = double(video_frames(:,:,:,1));
previous_frame = squeeze(previous_frame(:,:,1));
for frame_counter = 2:number_of_frames
    %get frame:
    current_frame = double(video_frames(:,:,:,frame_counter));
    current_frame = squeeze(current_frame(:,:,1));
    
    %get shifts:
    shifts_x = fread(fid_shifts,frame_height*frame_width,'double');
    shifts_y = fread(fid_shifts,frame_height*frame_width,'double');
    shifts_x = reshape(shifts_x,frame_height,frame_width);
    shifts_y = reshape(shifts_y,frame_height,frame_width);
    shifts_tot = shifts_x.^2 + shifts_y.^2;
    
    
    %calculate optical flow field:
    if flag_optical_flow_method == 1
        tic
        [u,v] = optical_flow_Lucas_Kanade(previous_frame,current_frame,block_size);
        v_tot = u.^2+v.^2;
        toc
    elseif flag_optical_flow_method == 2
        tic
        [u, v] = optical_flow_Horn_Schunck(previous_frame, ...
                                           current_frame, ...
                                           alpha_smoothness_parameter, ...
                                           number_of_iterations, ...
                                           u_initial, ...
                                           v_initial, ...
                                           flag_display_flow, ...
                                           flag_display_image);
         v_tot = u.^2+v.^2;
%          u_initial = u;
%          v_initial = v;
         toc
    elseif flag_optical_flow_method == 3
        tic
        [u, v] = optical_flow_Chan_Vo(previous_frame, current_frame, ...
                                                block_size, ...
                                                flag_do_median_filtering, ...
                                                flag_search_algorithm, ...
                                                flag_check_bidirectional);
         v_tot = u.^2 + v.^2;
         toc
    elseif flag_optical_flow_method == 4
        Image_sequence = cat(3,previous_frame,current_frame);
        tic
        [Optical_flow_over_time,point_where_gaussian_evelope_is_10_percent] = ...
                                                optical_flow_phase_based(Image_sequence, ...
                                                                         number_of_velocity_vectors_along_x_axis, ...
                                                                         linearity_threshold, ...
                                                                         min_number_of_valid_velocity_components);
        u =  Optical_flow_over_time(:,:,1);
        v =  Optical_flow_over_time(:,:,1);
        v_tot = u.^2 + v.^2;
        toc
    elseif flag_optical_flow_method == 5
        tic
        [u, v] = optic_flow_Theory_of_Warping2(previous_frame, current_frame);
        toc
        v_tot = u.^2 + v.^2;
    elseif flag_optical_flow_method == 6
        tic
        [u, v] = Motion_Est(current_frame, previous_frame, opts);
        toc
        v_tot = u.^2 + v.^2;
    elseif flag_optical_flow_method == 7
        tic
        flow = estimateFlow(opticFlowLK,current_frame); 
        u = flow.Vx;
        v = flow.Vy;
        v_tot = u.^2+v.^2;
        toc
    elseif flag_optical_flow_method == 8
        tic
        flow = estimateFlow(opticFlowFarnesback,current_frame); 
        u = flow.Vx;
        v = flow.Vy;
        v_tot = u.^2+v.^2;
        toc
    elseif flag_optical_flow_method == 9
        tic
        flow = estimateFlow(opticFlowHS,current_frame); 
        u = flow.Vx;
        v = flow.Vy;
        v_tot = u.^2+v.^2;
        toc
    elseif flag_optical_flow_method == 10
        tic
        flow = estimateFlow(opticFlowLKDoG,current_frame); 
        u = flow.Vx;
        v = flow.Vy;
        v_tot = u.^2+v.^2;
        toc
    end 
    
    %assign current frame to be previous frame for next calculation:
    previous_frame = current_frame;
    
%     figure(1);
%     subplot(2,1,1);
%     imagesc(v_tot(20:end-20,20:end-20));
% %     imagesc(v_tot);
%     colorbar;
%     subplot(2,1,2);
%     imagesc(shifts_tot);
%     colorbar;
%     drawnow;
    
    figure(1);
    subplot(3,1,1);
%     imagesc(u);
    imagesc(v_tot);
    colorbar;
    subplot(3,1,2);
    imagesc(v);
    colorbar;
    subplot(3,1,3);
    imagesc(shifts_tot);
    colorbar;
    drawnow;
    
    
    
%     pause(1);
end







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








