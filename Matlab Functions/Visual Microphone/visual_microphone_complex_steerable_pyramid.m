%visual microphone complex steerable pyramid:


%Get input variables:
video_file_name = 'Chips1-2200Hz-Mary_Had-input.avi';
video_file_name = 'image_warps.avi';
test_image_file_name = 'barbara.tif';
output_directory = 'C:\Users\master\Desktop\matlab';
low_cutoff_frequency = 100;
high_cutoff_frequency = 1000;
flag_use_temporally_filtered_phases = 1;
Fs = 2200;
phase_magnification_factor = 15*2;    
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

%Get Output File stuff:
output_file_name = 'final_phase_amplified_movie';
% video_writer_object = VideoWriter(output_file_name, 'Motion JPEG AVI');
video_writer_object = VideoWriter(output_file_name, 'Uncompressed AVI');
% video_writer_object.Quality = 90;
video_writer_object.FrameRate = Fs;
video_writer_object.open;


%Get reference frame for forward modeling and also structure tensor modeling:
reference_frame_index = 1;
mat_in_total_channels = squeeze(video_frames(:,:,:,reference_frame_index));
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
temporal_filter_order = 2;
%IMPLEMENT get_butterworth_filter_coefficients (in order to filter many
%channels and levels i need to put the phases into a matrix form).
[b,a] = butter(temporal_filter_order,[low_cutoff_frequency_normalized,high_cutoff_frequency_normalized],'bandpass');
IIR_filter_object = dsp.IIRFilter;
IIR_filter_object.Numerator = b;
IIR_filter_object.Denominator = a;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Complex Steerable Pyramid Stuff

%Parameters:
number_of_levels = get_max_complex_steerable_pyramid_height(video_frames(:,:,1,1));
number_of_orientations = 4; %up to 16 possible here due to imported pyramid toolbox
chrom_attenuation = 0;
reference_frame_index = 1;
transition_width = 1; 
%DERIVATIVE PARAMETERS:
%Define which of the three YIQ channels to process according to given chrom attenuation:
if chrom_attenuation == 0
    flag_process_channel_vec = logical([1,0,0]);
else
    flag_process_channel_vec = logical([1,1,1]);
end
filter_order = number_of_orientations - 1;

%Don't amplify high and low residulals. Use given Alpha for all other subbands
phase_magnification_factors_for_different_levels = [0 , repmat(phase_magnification_factor, [1, number_of_levels]) , 0]'; 

%Get reference frame and convert it to ntc space:
reference_frame = video_frames(:,:,:,reference_frame_index);
reference_frame = rgb2ntsc(reference_frame);
reference_frame = squeeze(reference_frame(:,:,1));
%build complex steerable pyramid using reference frame:
[reference_pyramid, index_matrix] = build_complex_steerable_pyramid(reference_frame, number_of_levels, filter_order);
number_of_scales = (size(index_matrix,1)-2)/number_of_orientations + 2;
number_of_filter_bands = size(index_matrix,1);
number_of_elements_in_pyramid = dot(index_matrix(:,1),index_matrix(:,2));

%Scale up magnification levels (from 1 to number of scales):
if (size(phase_magnification_factors_for_different_levels,1) == 1)
    phase_magnification_factors_for_different_levels = repmat(phase_magnification_factors_for_different_levels,[number_of_filter_bands 1]);
elseif (size(phase_magnification_factors_for_different_levels,1) == number_of_scales)
   phase_magnification_factors_for_different_levels = scaleBand2all(phase_magnification_factors_for_different_levels, number_of_scales, number_of_orientations); 
end


%Go over frames:
%for now only use chroma channel (should use all channels at the end)!!!!!!!!
% % % phase_differences = zeros(number_of_elements_in_pyramid, number_of_frames, number_of_channels);
final_frames_over_time = zeros(frame_height,frame_width, number_of_frames);
phase_differences_temp = zeros(number_of_elements_in_pyramid,1);
phase_differences_filtered = zeros(number_of_elements_in_pyramid, 1);
phase_differences_filtered_amplified = zeros(number_of_elements_in_pyramid, 1);
final_frame = zeros(frame_height,frame_width,number_of_channels);
previous_pyramid = reference_pyramid;
for frame_counter = reference_frame_index+1:number_of_frames
    tic
    
    %Get current frame (ONLY 1 CHANNEL FOR NOW):
    %(1). one color:
    current_frame = video_frames(:,:,1,frame_counter);
    %(2). 1st channel of ntsc:
    current_frame = video_frames(:,:,:,frame_counter);
    current_frame = rgb2ntsc(current_frame);
    current_frame = current_frame(:,:,1);
    
    
    % Transform the current frame:
    current_pyramid = build_complex_steerable_pyramid(current_frame, number_of_levels, filter_order, transition_width);
    
    % Get Phase difference:
    phase_differences_temp = angle(current_pyramid./(previous_pyramid+eps));
    phase_differences_temp_transposed = phase_differences_temp';
    
    
    % Temporally Filter Them:
    phase_differences_filtered(:) = transpose( step(IIR_filter_object, phase_differences_temp_transposed) );
    
    % Amplify the phase changes
    for k = 1:size(index_matrix,1)
        idx = pyrBandIndices(index_matrix,k);
        if flag_use_temporally_filtered_phases==1
            phase_differences_filtered_amplified(idx) = ...
                phase_differences_filtered(idx) * phase_magnification_factors_for_different_levels(k);
        else
            phase_differences_filtered_amplified(idx) = ...
                phase_differences_temp(idx) * phase_magnification_factors_for_different_levels(k);
        end
    end
    
    % Magnify and reconstruct
    final_frame = reconSCFpyr(exp(1i*phase_differences_filtered_amplified) .* (previous_pyramid), index_matrix,'all', 'all', transition_width);
    final_frames_over_time(:,:,frame_counter) = final_frame;
    
    % Write Frame:
%     writeVideo(video_writer_object, im2uint8(final_frame));
    writeVideo(video_writer_object, (final_frame)/300);
    
    % Assign previous frame:
    previous_pyramid = current_pyramid;
    
    % Plot:
    imagesc(final_frame);
%     imagesc(phase_differences_temp);
%     pause(1);
    drawnow;
    toc
end
video_writer_object.close;

for frame_counter = 1:number_of_frames
    imagesc(final_frames_over_time(:,:,frame_counter));
    pause(0.5);
end
 
1;





% %--------------------------------------------------------------------------
% % The temporal signal is the phase changes of each frame from the reference
% % frame. We compute this on the fly instead of storing the transform for
% % all the frames (this means we will recompute the transform again later 
% % for the magnification)
% 
% fprintf('Computing phase differences\n');
%  
% phase_differences = zeros(number_of_elements_in_pyramid, number_of_frames, number_of_channels);
% for ii = 1:number_of_frames
%     
%     tmp = zeros(number_of_elements_in_pyramid, number_of_channels);
%     
%     for c = find(flag_process_channel_vec) 
%         tic
%         % Transform the reference frame
%         pyrRef = build_complex_steerable_pyramid(video_frames(:,:,c,reference_frame_index), number_of_levels, filter_order, transition_width);
%          
%         % Transform the current frame
%         current_pyramid = build_complex_steerable_pyramid(video_frames(:,:,c,ii), number_of_levels, filter_order, transition_width);
%         toc
%         tmp(:,c) = angle(current_pyramid) - angle(pyrRef);
%     end 
%      
%     phase_differences(:,ii,:) = tmp;
% end
% 
% 
% %--------------------------------------------------------------------------
% % Bandpass the phases
% 
% fprintf('Bandpassing phases\n');
% 
% phase_differences = single(phase_differences);
% freqDom = fft(phase_differences, [], 2);
% 
% first = ceil(low_cutoff_frequency*number_of_frames);
% second = floor(high_cutoff_frequency*number_of_frames);
% freqDom(:,1:first) = 0;
% freqDom(:,second+1:end) = 0;
% phase_differences = real(ifft(freqDom,[],2));
% 
% 
% %--------------------------------------------------------------------------
% % Magnify
% 
% fprintf('Magnifying\n');
% 
% video_writer_object = VideoWriter(output_file_name, 'Motion JPEG AVI');
% video_writer_object.Quality = 90;
% video_writer_object.FrameRate = video_stream.FrameRate;
% video_writer_object.open;
% 
% for ii = 1:number_of_frames
%     ii
%     
%     frame = video_frames(:,:,:,ii);
%     
%     for c = find(flag_process_channel_vec)
%         
%         % Amplify the phase changes
%         phase_differences_filtered_amplified = phase_differences(:,ii,c);
%         for k = 1:size(index_matrix,1)
%             idx = pyrBandIndices(index_matrix,k);
%             phase_differences_filtered_amplified(idx) = phase_differences_filtered_amplified(idx) * phase_magnification_factors_for_different_levels(k);
%         end
%         
%         % Attenuate the amplification in the chroma channels
%         if c > 1
%             phase_differences_filtered_amplified = phase_differences_filtered_amplified * chrom_attenuation;
%         end
%         
%         % Transform
%         current_pyramid = build_complex_steerable_pyramid(video_frames(:,:,c,ii), number_of_levels, filter_order, transition_width);
%     
%         % Magnify and reconstruct
%         frame(:,:,c) = reconSCFpyr(exp(1i*phase_differences_filtered_amplified) .* current_pyramid, index_matrix,'all', 'all', transition_width);
%     end
%     
%     % Back to RGB
%     frame = ntsc2rgb(frame); 
%     
%     writeVideo(video_writer_object, im2uint8(frame));
% end
% 
% video_writer_object.close;



















