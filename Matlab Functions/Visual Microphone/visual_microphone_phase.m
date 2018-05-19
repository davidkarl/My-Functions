function output_name = visual_microphone_phase(...
    video_file, ...
    phase_magnification_factor, ...
    low_cutoff_frequency, ...
    high_cutoff_frequency, ...
    Fs, ...
    output_directory, ...
    varargin)
% PHASEAMPLIFY(VIDFILE, MAGPHASE, FL, FH, FS, OUTDIR, VARARGIN)
%
% Takes input VIDFILE and motion magnifies the motions that are within a
% passband of FL to FH Hz by MAGPHASE times. FS is the videos sampling rate
% and OUTDIR is the output directory. 
%
% Optional arguments:
% attenuateOtherFrequencies (false)
%   - Whether to attenuate frequencies in the stopband
% pyrType                   ('halfOctave')
%   - Spatial representation to use (see paper)
% sigma                     (0)
%   - Amount of spatial smoothing (in px) to apply to phases
% temporalFilter            (FIRWindowBP)
%   - What temporal filter to use
%

%Get input variables:
video_file = 'Chips1-2200Hz-Mary_Had-input.avi';
output_directory = 'C:\Users\master\Desktop\matlab';
low_cutoff_frequency = 100;
high_cutoff_frequency = 1000;
Fs = 2200;
phase_magnification_factor = 15;    

%Read Video Parameters:
video_reader_object = VideoReader(video_file);
[~, writeTag, ~] = fileparts(video_file);
frame_rate = video_reader_object.FrameRate;
video_frames = video_reader_object.read();
[frame_height, frame_width, number_of_channels, number_of_frames] = size(video_frames);

%Default inputs:
pyramid_types_strings = {'octave', 'halfOctave', 'smoothHalfOctave', 'quarterOctave'};
reference_frame_index = 1; %frame to use as a reference for phase calculations
flag_attenuate_other_frequencies = false;
pyramid_type = 'octave';
sigma_spatial_filter = 3;
temporal_filter_function = @FIRWindowBP;
video_scale = 1;
relevant_frames_vec = [1, number_of_frames];

%If wanted scale is different from 1 then get the equivalent resolution frame height and width:
%Get all frames into video stream:
video_frames = video_frames(:,:,:,relevant_frames_vec(1):relevant_frames_vec(2));
[frame_height, frame_width, number_of_channels, number_of_frames] = size(video_frames);
%Get new sizes if wanted scale is different from 1:
if video_scale ~= 1
    flag_resize_image = 1;
    [frame_height,frame_width] = size(imresize(video_frames(:,:,1,1), video_scale));
else
    flag_resize_image = 0;
end

%Get spatial filters:
pyramid_height = floor(log2(min(frame_height,frame_width))) - 2;
switch pyramid_type
    case 'octave'
        filters = ...
            get_complex_steerable_pyramid_filters([frame_height frame_width], 2.^[0:-1:-pyramid_height], 4);
    case 'halfOctave'
        filters = ...
            get_complex_steerable_pyramid_filters([frame_height frame_width], 2.^[0:-0.5:-pyramid_height], 8,'twidth', 0.75);
    case 'smoothHalfOctave'
        filters = ...
            get_complex_steerable_pyramid_filters_smooth([frame_height frame_width], 8, 'filtersPerOctave', 2);
    case 'quarterOctave'
        filters = ...
            get_complex_steerable_pyramid_filters_smooth([frame_height frame_width], 8, 'filtersPerOctave', 4);
    otherwise
        error('Invalid Filter Types');
end

%Get only relevant parts of filters for efficiency:
[cropped_filters, filters_non_zero_indices_cell_mat] = get_filters_and_indices_which_are_non_zero(filters);

%Initialization of motion magnified luma component:
magnified_LUMA_component_fft = zeros(frame_height,frame_width,number_of_frames,'single');

%Define build_level and reconstruct_level function:
build_level_function = @(im_dft, k) ifft2(ifftshift...
    (...
    cropped_filters{k} .* ...
    im_dft(filters_non_zero_indices_cell_mat{k,1}, filters_non_zero_indices_cell_mat{k,2})...
    )...
    );

reconstruct_level_function = @(im_dft, k) 2*(cropped_filters{k}.*fftshift(fft2(im_dft)));


%For easier reading define filters_non_zero_indices:
number_of_filters = numel(filters);
for level_counter = 1:number_of_filters
    filters_non_zero_indices{k} = [filters_non_zero_indices_cell_mat{k,1}, filters_non_zero_indices_cell_mat{k,2}];
end

%Shift all frames from uint8 to single precision:
for frame_counter = 1:number_of_frames
    video_frames = im2single(video_frames(:,:,:,frame_counter));
end


%LOOP OVER THE DIFFERENT FRAMES:
%written as workflow that will approximately be on FPGA - i cannot afford
%to use only reference frame since scenery can change:
video_frame_fft = zeros(frame_height,frame_width,number_of_frames,'single');
reference_frame = zeros(frame_height,frame_width);
reference_frame_fft = zeros(frame_height,frame_width);
previous_frame_angle = zeros(frame_height,frame_width);
filter_counter = 1;
number_of_points_to_filter = 2000;
for frame_counter = 1:number_of_frames
    
    %get current frame (RGB):
    current_frame = video_frames(:,:,:,frame_counter);
    
    %turn rgb frame to NTSC:
    current_frame = rgb_to_NTSC_luminance(current_frame);
    
    %resize current frame if needed:
    if flag_resize_image
        current_frame = imresize(current_frame(:,:,1), [frame_height,frame_width]);
    else
        current_frame = current_frame(:,:,1); 
    end
    
    %Transform current frame to fft domain:
    current_frame_fft = single( fftshift( fft2(current_frame) ) );
    
    %effectively BUILD PYRAMID LEVEL by convolving (in the frequency domain) with the appropriate filter:
    %AND FFTSHIFT AND IFFTSHIFT ARE VERY SUPURFLULOUS!!!! SIMPLY GET RID OF THEM!!!!
    current_frame_and_level_response = ifft2( ifftshift( cropped_filters{current_frame}.*current_frame_fft(filters_non_zero_indices{current_frame}) ) );
    
    %Get current level representation angle:
    current_pyramid_angle = angle(current_frame_and_level_response);
    
    %Get phase difference between former frame and current frame:
    frames_phase_difference_from_ref_over_time(:,:,frame_counter) = ...
        single(mod(pi+current_pyramid_angle-previous_frame_angle,2*pi)-pi); %get bounded phase difference from reference
    
    %Update reference frame angle:
    previous_frame_angle = current_pyramid_angle;
     
    %Check if we have reached the point at which we transfer the samples we
    %have so far into a temporal FIR filter or maybe simply use IIR:
    if filter_counter == number_of_points_to_filter
        %DO FILTERING...
        filter_counter = 0;
    end
    filter_counter = filter_counter + 1;
    
    %Do amplitude weighted blur of phases (MAYBE DO THIS BEFORE?- is it legite or will it enter noise?):
    %IS IT POSSIBLE TO DO THIS EFFICIENTLY? I SHOULD UNDERSTAND HOW IMPORTANT THIS IS:
    current_frame_phase_difference = ...
                              amplitude_weighted_blur(...
                                        current_frame_phase_difference, ...
                                        abs(current_frame_and_level_response)+eps, ...
                                        sigma_spatial_filter);
    
    %Align phases over time:
    
    
end %END OF FRAMES LOOP









end %END OF VISUAL MICROPHONE FUNCTION
