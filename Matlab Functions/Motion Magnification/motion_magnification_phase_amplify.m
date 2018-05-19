function output_name = motion_magnification_phase_amplify(...
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

% Read Video
video_reader_object = VideoReader(video_file);
[~, writeTag, ~] = fileparts(video_file);
frame_rate = video_reader_object.FrameRate;
video_stream = video_reader_object.read();
[frame_height, frame_width, number_of_channels, number_of_frames] = size(video_stream);

%Parse Input:
p = inputParser();

flag_attenuate_other_frequencies_default = false; %If true, use reference frame phases
pyramid_types = {'octave', 'halfOctave', 'smoothHalfOctave', 'quarterOctave'};
check_pyramid_type = @(x) find(ismember(x, pyramid_types));
default_pyramid_type = 'octave';
default_sigma = 3;
default_temporal_filter = @FIRWindowBP;
default_scale = 1;
default_frames = [1, number_of_frames];
addOptional(p, 'flag_attenuate_other_frequencies', flag_attenuate_other_frequencies_default, @islogical);
addOptional(p, 'pyramid_type', default_pyramid_type, check_pyramid_type);
addOptional(p, 'sigma', default_sigma, @isnumeric);
addOptional(p, 'temporal_filter', default_temporal_filter);
addOptional(p, 'scale_video', default_scale);
addOptional(p, 'relevant_frames', default_frames);
parse(p, varargin{:});
reference_frame_index = 1;
flag_attenuate_other_frequencies = p.Results.flag_attenuate_other_frequencies;
pyramid_type = p.Results.pyramid_type;
sigma_spatial_filter = p.Results.sigma;
temporal_filter_function = p.Results.temporal_filter;
video_scale = p.Results.scale_video;
relevant_frames_vec = p.Results.relevant_frames;

%if wanted scale is different from 1 then get the equivalent resolution frame height and width:
video_stream = video_stream(:,:,:,relevant_frames_vec(1):relevant_frames_vec(2));
[frame_height, frame_width, number_of_channels, number_of_frames] = size(video_stream);
if video_scale ~= 1
    [frame_height,frame_width] = size(imresize(video_stream(:,:,1,1), video_scale));
end

%Get spatial filters:
pyramid_height = get_max_complex_steerable_pyramid_height(zeros(frame_height,frame_width));
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


%SWITCH ALL FRAMES TO FFT DOMAIN:
%First compute phase differences from reference frame:
number_of_levels = numel(filters);
fprintf('Moving video to Fourier domain\n');
video_fft = zeros(frame_height,frame_width,number_of_frames,'single');
for k = 1:number_of_frames
    current_frame = rgb_to_NTSC_luminance(im2single(video_stream(:,:,:,k)));
    current_frame_resized = imresize(current_frame(:,:,1), [frame_height frame_width]);
    video_fft(:,:,k) = single(fftshift(fft2(current_frame_resized)));
end
clear video_stream;

%LOOP OVER THE DIFFERENT LEVELS:
for current_level_counter = 2:number_of_levels-1
    % Compute phases of level
    % We assume that the video is mostly static
    
    %build_level_function gets image fft and computes the convolved
    %image with the appropriate
    reference_pyramid = build_level_function(video_fft(:,:,reference_frame_index), current_level_counter);
    reference_pyramid_phase_term_only = reference_pyramid./abs(reference_pyramid);
    reference_pyramid_angle = angle(reference_pyramid);
    
    %Initialize frames phase difference over time variable:
    frames_phase_difference_over_time = ...
        zeros(size(reference_pyramid_angle,1), size(reference_pyramid_angle,2), number_of_frames, 'single');
    
    %Loop over the different frames, get the phase and find the difference from reference image phase:
    for frame_counter = 1:number_of_frames
        current_frame_and_level_response = build_level_function(video_fft(:,:,frame_counter), current_level_counter);
        current_pyramid_angle = angle(current_frame_and_level_response);
        frames_phase_difference_over_time(:,:,frame_counter) = ...
            single(mod(pi+current_pyramid_angle-reference_pyramid_angle,2*pi)-pi); %get bounded phase difference from reference
    end
    
    
    %Do Temporal Filtering of phase differences using some defined function (DEFAULTED TO FIRWindowBP):
    frames_phase_difference_over_time = ...
        temporal_filter_function(frames_phase_difference_over_time, low_cutoff_frequency/Fs,high_cutoff_frequency/Fs);
    
    
    %Loop over the difference frames and apply Magnification to each frame:
    for frame_counter = 1:number_of_frames
        %get current frame phase difference:
        current_frame_phase_difference = frames_phase_difference_over_time(:,:,frame_counter);
        
        %AGAIN calculate current frame and level response (MUST BE SOME WAY NOT TO DO THIS AGAIN!!!!):
        current_frame_and_level_response = build_level_function(video_fft(:,:,frame_counter),current_level_counter);
        
        %Do Amplitude Weighted Blur of phase difference as the phase is less reliable where amplitude is low:
        %(maybe add zeroing of phase under ceratin amplitude like SOL):
        if sigma_spatial_filter ~= 0
            current_frame_phase_difference = ...
                              amplitude_weighted_blur(...
                                        current_frame_phase_difference, ...
                                        abs(current_frame_and_level_response)+eps, ...
                                        sigma_spatial_filter);
        end
         
        %Increase phase variation:
        current_frame_phase_difference_amplified = phase_magnification_factor * current_frame_phase_difference;
        
        %NOTICE WE ARE DEALING WITH PHASE DIFFERENCE, SO THE FILTER
        %SHOULD CONSIDER THAT!!!! AND IT DOESN'T!!!! WE SHOULD ADD AN
        %EQUALIZER!!!!
        if flag_attenuate_other_frequencies == 1
            %just use the amplified phase only plus the original phase
            %(the magnification is for phase difference):
            current_frame_and_level_response_temp = ...
                abs(current_frame_and_level_response) .* reference_pyramid_phase_term_only;
        else
            %add magnified phase difference to curent phase:
            current_frame_and_level_response_temp = ...
                current_frame_and_level_response;
        end
        current_frame_and_level_response_final = exp(1i*current_frame_phase_difference_amplified) .* ...
                                                                        current_frame_and_level_response_temp;
        
        %Reconstruct current frame and level by fft-ing the final outcome and multiplying it by the filter:
        current_frame_and_level_reconstructed = ...
            reconstruct_level_function(current_frame_and_level_response_final, current_level_counter);
        
        magnified_LUMA_component_fft(filters_non_zero_indices_cell_mat{current_level_counter,1}, ...
                                     filters_non_zero_indices_cell_mat{current_level_counter,2},frame_counter) = ...
                                     current_frame_and_level_reconstructed + ...
                                     magnified_LUMA_component_fft(filters_non_zero_indices_cell_mat{current_level_counter,1}, filters_non_zero_indices_cell_mat{current_level_counter,2},frame_counter);
    end %END OF FRAMES LOOP
    
    
    
end %END OF LEVELS LOOP


%Add unmolested lowpass residual:
current_level_counter = numel(filters);
for frame_counter = 1:number_of_frames
    low_passed_frame_fft = video_fft(filters_non_zero_indices_cell_mat{current_level_counter,1},...
                                     filters_non_zero_indices_cell_mat{current_level_counter,2},frame_counter) ...
                                     .* cropped_filters{end}.^2;
    magnified_LUMA_component_fft(filters_non_zero_indices_cell_mat{current_level_counter,1},...
                                 filters_non_zero_indices_cell_mat{current_level_counter,2},frame_counter) = ...
        magnified_LUMA_component_fft(filters_non_zero_indices_cell_mat{current_level_counter,1},filters_non_zero_indices_cell_mat{current_level_counter,2},frame_counter) ...
        + low_passed_frame_fft;
end


clear video_fft;
video_reader_object = VideoReader(video_file);
video_stream = video_reader_object.read([relevant_frames_vec]);
final_result_video = zeros(frame_height,frame_width,number_of_channels,number_of_frames,'uint8');
for k = 1:number_of_frames
    magnified_LUMA_component = real(ifft2(ifftshift(magnified_LUMA_component_fft(:,:,k))));
    output_frame(:,:,1) = magnified_LUMA_component;
    current_frame = rgb_to_NTSC_luminance(im2single(video_stream(:,:,:,k)));
    current_frame = imresize(current_frame, [frame_height, frame_width]);
    output_frame(:,:,2:3) = current_frame(:,:,2:3);
    output_frame = ntsc2rgb(output_frame);
    %Put frame in output:
    final_result_video(:,:,:,k) = im2uint8(output_frame);
end

output_name = sprintf('%s-%s-band%0.2f-%0.2f-sr%d-alpha%d-mp%d-sigma%d-scale%0.2f-frames%d-%d-%s.avi', writeTag, func2str(temporal_filter_function), low_cutoff_frequency, high_cutoff_frequency,Fs, phase_magnification_factor, flag_attenuate_other_frequencies, sigma_spatial_filter, video_scale, relevant_frames_vec(1), relevant_frames_vec(2), pyramid_type);
writeVideo(final_result_video, frame_rate, fullfile(output_directory, output_name));
end
