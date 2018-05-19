function motion_magnification_phase_amplify_simple(video_file, phase_magnification_factor, low_cutoff_frequency, high_cutoff_frequency, output_file_name, varargin)
% Eulerian motion magnification using phase manipulation in complex 
% steerable pyramids.
% This is a simple implementation we provide for refernce. It is slower and
% takes more memory, but is easier to read. It should only be used with
% small and short videos.
%
% Input:
%   vidFile - Path to video file
%   alpha - Magnification factor
%   fl - Low frequency cutoff
%   fh - High frequency cutoff
%   outFile - File name to write the result into
% 
% Requires Simoncelli's buildSCFpyr function for the complex steerable 
% pyramid transform, and reconSCFpyr to reconstruct from it.
%
% Design decisions:
%   - Amplify w.r.t some reference frame (typically the first) to make the
%     phases smaller. Otherwise discontinuities may occur when filtering phases
%     near the boundaries -pi, pi
%   - Typically processing the luminance channel only
%   - Not processing the high- and low-pass residuals. Those will just be
%     added back without modification during reconstruction
%   - Instead of filtering the complex phase e^(iw(x+delta(t))), we filter
%     on w(x+delta(t))
% 


%Load the sequence:
video_stream = VideoReader(video_file);
video_frames = im2double(video_stream.read);
[frame_height, frame_width, number_of_channels, number_of_frames] = size(video_frames);


%Parameters:
number_of_levels = get_max_complex_steerable_pyramid_height(video_frames(:,:,1,1));
number_of_orientations = 4; %up to 16 possible here due to imported pyramid toolbox
chrom_attenuation = 0;
reference_frame = 1;
transition_width = 1; 

%Don't amplify high and low residulals. Use given Alpha for all other subbands
phase_magnification_factors_for_different_levels = [0 , repmat(phase_magnification_factor, [1, number_of_levels]) , 0]'; 

%Convert to YIQ:
%TODO: deal correctly with single channel videos
for ii = 1:number_of_frames
    video_frames(:,:,:,ii) = rgb2ntsc(video_frames(:,:,:,ii));
end

%Define which of the three YIQ channels to process according to given chrom attenuation:
if chrom_attenuation == 0
    flag_process_channel_vec = logical([1,0,0]);
else
    flag_process_channel_vec = logical([1,1,1]);
end


filter_order = number_of_orientations - 1;
[~, pind] = build_complex_steerable_pyramid(video_frames(:,:,1,1), number_of_levels, filter_order);
numScales = (size(pind,1)-2)/number_of_orientations + 2;
numBands = size(pind,1);
numElements = dot(pind(:,1),pind(:,2));

% Scale up magnification levels
if (size(phase_magnification_factors_for_different_levels,1) == 1)
    phase_magnification_factors_for_different_levels = repmat(phase_magnification_factors_for_different_levels,[numBands 1]);
elseif (size(phase_magnification_factors_for_different_levels,1) == numScales)
   phase_magnification_factors_for_different_levels = scaleBand2all(phase_magnification_factors_for_different_levels, numScales, number_of_orientations); 
end


%--------------------------------------------------------------------------
% The temporal signal is the phase changes of each frame from the reference
% frame. We compute this on the fly instead of storing the transform for
% all the frames (this means we will recompute the transform again later 
% for the magnification)

fprintf('Computing phase differences\n');

deltaPhase = zeros(numElements, number_of_frames, number_of_channels);
parfor ii = 1:number_of_frames
    
    tmp = zeros(numElements, number_of_channels);
    
    for c = find(flag_process_channel_vec) 
        
        % Transform the reference frame
        pyrRef = build_complex_steerable_pyramid(video_frames(:,:,c,reference_frame), number_of_levels, filter_order, transition_width);
        
        % Transform the current frame
        pyrCA = build_complex_steerable_pyramid(video_frames(:,:,c,ii), number_of_levels, filter_order, transition_width);
        
        tmp(:,c) = angle(pyrCA) - angle(pyrRef);
    end
    
    deltaPhase(:,ii,:) = tmp;
end


%--------------------------------------------------------------------------
% Bandpass the phases

fprintf('Bandpassing phases\n');

deltaPhase = single(deltaPhase);
freqDom = fft(deltaPhase, [], 2);

first = ceil(low_cutoff_frequency*number_of_frames);
second = floor(high_cutoff_frequency*number_of_frames);
freqDom(:,1:first) = 0;
freqDom(:,second+1:end) = 0;
deltaPhase = real(ifft(freqDom,[],2));


%--------------------------------------------------------------------------
% Magnify

fprintf('Magnifying\n');

vw = VideoWriter(output_file_name, 'Motion JPEG AVI');
vw.Quality = 90;
vw.FrameRate = video_stream.FrameRate;
vw.open;

for ii = 1:number_of_frames
    ii
    
    frame = video_frames(:,:,:,ii);
    
    for c = find(flag_process_channel_vec)
        
        % Amplify the phase changes
        phase1 = deltaPhase(:,ii,c);
        for k = 1:size(pind,1)
            idx = pyrBandIndices(pind,k);
            phase1(idx) = phase1(idx) * phase_magnification_factors_for_different_levels(k);
        end
         
        % Attenuate the amplification in the chroma channels
        if c > 1
            phase1 = phase1 * chrom_attenuation;
        end
        
        % Transform
        pyrCA = build_complex_steerable_pyramid(video_frames(:,:,c,ii), number_of_levels, filter_order, transition_width);
    
        % Magnify and reconstruct
        frame(:,:,c) = reconSCFpyr(exp(1i*phase1) .* pyrCA, pind,'all', 'all', transition_width);
    end
    
    % Back to RGB
    frame = ntsc2rgb(frame); 
    
    writeVideo(vw, im2uint8(frame));
end

vw.close;

