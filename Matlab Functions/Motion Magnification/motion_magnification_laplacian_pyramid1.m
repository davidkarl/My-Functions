function motion_magnification_laplacian_pyramid1(...
    video_file, ...
    output_directory, ...
    alpha, ...
    lambda_c, ...
    low_cutoff_frequency, ...
    high_cutoff_frequency, ...
    Fs, ... 
    chrom_attenuation)
% amplify_spatial_lpyr_temporal_ideal(vidFile, outDir, alpha, lambda_c,
%                                     wl, wh, samplingRate, chromAttenuation)
%
% Spatial Filtering: Laplacian pyramid
% Temporal Filtering: Ideal bandpass


%Get input variables:
video_file = 'Chips1-2200Hz-Mary_Had-input.avi';
output_directory = 'C:\Users\master\Desktop\matlab';
alpha = 3;
lambda_c = 6;
low_cutoff_frequency = 100;
high_cutoff_frequency = 1000;
Fs = 2200;
chrom_attenuation = 1;


%get video file name:
[~,video_name] = fileparts(video_file);
output_file_nme = fullfile(output_directory,[video_name '-ideal-from-' num2str(low_cutoff_frequency) ...
    '-to-' num2str(high_cutoff_frequency) '-alpha-' num2str(alpha) ...
    '-lambda_c-' num2str(lambda_c) '-chromAtn-' ...
    num2str(chrom_attenuation) '.avi']);

%Read video:
video_stream = VideoReader(video_file);
%Extract video info:
frame_height = video_stream.Height;
frame_width = video_stream.Width;
number_of_channels = 3;
frame_rate = video_stream.FrameRate;
% number_of_frames = video_stream.CurrentTime;
number_of_frames = 200; %check out later which property gives me this, NumberOfFrames doesn't exist
temp_frame_struct = struct('cdata', zeros(frame_height, frame_width, number_of_channels, 'uint8'), 'colormap', []);
start_frame = 1;
% final_frame = number_of_frames-10;
final_frame = 100;

% %Define output_video writer:
% output_video = VideoWriter(output_file_nme);
% output_video.FrameRate = frame_rate;
% open(output_video)

%read first frame:
current_frame_struct = struct('cdata', zeros(frame_height, frame_width, number_of_channels, 'uint8'), 'colormap', []);
current_frame_struct.cdata = readframe(video_stream);
[current_frame_rgb,~] = frame2im(current_frame_struct);
current_frame_rgb = im2double(current_frame_rgb);
current_frame_ntsc = rgb2ntsc(current_frame_rgb);
[laplacian_pyramid,index_matrix] = build_laplacian_pyramid(current_frame_ntsc(:,:,1),'auto');

%Pre-allocate pyr stack
laplacian_pyramid_stack = zeros(size(laplacian_pyramid,1), 3, final_frame - start_frame +1);
laplacian_pyramid_stack(:,1,1) = laplacian_pyramid;
[laplacian_pyramid_stack(:,2,1), ~] = build_laplacian_pyramid(current_frame_ntsc(:,:,2),'auto');
[laplacian_pyramid_stack(:,3,1), ~] = build_laplacian_pyramid(current_frame_ntsc(:,:,3),'auto');

%Calculate the laplacian pyramid for each frame:
k = 1;
for frame_counter = start_frame+1:final_frame
    k = k+1;
    current_frame_struct.cdata = read(video_stream, frame_counter);
    [current_frame_rgb,~] = frame2im(current_frame_struct);
    current_frame_rgb = im2double(current_frame_rgb);
    current_frame_ntsc = rgb2ntsc(current_frame_rgb);
    
    %Calculate the laplacian pyramid for each color channel:
    [laplacian_pyramid_stack(:,1,k),~] = build_laplacian_pyramid(current_frame_ntsc(:,:,1),'auto');
    [laplacian_pyramid_stack(:,2,k),~] = build_laplacian_pyramid(current_frame_ntsc(:,:,2),'auto');
    [laplacian_pyramid_stack(:,3,k),~] = build_laplacian_pyramid(current_frame_ntsc(:,:,3),'auto');
end

%Bandpass the signal:
temporal_filtered_stack = ...
    apply_ideal_bandpass_to_mat(laplacian_pyramid_stack, 3, low_cutoff_frequency, high_cutoff_frequency, Fs);


%Amplify each spatial frequency bands according to Figure 6 of our paper:
total_number_of_stacked_elements = size(laplacian_pyramid_stack(:,1,1),1);
number_of_levels = size(index_matrix,1);

%the factor to boost alpha above the bound we have in the paper. (for better visualization)
exaggeration_factor = 2;

%compute the representative wavelength lambda for the lowest spatial freqency band of Laplacian pyramid:
lambda = (frame_height^2 + frame_width^2).^0.5 / 3; % 3 is experimental constant
delta = lambda_c/8/(1+alpha);

%Loop over the different levels and do linear magnification:
for level_counter = number_of_levels:-1:1
    %Get current stack indices:
    current_indices = total_number_of_stacked_elements - prod(index_matrix(level_counter,:)) + 1:total_number_of_stacked_elements;
    %compute modified alpha for this level
    current_alpha = lambda/delta/8 - 1;
    current_alpha = current_alpha*exaggeration_factor;
    
    %Ignore the highest and lowest spatial frequency band:
    if (level_counter == number_of_levels || level_counter == 1)
        temporal_filtered_stack(current_indices,:,:) = 0;    
    elseif (current_alpha > alpha)
        %representative lambda exceeds lambda_c
        temporal_filtered_stack(current_indices,:,:) = alpha*temporal_filtered_stack(current_indices,:,:);
    else
        temporal_filtered_stack(current_indices,:,:) = current_alpha*temporal_filtered_stack(current_indices,:,:);
    end
    
    %total number of remaining stacked elements:
    total_number_of_stacked_elements = total_number_of_stacked_elements - prod(index_matrix(level_counter,:));
    
    %go one level down on pyramid, representative lambda will reduce by factor of 2:
    lambda = lambda/2;
end


%output video:
k = 0;
for frame_index = start_frame+1:final_frame
    k = k + 1;
    temp_frame_struct.cdata = read(video_stream, frame_index);
    [rgb_frame,~] = frame2im(temp_frame_struct);
    rgb_frame = im2double(rgb_frame);
    frame_ntsc = rgb2ntsc(rgb_frame);
    
    filtered_frame = zeros(frame_height,frame_width,3);
    filtered_frame(:,:,1) = reconstruct_laplacian_pyramid(temporal_filtered_stack(:,1,k),index_matrix);
    filtered_frame(:,:,2) = reconstruct_laplacian_pyramid(temporal_filtered_stack(:,2,k),index_matrix)*chrom_attenuation;
    filtered_frame(:,:,3) = reconstruct_laplacian_pyramid(temporal_filtered_stack(:,3,k),index_matrix)*chrom_attenuation;
    
    filtered_frame = filtered_frame + frame_ntsc;
    
    frame_rgb = ntsc2rgb(filtered_frame);
    
    frame_rgb(frame_rgb > 1) = 1;
    frame_rgb(frame_rgb < 0) = 0;
    
    
    writeVideo(output_video,im2uint8(frame_rgb));
end


close(output_video);

end








function [laplacian_pyramid,index_matrix] = build_laplacian_pyramid(mat_in, pyramid_height, filter1, filter2, boundary_conditions_string)
% [PYR, INDICES] = buildLpyr(IM, HEIGHT, FILT1, FILT2, EDGES)
%
% FILT1 (optional) can be a string naming a standard filter (see namedFilter), 
% or a vector which will be used for (separable) convolution. Default = 'binom5'.  
% FILT2 specifies the "expansion" filter (default = filt1).  
% EDGES specifies edge-handling, and defaults to 'reflect1' (see corrDn).
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.

mat_in_size = size(mat_in);
filter1 = 'binom5';
filter1 = filter1(:);
filter2 = filter1;
pyramid_height = 1 + get_max_pyramid_height(mat_in_size, max(size(filter1,1), size(filter2,1)));
boundary_conditions_string = 'reflect1';


if pyramid_height <= 1
    %Stop condition:
    laplacian_pyramid = mat_in(:); %in the final stage return the low pass version
    index_matrix = mat_in_size;
    
else
    
    if mat_in_size(2) == 1
        low_passed_downsampled_mat_in = corr2_downsample(mat_in, filter1, boundary_conditions_string, [2 1], [1 1]);
    elseif mat_in_size(1) == 1
        low_passed_downsampled_mat_in = corr2_downsample(mat_in, filter1', boundary_conditions_string, [1 2], [1 1]);
    else
        low_passed_downsampled_mat_in = corr2_downsample(mat_in, filter1', boundary_conditions_string, [1 2], [1 1]);
        int_sz = size(low_passed_downsampled_mat_in);
        low_passed_downsampled_mat_in = corr2_downsample(low_passed_downsampled_mat_in, filter1, boundary_conditions_string, [2 1], [1 1]);
    end
    
    %Use recursion to build pyramid at coarser scales:
    [laplacian_pyramid_next_level,index_matrix_next_level] = ...
              build_laplacian_pyramid(low_passed_downsampled_mat_in, pyramid_height-1, filter1, filter2, boundary_conditions_string);
    
    %Upsample low-passed and downsampled mat and filter it again to remove spectral replicas:
    %CAN WE GET RID OF THIS MAYBE BY USING THE RESULT FROM THE LOW-PASS BEFORE DOWNSAMPLING???
    if (mat_in_size(1) == 1)
        upsampled_low_passed_mat = upsample_inserting_zeros_convolve(low_passed_downsampled_mat_in, filter2', boundary_conditions_string, [1 2], [1 1], mat_in_size);
    elseif (mat_in_size(2) == 1)
        upsampled_low_passed_mat = upsample_inserting_zeros_convolve(low_passed_downsampled_mat_in, filter2, boundary_conditions_string, [2 1], [1 1], mat_in_size);
    else
        upsampled_low_passed_mat = upsample_inserting_zeros_convolve(low_passed_downsampled_mat_in, filter2, boundary_conditions_string, [2 1], [1 1], int_sz);
        upsampled_low_passed_mat = upsample_inserting_zeros_convolve(upsampled_low_passed_mat, filter2', boundary_conditions_string, [1 2], [1 1], mat_in_size);
    end
    
    %Substract high passed upsampled mat:
    high_passed_difference_from_low_passed_mat = mat_in - upsampled_low_passed_mat;
    
    %Add another layer to the laplacian pyramid consisting 
    laplacian_pyramid = [high_passed_difference_from_low_passed_mat(:); laplacian_pyramid_next_level];
    index_matrix = [mat_in_size; index_matrix_next_level];
    
end

end %END OF BUILD LAPLACIAN PYRAMID


function result = upsample_inserting_zeros_convolve(mat_in,filter_mat,boundary_conditions_string,upsampling_factor_vec,start_indices,stop_indices,res)
% THIS CODE IS NOT ACTUALLY USED! (MEX FILE IS CALLED INSTEAD)
% EDGES is a string determining boundary handling:
%    'circular' - Circular convolution
%    'reflect1' - Reflect about the edge pixels
%    'reflect2' - Reflect, doubling the edge pixels
%    'repeat'   - Repeat the edge pixels
%    'zero'     - Assume values of zero outside image boundary
%    'extend'   - Reflect and invert
%    'dont-compute' - Zero output when filter overhangs OUTPUT boundaries
%

start_indices = [1,1];
%A multiple of step:
stop_indices = ...
    upsampling_factor_vec .* (floor((start_indices-ones(size(start_indices)))./upsampling_factor_vec) + size(mat_in));

%Insert Zeros before convolution
tmp = zeros(size(res));
tmp(start_indices(1):upsampling_factor_vec(1):stop_indices(1),...
    start_indices(2):upsampling_factor_vec(2):stop_indices(2)) = mat_in;

%Convolve:
result = conv2_reflective_boundary_conditions(tmp,filter_mat);
if exist('res','var')
    result = result + res;
end

end %END OF UPSAMPLE CONV FUNCTION!



function res = corr2_downsample(mat_in, filter_mat, boundary_conditions_string, downsample_factor_vec, start, stop)
% EDGES is a string determining boundary handling:
%    'circular' - Circular convolution
%    'reflect1' - Reflect about the edge pixels
%    'reflect2' - Reflect, doubling the edge pixels
%    'repeat'   - Repeat the edge pixels
%    'zero'     - Assume values of zero outside image boundary
%    'extend'   - Reflect and invert (continuous values and derivs)
%    'dont-compute' - Zero output when filter overhangs input boundaries

start = [1,1];
stop = size(mat_in);

%Reverse order of taps in filt, to do correlation instead of convolution
filter_mat = filter_mat(size(filter_mat,1):-1:1,size(filter_mat,2):-1:1);

%Convolve with filter:
tmp = conv2_reflective_boundary_conditions(mat_in,filter_mat);

%Downsample:
res = tmp(start(1):downsample_factor_vec(1):stop(1),start(2):downsample_factor_vec(2):stop(2));
end %END OF CORR2 DOWNSAMPLE FUNCTION!


function [filtered] = apply_ideal_bandpass_to_mat(...
                mat_in, dimension_to_do_filtering, low_cutoff_frequency, high_cutoff_frequency, sampling_rate)

    %Shift dimension we want to do the filtering on to be the first dimension for easier handling:
    mat_in_shifted = shiftdim(mat_in,dimension_to_do_filtering-1);
    mat_in_shifted_size = size(mat_in_shifted);
    
    number_of_elements_to_filter = mat_in_shifted_size(1);
    dn = size(mat_in_shifted_size,2);
    
    %Define frequency vec and logical mask of band-passed frequencies:
    frequency_vec = 1:number_of_elements_to_filter;
    frequency_vec = (frequency_vec-1)/number_of_elements_to_filter*sampling_rate;
    band_pass_logical_mask = frequency_vec > low_cutoff_frequency & frequency_vec < high_cutoff_frequency;
    
    %Expand the logical mask to all elements in the mat:
    mat_in_shifted_size(1) = 1;
    band_pass_logical_mask = band_pass_logical_mask(:);
    band_pass_logical_mask = repmat(band_pass_logical_mask, mat_in_shifted_size);

    %FFT the signal, bandpass it there, and use ifft to get the filtered signal back:
    F = fft(mat_in_shifted,[],1);
    F(~band_pass_logical_mask) = 0;
    filtered = real(ifft(F,[],1));
    
    %return the mat to its original state:
    filtered = shiftdim(filtered,dn-(dimension_to_do_filtering-1));
    
end %END OF APPLY IDEAL BANDPASS TO MAT FUNCTION!



function [filtered_mat] = conv2_reflective_boundary_conditions(mat1,mat2,flag_filter_origin_location)

flag_filter_origin_location = 0;

if (size(mat1,1) >= size(mat2,1)) && (size(mat1,2) >= size(mat2,2))
    large_mat = mat1; 
    small_mat = mat2;
elseif  (size(mat1,1) <= size(mat2,1)) && (size(mat1,2) <= size(mat2,2))
    large_mat = mat2; 
    small_mat = mat1;
end

%Get mat sizes:
large_mat_rows = size(large_mat,1);
large_mat_columns = size(large_mat,2);
small_mat_rows = size(small_mat,1);
small_mat_columns = size(small_mat,2);

% These values are one less than the index of the small mtx that falls on 
% the border pixel of the large matrix when computing the first convolution response sample:
small_mat_row_center = floor( (small_mat_rows+flag_filter_origin_location-1)/2 );
small_mat_column_center = floor( (small_mat_columns+flag_filter_origin_location-1)/2 );
small_mat_shifted_row_center = small_mat_rows - small_mat_row_center;
small_mat_shifted_column_center = small_mat_columns - small_mat_column_center;
large_mat_valid_rows = large_mat_rows - small_mat_row_center;
large_mat_valid_columns = large_mat_columns - small_mat_column_center;

%Pad with reflected copies:
large_mat_padded = [ 
    large_mat(small_mat_shifted_row_center:-1:2, small_mat_shifted_column_center:-1:2), ...
    large_mat(small_mat_shifted_row_center:-1:2, :), ...
	large_mat(small_mat_shifted_row_center:-1:2, large_mat_columns-1:-1:large_mat_valid_columns)...
    ; ... 
    large_mat(:, small_mat_shifted_column_center:-1:2),    ...
    large_mat,   ...
    large_mat(:, large_mat_columns-1:-1:large_mat_columns-small_mat_column_center)...
    ; ...
    large_mat(large_mat_rows-1:-1:large_mat_valid_rows, small_mat_shifted_column_center:-1:2), ...
    large_mat(large_mat_rows-1:-1:large_mat_valid_rows, :), ...
    large_mat(large_mat_rows-1:-1:large_mat_valid_rows, large_mat_columns-1:-1:large_mat_valid_columns) ...
    ];

%Actually convolve mats:
filtered_mat = conv2(large_mat_padded,small_mat,'valid');
end %END OF CONV2 WITH REFLECTIVE BOUNDARY CONDITIONS FUNCTION!


