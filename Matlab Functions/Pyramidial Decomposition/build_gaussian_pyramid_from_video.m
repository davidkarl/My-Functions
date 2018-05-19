function gaussian_pyramid_stack = build_gaussian_pyramid_from_video(video_file, start_index, end_index, level)
% GDOWN_STACK = build_GDown_stack(VID_FILE, START_INDEX, END_INDEX, LEVEL)
% 
% Apply Gaussian pyramid decomposition on VID_FILE from START_INDEX to
% END_INDEX and select a specific band indicated by LEVEL
% 
% GDOWN_STACK: stack of one band of Gaussian pyramid of each frame 
% the first dimension is the time axis
% the second dimension is the y axis of the video
% the third dimension is the x axis of the video
% the forth dimension is the color channel
% 
% Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
% Quanta Research Cambridge, Inc.
%
% Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih, 
% License: Please refer to the LICENCE file
% Date: June 2012
%

    %Read video:
    video_stream = VideoReader(video_file);
    %Extract video info
    video_row_size = video_stream.Height;
    video_column_size = video_stream.Width;
    number_of_channels = 3;
    
    %WHY UINT8?!??!!?!?!?:
    current_frame = struct('cdata', zeros(video_row_size, video_column_size, number_of_channels, 'uint8'), 'colormap', []);

    %first frame:
    current_frame.cdata = read(video_stream, start_index);
    [current_frame_rgb, ~] = frame2im(current_frame);
    current_frame_rgb = im2double(current_frame_rgb);
    current_frame_ntsc = rgb2ntsc(current_frame_rgb);

    blurred = blur_downsample_in_pyramidal_fashion_color(current_frame_ntsc,level);

    %create pyr stack:
    gaussian_pyramid_stack = zeros(end_index - start_index +1, size(blurred,1),size(blurred,2),size(blurred,3));
    gaussian_pyramid_stack(1,:,:,:) = blurred;

    k = 1;
    for frame_index = start_index+1:end_index
            k = k+1;
            current_frame.cdata = read(video_stream, frame_index);
            [current_frame_rgb,~] = frame2im(current_frame);

            current_frame_rgb = im2double(current_frame_rgb);
            current_frame_ntsc = rgb2ntsc(current_frame_rgb);

            blurred = blur_downsample_in_pyramidal_fashion_color(current_frame_ntsc,level);
            gaussian_pyramid_stack(k,:,:,:) = blurred;

    end
    
end
