function [Lpyr_stack, index_matrix] = build_laplacian_pyramid_from_video(video_file, start_index, end_index)
% [LPYR_STACK, pind] = build_Lpyr_stack(VID_FILE, START_INDEX, END_INDEX)
% 
% Apply Laplacian pyramid decomposition on vidFile from startIndex to
% endIndex
% 
% LPYR_STACK: stack of Laplacian pyramid of each frame 
% the second dimension is the color channel
% the third dimension is the time
%
% pind: see buildLpyr function in matlabPyrTools library
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
    video_height = video_stream.Height;
    video_width = video_stream.Width;
    number_of_channels = 3; %rgb
    current_frame = struct('cdata', zeros(video_height, video_width, number_of_channels, 'uint8'), 'colormap', []);


    %firstFrame:
    current_frame.cdata = read(video_stream, start_index);
    [current_frame_rgb,~] = frame2im(current_frame);
    current_frame_rgb = im2double(current_frame_rgb);
    current_frame_ntsc = rgb2ntsc(current_frame_rgb);

    [laplacian_pyramid,index_matrix] = build_laplacian_pyramid(current_frame_ntsc(:,:,1),'auto');

    % pre-allocate pyr stack
    Lpyr_stack = zeros(size(laplacian_pyramid,1),3,end_index - start_index +1);
    Lpyr_stack(:,1,1) = laplacian_pyramid;

    [Lpyr_stack(:,2,1), ~] = build_laplacian_pyramid(current_frame_ntsc(:,:,2),'auto');
    [Lpyr_stack(:,3,1), ~] = build_laplacian_pyramid(current_frame_ntsc(:,:,3),'auto');

    k = 1;
    for frame_counter = start_index+1:end_index
            k = k+1;
            current_frame.cdata = read(video_stream, frame_counter);
            [current_frame_rgb,~] = frame2im(current_frame);

            current_frame_rgb = im2double(current_frame_rgb);
            current_frame_ntsc = rgb2ntsc(current_frame_rgb);

            [Lpyr_stack(:,1,k),~] = build_laplacian_pyramid(current_frame_ntsc(:,:,1),'auto');
            [Lpyr_stack(:,2,k),~] = build_laplacian_pyramid(current_frame_ntsc(:,:,2),'auto');
            [Lpyr_stack(:,3,k),~] = build_laplacian_pyramid(current_frame_ntsc(:,:,3),'auto');
    end
end
