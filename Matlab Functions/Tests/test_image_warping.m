%test image warping

%get reference image:
original_image = imread('barbara.tif');
video_file_writer = VideoWriter('image_warps.avi','Uncompressed AVI');
open(video_file_writer);
flag_shift_random_method = 2; %1=rand+filtering, 2=speckles
flag_interpolation_method = 2; %1=linear, 2=cubic, 3=makima, 4=spline, 5=MEX
warp_sigma = 0.2;
optical_SNR_dB = 20;
number_of_bits = 12; %for now i will simply use double
number_of_images_to_save = 50;

%Direct filtering stuff:
filter_size = 100; 
filter_sigma = 20;
warp_filter = fspecial('gaussian',filter_size,filter_sigma);


%noise:
optical_SNR = 10.^(optical_SNR_dB/10);
image_mean = mean(original_image(:));
noise_std = image_mean/optical_SNR;

%output file:
% file_name = 'warp_images.bin';
% fid = fopen(file_name,'W');
shifts_file_name = 'xy_shifts.bin';
fid_shifts = fopen(shifts_file_name,'W');

%interpoation metho stuff:
method_strings = {'linear','cubic','makima','spline','MEX'};
method_string = method_strings{flag_interpolation_method};

%original image stuff:
original_image = double(original_image);
[image_width,image_height] = size(original_image);
original_image = original_image + noise_std*randn(image_width,image_height);

%grid:
[X,Y] = meshgrid(1:image_width,1:image_height);

%Spline coefficients:
spline = fn2fm(spapi({aptknt(1:image_width,3),aptknt(1:image_height,3)},{1:image_width,1:image_height},original_image),'pp');

%Generate and save images:
% fwrite(fid,original_image,'double');
for image_counter = 1:number_of_images_to_save
    tic
    %Generate Shift Field:
    if flag_shift_random_method == 1
        %(1).filter warp
        X_warp = randn(image_width+filter_size*2,image_height+filter_size*2);
        Y_warp = randn(image_width+filter_size*2,image_height+filter_size*2);
        X_warp = conv2(X_warp,warp_filter,'same');
        Y_warp = conv2(Y_warp,warp_filter,'same');
        X_warp = X_warp(filter_size+1:filter_size+image_width,filter_size+1:filter_size+image_height);
        Y_warp = Y_warp(filter_size+1:filter_size+image_width,filter_size+1:filter_size+image_height);
        X_warp = X_warp/max(abs(X_warp(:)));
        Y_warp = Y_warp/min(abs(Y_warp(:)));
    elseif flag_shift_random_method == 2
        %(2).speckles warp:
        speckle_size = 80;
        N = max(image_width,image_height);
        % X_warp = abs(create_speckles_of_certain_size_in_pixels(speckle_size,N,1,1)).^2;
        % Y_warp = abs(create_speckles_of_certain_size_in_pixels(speckle_size,N,1,1)).^2;
        X_warp = real(create_speckles_of_certain_size_in_pixels(speckle_size,N,1,1));
        Y_warp = real(create_speckles_of_certain_size_in_pixels(speckle_size,N,1,1));
        X_warp = X_warp(1:image_width,1:image_height);
        Y_warp = Y_warp(1:image_width,1:image_height);
        X_warp = X_warp - mean(X_warp(:));
        Y_warp = Y_warp - mean(Y_warp(:));
        %now it's approximately the histogram of 1*randn
    end
    
%     %Temporary - only linear global shift:
%     X_warp = ones(image_height,image_width)*(-1)^image_counter;
%     Y_warp = zeros(image_height,image_width);
    
    %Warp grid:
    shifts_x = warp_sigma*X_warp;
    shifts_y = warp_sigma*Y_warp;
    X2 = X + shifts_x;
    Y2 = Y + shifts_y;
    
    %Interpolate and add noise:
    if flag_interpolation_method ~= 5
        %(1). Matlab's native interp2:
        warped_image = interp2(X,Y,original_image,X2,Y2,method_string);
    elseif flag_interpolation_method == 5
        %(2). MEX:
        warped_image =  ppmval([X2(:)',Y2(:)'],spline);
        warped_image = reshape(warped_image,image_height,image_width);
    end
    %(3). Add Noise:
    warped_image = warped_image + noise_std*randn(image_width,image_height);
    %(4). Devide by 300 to put be able to have it definitely between 0 and 1:
    warped_image = warped_image / 300;
     
    
    
    
    %Save images:
%     fwrite(fid,warped_image,'double');
    fwrite(fid_shifts,X_warp,'double');
    fwrite(fid_shifts,Y_warp,'double');
    writeVideo(video_file_writer,warped_image);
    toc 
end
% fclose(fid);
fclose(fid_shifts);
close(video_file_writer);



% %show filtered warp:
% figure(3) 
% imagesc(X_warp);  
% colorbar;
% title('X axis warp translation map');

% %show images:
% figure(1);
% imagesc(original_image);
% figure(2);
% imagesc(warped_image);



