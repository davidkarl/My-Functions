% % function [zoomed_in_matrix] = zoom_in_using_fourier_interpolation(input_matrix,row_shift,col_shift,number_of_rows_final,number_of_columns_final)
%efficiently "zoom in" using fourier interpolation:
clear all;
clc; 

speckle_size = 50;
ROI = 100;
N = 500;

input_matrix = abs(create_speckles_of_certain_size_in_pixels(speckle_size,N,1));
input_matrix = imread('000000000000000000000001150m  ,500Hz sinus series.jpg');
input_matrix = rgb2gray(input_matrix);
accuracy = 1000;
row_shift=0;
col_shift=0;
% dftshift = fix(ceil(accuracy*1.5)/2);
% % dftshift = 0;
% number_of_rows_final = N;
% number_of_columns_final = N;
% factor=50;
% zoomed_in = conj(dftups(input_matrix,ceil(accuracy*1.5),ceil(accuracy*1.5),accuracy,...
%             dftshift-row_shift*accuracy,dftshift-col_shift*accuracy,factor));
% 
% figure(1)
% imagesc(abs(input_matrix));
% figure(2)
% imagesc(abs(zoomed_in));
        
        
        


    % First upsample by a factor of 2 to obtain initial estimate
    % Embed Fourier data in a 2x larger array
    buf1ft=fft2(input_matrix);
    buf2ft=fft2(input_matrix);
    [m,n]=size(buf1ft);
    mlarge=m*2;
    nlarge=n*2;
    CC=zeros(mlarge,nlarge);
    CC(m+1-fix(m/2):m+1+fix((m-1)/2),n+1-fix(n/2):n+1+fix((n-1)/2)) = fftshift(buf1ft).*conj(fftshift(buf2ft));
  
    % Compute crosscorrelation and locate the peak 
    CC = ifftshift(ifft2(CC)); % Calculate cross-correlation
    [max1,loc1] = max(CC);
    [max2,loc2] = max(max1);
    rloc=loc1(loc2);cloc=loc2;
    CCmax=CC(rloc,cloc);
    
%     % Obtain shift in original pixel grid from the position of the
%     % cross-correlation peak 
    [m,n] = size(CC); md2 = fix(m/2); nd2 = fix(n/2);
%     if rloc > md2 
%         row_shift = rloc - m - 1;
%     else
%         row_shift = rloc - 1;
%     end
%     if cloc > nd2
%         col_shift = cloc - n - 1;
%     else
%         col_shift = cloc - 1;
%     end
%     row_shift=row_shift/2; %we devide by two because in the beginning we upsampled the matrices by 2!
%     col_shift=col_shift/2;

        %%% DFT computation %%%
        % Initial shift estimate in upsampled grid
        row_shift=2;
        col_shift=0;
        row_shift = round(row_shift*accuracy)/accuracy; 
        col_shift = round(col_shift*accuracy)/accuracy;     
        dftshift = fix(ceil(accuracy*1.5)/2); %% Center of output array at dftshift+1-?????? why 1.5?!?!?!?!?
        factor=500;
%         dftshift = fix(ceil(accuracy)*2/factor); %WITH THIS IT WORKS
        dftshift = 0;
        % Matrix multiply DFT around the current shift estimate
%         figure(1) 
%         imagesc(abs(input_matrix));
%         CC = conj(dftups(buf2ft.*conj(buf1ft),ceil(accuracy*1.5),ceil(accuracy*1.5),accuracy,...
%             dftshift-row_shift*accuracy,dftshift-col_shift*accuracy))/(md2*nd2*accuracy^2);
        
        %this returns the original image with a shift [row_shift=2,col_shift=3]:
        CC = conj(dftups(fft2(fftshift(input_matrix)),N,N,1,...
            -1,-1))/(md2*nd2*accuracy^2);
        CC=fftshift(CC);
        CC=fliplr(flipud(CC));
        
        %this also returns the original image with zero shift:
        CC = conj(dftups(fft2(fliplr(flipud(input_matrix))),N,N,1,...
            -1,-1))/(md2*nd2*accuracy^2);
        
        %THIS IS TRIALS:
        factor=2;
%         CC = conj(dftups(fft2(fliplr(flipud(input_matrix))),N,N,factor,...
%            -1,-1))/(md2*nd2*accuracy^2);
       input_matrix = interpft(input_matrix,N,1);
       input_matrix = interpft(input_matrix,N,2);
       figure(1)
       imagesc(input_matrix);
       input_matrix = shift_matrix(input_matrix,1,-size(input_matrix,2)/4,-size(input_matrix,1)/4);
%        figure(2)
%        imagesc(abs(input_matrix));
%        CC = conj(dftups(fft2(input_matrix),N,N,factor,...
%            -1,-1))/(md2*nd2*accuracy^2);
%        CC=flipud(fliplr(CC));
CC = conj(dftups(fft2(fliplr(flipud(input_matrix))),N,N,factor,...
           -factor,-factor))/(md2*nd2*accuracy^2);


%        CC=fftshift(CC);
%         CC=fftshift(CC); 
%         CC=fliplr(flipud(CC)); 
%         CC=circshift(CC,[1,1]); 
        % Locate maximum and map back to original pixel grid
%         CC=abs(CC); 
%         [a,b,c] = return_shifts_between_speckle_patterns(abs(CC),abs(input_matrix),1,3,4,1,1);
%         [col_shift,row_shift,CCmax] = return_shifts_with_fourier_sampling(CC,input_matrix,1,1000);
%        CC=fftshift(CC);
        input_matrix=interpft(input_matrix,factor*N,1);
        input_matrix=interpft(input_matrix,factor*N,2);
        N=N*2;
        input_matrix = input_matrix(N/2-N/4:N/2+N/4-1,N/2-N/4:N/2+N/4-1);
        [col_shift,row_shift,CCmax] = return_shifts_with_fourier_sampling(abs(CC),abs(input_matrix),1,1000);
        figure(2)
        imagesc(abs(input_matrix));
        figure(3)
        imagesc(abs(CC));
        [max1,loc1] = max(CC);   
        [max2,loc2] = max(max1);  
        rloc = loc1(loc2); cloc = loc2;
        CCmax = CC(rloc,cloc);
%         rg00 = dftups(buf1ft.*conj(buf1ft),1,1,accuracy)/(md2*nd2*accuracy^2);
%         rf00 = dftups(buf2ft.*conj(buf2ft),1,1,accuracy)/(md2*nd2*accuracy^2);  
        rloc = rloc - dftshift - 1;
        cloc = cloc - dftshift - 1;
        row_shift = row_shift + rloc/accuracy;
        col_shift = col_shift + cloc/accuracy;    

    
    



