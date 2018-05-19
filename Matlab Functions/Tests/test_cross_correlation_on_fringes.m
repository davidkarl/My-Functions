%check cross correlation result on fringe movement:
clear all;
Fs=5000;
Fc=300;
t_duration=2;
t_vec = 0:1/Fs:(round(Fs*t_duration)-1)/Fs;
sine = real(exp(1i*2*pi*Fc*t_vec));
% plot(sine);

fringe_size=10;
current_speckles=create_speckle_fringes_of_certain_size_in_pixels(80,fringe_size,1,0,1024,1,0);
fringe_shift_between_frames = fringe_size*Fc/Fs;
[section_x,section_y] = meshgrid(1:size(current_speckles,1));
fft_speckles = ft2(current_speckles,1);
original_speckles_ROI=abs(current_speckles(1:100,1:100)).^2;
for k=1:100
    tic
%    fft_speckles = ft2(current_speckles,1);
%    imagesc(abs(fft_speckles));
%    drawnow;
   fft_speckles = fft_speckles.*exp(1i*2*pi*Fc*1/Fs*(section_y<size(fft_speckles,1)/2));
   
   %find shift:
   current_speckles = ft2(fft_speckles,1);
   S{k}=abs(current_speckles(1:100,1:100)).^2;
  
   imagesc(abs(current_speckles(1:250,1:250)));
   drawnow;
   
   toc
end

[Dx,Dy]=shift_corelation2_clean(S,3,false);
figure(1)
subplot(1,2,1)
plot((Dy));
subplot(1,2,2)
plot((Dx));











