%make speckle correlation movie:
%basic parameters:
N=256;
ROI=64;
speckle_size = 5;
max_shift = 0.1;
%sound file:
file_name = 'thank you speckle.wma';
% sound_recording = wavread(strcat(file_name,'.wav'));
sound_recording = audioread(file_name);
sound_recording = sound_recording/max(abs(sound_recording))*max_shift;
sound_recording = resample(sound_recording,5000,44100);
%initial speckle pattern:
speckle_pattern_first = create_speckles_of_certain_size_in_pixels(speckle_size,N,1,0);
speckle_pattern_first = abs(speckle_pattern_first).^2;
speckle_pattern_initial = speckle_pattern_first;
%random shift vec:
max_random_shift = 0.1; 
random_shift_x = rand(size(sound_recording));
random_shift_y = rand(size(sound_recording));
%filter for random shift:
[lowpass_filter] = get_filter('kaiser',10,200,5000,200,0,'low');
random_shift_x = filter(lowpass_filter,random_shift_x);
random_shift_x = random_shift_x/max(abs(random_shift_x))*max_random_shift;
random_shift_y = filter(lowpass_filter,random_shift_y);
random_shift_y = random_shift_y/max(abs(random_shift_y))*max_random_shift;
%movie recording:
movie_string = 'speckle correlation movie.avi';
movie_writer = VideoWriter(movie_string);
movie_writer.FrameRate = 5000;
open(movie_writer);
count=1;
%final audio vec:
final_audio_vec_x = zeros(size(sound_recording));
final_audio_vec_y = zeros(size(sound_recording));

tic
while count<length(sound_recording)
   
   %create new speckle pattern:
   speckle_pattern_second = abs(shift_matrix(speckle_pattern_initial,1,sound_recording(count)+random_shift_x(count),sound_recording(count)+random_shift_y(count)));
    
   %find shift:
   [output_vec] = return_shifts_between_speckle_patterns(speckle_pattern_first(1:ROI,1:ROI),speckle_pattern_second(1:ROI,1:ROI),1,3,1,1,1,0);
   shiftx_parabola = output_vec{1};
   shifty_parabola = output_vec{2};
   z_max_parabola = output_vec{3};
   final_audio_vec_x(count) = shiftx_parabola;
   final_audio_vec_y(count) = shifty_parabola;
   
   %record movie frame:
   figure(1);
   imagesc(speckle_pattern_first(1:ROI,1:ROI));
   F_surface = getframe;
   try
   writeVideo(movie_writer,getframe);
   catch
   end
   
   %now second pattern becomes the first:
   speckle_pattern_first = speckle_pattern_second;
    
   %update count:
   count=count+1;
   
   %update clock:
   toc;
end
close(movie_writer);


