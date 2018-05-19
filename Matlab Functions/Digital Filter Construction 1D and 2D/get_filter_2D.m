function [filter_2D] = get_filter_2D(window_name,filter_type_low_high_band,filter_order,filter_parameter,low_cutoff_pixels_per_cycle,high_cutoff_pixels_per_cycle)
%filter length = filter_order + 1

% it's easier in 2D to think in terms of pixels/cycle rather than Fs and frequenies.
% therefore the spatial frequencies will be stated in terms of pixels/cycle so that

% a lowpass filter with low_cutoff_pixels=3 will filter out any frequency above 3[pixels/cycle] so that
% and only structures larger than 3 pixels will survive etc', such that 1D
% filter design logic is reversed

%a highpass filter with high_cutoff_pixel=2 will filter out any frequency
%below 2[pixels/cycle], which is the max, such that only structures smaller
%than 2 pixels will survive...which is nothing.

%if i want a really tight lowpass such that only DC approximately will
%remain i should use low_cutoff_pixels=inf or something like that, because
%this would mean only very large structures remain

% an outcome of this is that we can't state frequecies lower than 2[pixels/cycle] and a lowpass
% with low cutoff of 2[pixels/cycle] will pass everything.

%another outcome is that high_cutoff_pixels_per_cycle < low_cutoff_pixels_per_cycle

%(****) this is less intuitive because here i detached the close intuitive
%relations between filter order (~ filter size) and frequency units, just
%like one does when one creates a 1D filter... this is quick un-intuitive
%but maybe a little bit more flexible.


% %(****) i divide by two to make it more closely resemble the stuff i get when i
% %define speckle size, in the end what i want is that when i define
% %speckle_size=50, that i can use that as a reference "frequency radius" in my functions
% low_cutoff_pixels_per_cycle = low_cutoff_pixels_per_cycle/2;
% high_cutoff_pixels_per_cycle = high_cutoff_pixels_per_cycle/2;

%reverse 1D filter degisn logic, instead multiplying by 1/2 i multiply by 2:
Fs = 1; %1[pixel]
low_cutoff_pixels_per_cycle = (1/low_cutoff_pixels_per_cycle)*(2*Fs);
high_cutoff_pixels_per_cycle = (1/high_cutoff_pixels_per_cycle)*(2*Fs);

%check input to avoid thinking too much:
if strcmp(filter_type_low_high_band,'low')==1
   high_cutoff_pixels_per_cycle = 1; 
end
if strcmp(filter_type_low_high_band,'high')==1
   low_cutoff_pixels_per_cycle = 0; 
end
if strcmp(filter_type_low_high_band,'bandstop')==1 || strcmp(filter_type_low_high_band,'bandpass')==1
   if low_cutoff_pixels_per_cycle > high_cutoff_pixels_per_cycle
      temp = low_cutoff_pixels_per_cycle;
      low_cutoff_pixels_per_cycle = high_cutoff_pixels_per_cycle;
      high_cutoff_pixels_per_cycle = temp;
   end
end 

%create 1D and then 2D filters:
% filter_1D = get_filter_1D(window_name,filter_parameter,filter_order,2*Fs,low_cutoff_pixels_per_cycle,high_cutoff_pixels_per_cycle,filter_type_low_high_band);
filter_1D = get_filter_1D(window_name,filter_parameter,filter_order,Fs,low_cutoff_pixels_per_cycle,high_cutoff_pixels_per_cycle,filter_type_low_high_band);
filter_2D=ftrans2(filter_1D.Numerator);


 


