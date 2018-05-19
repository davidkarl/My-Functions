function [filter_2D] = get_filter_2D_intuitive(window_name,filter_type_low_high_band,filter_size,filter_parameter,low_cutoff_in_pixels,high_cutoff_in_pixels)
%filter length = filter_order + 1

%(****) here, unlike get_filter_2D, the frequency units are intuitive
%because i am matching filter size to filter order TO IMAGE-TO-BE-FILTERED
%size. therefore, i can use intuitive units such that frequency units are
%RADIUS IN PIXELS IN THE FREQUENCY DOMAIN, such that its easier to think of
%when doing speckle analysis.

%for instance, if i want a bandpass filter i need something like:
%(1). filter_low = 10;
%(2). filter_high = 20;
%just like one pictures it when he pictures the frequency domain.

%(****) i divide by two to make it more closely resemble the stuff i get when i
%define speckle size, in the end what i want is that when i define
%speckle_size=50, that i can use that as a reference "frequency radius" in my functions
Fs = filter_size; %1[pixel]
low_cutoff = low_cutoff_in_pixels/2;
high_cutoff = high_cutoff_in_pixels/2;

%check input to avoid thinking too much:
if strcmp(filter_type_low_high_band,'low')==1
   high_cutoff = high_cutoff_in_pixels; 
end
if strcmp(filter_type_low_high_band,'high')==1
   low_cutoff = 0; 
end
if strcmp(filter_type_low_high_band,'bandstop')==1 || strcmp(filter_type_low_high_band,'bandpass')==1
   if low_cutoff > high_cutoff
      temp = low_cutoff;
      low_cutoff = high_cutoff;
      high_cutoff = temp;
   end
end 

%create 1D and then 2D filters:
% filter_1D = get_filter_1D(window_name,filter_parameter,filter_order,2*Fs,low_cutoff_pixels_per_cycle,high_cutoff_pixels_per_cycle,filter_type_low_high_band);
filter_1D = get_filter_1D(window_name,filter_parameter,filter_size,Fs,low_cutoff,high_cutoff,filter_type_low_high_band);
filter_2D=ftrans2(filter_1D.Numerator);


 


