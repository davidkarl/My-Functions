function [filter_current_real] = get_filter_2D_quick(filter_type_low_high_bandpass_bandstop,filter_size,filter_order,low_cutoff,high_cutoff)
%one must be carefull when using the butterworth filter because it is not FIR but IIR,
%the filter_type can be: gaussian or butterworth

%FOR NOW I WILL ONLY USE BUTTERWORTH BECAUSE IT'S SO SUPERIOR:
flag_gaussian_or_butterworth = 2;
% filter_type_low_high_bandpass_bandstop = 'bandstop';
% filter_size = 100;
% filter_order = 30;
% filter_parameter = 10;
% low_cutoff = 20;
% high_cutoff = 30;

%filter length = filter_order + 1

%what i mean by "quick" is that the implementation is quick because it is
%analytic and not interpolatory using chebychev polynomials.

%(****) i divide by two to make it more closely resemble the stuff i get when i
%define speckle size, in the end what i want is that when i define
%speckle_size=50, that i can use that as a reference "frequency radius" in my functions
low_cutoff = low_cutoff/2;
high_cutoff = high_cutoff/2;

% Initialize filter.
%set up grid (including dummy check-up variables):
[X,Y] = meshgrid(1:filter_size); 
filter_center = ceil(filter_size/2) + (1-mod(filter_size,2));
% filter_center = ceil(filter_size/2);
filter_end = floor((filter_size-1)/2);
filter_start = -ceil((filter_size-1)/2);
filter_length = filter_end-filter_start+1;
%set up filter grid:
distance_from_center  = sqrt((X-filter_center).^2 + (Y-filter_center).^2);

%reverse 1D filter degisn logic, instead multiplying by 1/2 i multiply by 2:
Fs = filter_size; %1[pixel]


%check input to avoid thinking too much:
if strcmp(filter_type_low_high_bandpass_bandstop,'low')==1
   if flag_gaussian_or_butterworth==2
       high_cutoff = Fs/2; 
       filter_current = 1./(1 + (distance_from_center/low_cutoff).^(2*filter_order)); 
   elseif flag_gaussian_or_butterworth==1
       high_cutoff = Fs/2;
       filter_current = exp(-(X-filter_center).^2/(low_cutoff.^2) - (Y-filter_center).^2/(low_cutoff.^2));
   end
elseif strcmp(filter_type_low_high_bandpass_bandstop,'high')==1
    if flag_gaussian_or_butterworth==2
       low_cutoff = 0; 
       filter_current = 1 - 1./(1 + (distance_from_center/high_cutoff).^(2*filter_order));
    elseif flag_gaussian_or_butterworth==1
       low_cutoff = 0;
       filter_current = 1-exp(-(X-filter_center).^2/(high_cutoff.^2) - (Y-filter_center).^2/(high_cutoff.^2));
    end
elseif (strcmp(filter_type_low_high_bandpass_bandstop,'bandpass')==1)
   %i made this option dummy proof:
   if low_cutoff<high_cutoff
      warning('i made the filter but you stated frequencies wrong, with bandpass you need lowpass filter radius to be larger so high_cutoff<low_cutoff'); 
   end
   if flag_gaussian_or_butterworth==2
       filter_low_pass = 1./(1 + (distance_from_center/max(low_cutoff,high_cutoff)).^(2*filter_order));
       filter_high_pass = 1 - 1./(1 + (distance_from_center/min(low_cutoff,high_cutoff)).^(2*filter_order));
   elseif flag_gaussian_or_butterworth==1
       filter_low_pass = exp(-(X-filter_center).^2/(max(low_cutoff,high_cutoff).^2) - (Y-filter_center).^2/(max(low_cutoff,high_cutoff).^2));
       filter_high_pass = 1 - exp(-(X-filter_center).^2/(min(low_cutoff,high_cutoff).^2) - (Y-filter_center).^2/(min(low_cutoff,high_cutoff).^2));
   end
   filter_current = filter_low_pass .* filter_high_pass;
elseif (strcmp(filter_type_low_high_bandpass_bandstop,'bandstop')==1)
   %i made this option dummy proof:
   if low_cutoff<high_cutoff
      warning('i made the filter but you stated frequencies wrong, with bandstop you need highpass filter radius to be larger so low_cutoff<high_cutoff'); 
   end
   if flag_gaussian_or_butterworth==2
       filter_low_pass = 1./(1 + (distance_from_center/max(low_cutoff,high_cutoff)).^(2*filter_order));
       filter_high_pass = 1 - 1./(1 + (distance_from_center/min(low_cutoff,high_cutoff)).^(2*filter_order));
   elseif flag_gaussian_or_butterworth==1
       filter_low_pass = exp(-(X-filter_center).^2/(max(low_cutoff,high_cutoff).^2) - (Y-filter_center).^2/(max(low_cutoff,high_cutoff).^2));
       filter_high_pass = 1 - exp(-(X-filter_center).^2/(min(low_cutoff,high_cutoff).^2) - (Y-filter_center).^2/(min(low_cutoff,high_cutoff).^2));
   end
   filter_current = 1 - filter_low_pass .* filter_high_pass; 
end

%get filter in real space for easy insert into filter2:
filter_current_real = real(ift2(filter_current,1))/filter_size^2;

% subplot(3,1,1)
% imagesc(filter_current)
% colorbar;
% subplot(3,1,2)
% filter_current_real = real(ift2(filter_current,1));
% imagesc(filter_current_real);
% colorbar;
% subplot(3,1,3)
% imagesc(real(ft2(filter_current_real,1/filter_size)));
% colorbar;

