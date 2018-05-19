function [] = freqz_full(current_filter,Fs)

fvtool(current_filter,'Fs',Fs,'magnitudedisplay','zero-phase',...
    'FrequencyRange','Specify freq. vector','FrequencyVector',linspace(-Fs/2,Fs/2-1,4000));


