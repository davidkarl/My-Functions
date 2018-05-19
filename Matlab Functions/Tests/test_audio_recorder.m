% device_name = 'Line (2- SB X-Fi Surround 5.1 Pro)';%'Default';%
% device_data_type = '24-bit integer';
% Fs=44100;
% samples_per_frame=1024;
% output_data_type = 'double';
% audio_recorder_object = dsp.AudioRecorder('OutputNumOverrunSamples',true,'SampleRate',Fs,'SamplesPerFrame',samples_per_frame,...
%     'DeviceName',device_name,'DeviceDataType',device_data_type,...
%     'OutputDataType',output_data_type);
% t = datetime('now','Format','yyyy-MM-dd''T''HHmmss');
% S = char(t);
% filename = ['myTest_',S,'.wav'];
% AFW = dsp.AudioFileWriter(filename,'FileFormat', 'WAV','SampleRate',Fs,'DataType','int32');


%setup audio recorder object
clear all;
Fs=44100;
samples_per_frame = 2048;
audio_recorder_object = dsp.AudioRecorder('SampleRate',Fs,'SamplesPerFrame',samples_per_frame);
audio_recorder_object.DeviceName = 'Line (SB X-Fi Surround 5.1 Pro)';

%medial player object
audio_media_player = dsp.AudioPlayer('SampleRate',Fs);
audio_media_player.DeviceName='Primary Sound Driver';

%setup scope:
array_plot_object = dsp.ArrayPlot;

%Loop:
while 1
bla = step(audio_recorder_object);

step(array_plot_object,bla);
step(audio_media_player,bla);
end

