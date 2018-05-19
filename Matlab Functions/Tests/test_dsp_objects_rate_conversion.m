%test dsp objects rate conversion:
clear all;
try
   release(signal_source_object);
   release(fir_decimator_object);
   release(fir_rate_converter_object);
   release(fir_interpolator_object);
   release(cic_decimator_object);
   release(cic_interpolator_object);
   release(logger1);
   release(logger2);
   release(logger3);
   release(logger4);
   release(logger5);
   release(logger6);
   release(logger7);
   release(logger8);
catch
end

%create original stream:
N=100000;
Fs=44100;
three_dB_point = 300; 
n = 4;
std_noise = 0.2;
y = create_noise_with_one_over_f(N,1,Fs,600,n,std_noise)';

%signal source object:
signal_source_object = dsp.SignalSource(y);
signal_source_object.SamplesPerFrame = 4096;

%FIR decimator (default is to decimate by 2):
fir_decimator_object = dsp.FIRDecimator; 
fir_decimator_object.DecimationFactor = 2;

%FIR rate converter (change rate by a rational fraction):
fir_rate_converter_object = dsp.FIRRateConverter;
fir_rate_converter_object.DecimationFactor = 2;
fir_rate_converter_object.InterpolationFactor = 6;

%FIR interpolator:
fir_interpolator_object = dsp.FIRInterpolator; 
fir_interpolator_object.InterpolationFactor = 6;

%CIC decimator:
cic_decimator_object = dsp.CICDecimator;
cic_decimator_object.DecimationFactor = 2;

%CIC interpolator: 
cic_interpolator_object = dsp.CICInterpolator;
cic_interpolator_object.InterpolationFactor = 6;

%regular matlab functions:
% downsample
% upsample
% resample

%logger objects:
logger1 = dsp.SignalLogger;
logger2 = dsp.SignalLogger;
logger3 = dsp.SignalLogger;
logger4 = dsp.SignalLogger;
logger5 = dsp.SignalLogger;
logger6 = dsp.SignalLogger;
logger7 = dsp.SignalLogger;
logger8 = dsp.SignalLogger;

try
   release(signal_source_object);
   release(fir_decimator_object);
   release(fir_rate_converter_object);
   release(fir_interpolator_object);
   release(cic_decimator_object);
   release(cic_interpolator_object);
   release(logger1);
   release(logger2);
   release(logger3);
   release(logger4);
   release(logger5);
   release(logger6);
   release(logger7);
   release(logger8);
catch
end

%GO OVER SIGNAL AND USE EVERYTHING:
while ~isDone(signal_source_object)
   tic
   current_frame = step(signal_source_object);
   current_frame = current_frame(:);
   
   %FIR decimator: 
   fir_decimator_current_frame = step(fir_decimator_object,current_frame);
   %FIR rate converter:
   fir_rate_converter_current_frame = step(fir_rate_converter_object,current_frame);
   %FIR interpolator: 
   fir_interpolator_current_frame = step(fir_interpolator_object,current_frame);
   %CIC decimator:
   cic_decimator_current_frame = step(cic_decimator_object,fi(current_frame));
   %CIC interpolator:
   cic_interpolator_current_frame = step(cic_interpolator_object,fi(current_frame));
   %Downsample:
   downsampled_current_frame = downsample(current_frame,2);
   %Upsample:
   upsampled_current_frame = upsample(current_frame,2);
   %Resample:
   resampled_current_frame = resample(current_frame,6,2);
   
   
   %save results to workspace:
   step(logger1,fir_decimator_current_frame);
   step(logger2,fir_rate_converter_current_frame);
   step(logger3,fir_interpolator_current_frame);
   step(logger4,cic_decimator_current_frame);
   step(logger5,cic_interpolator_current_frame);
   step(logger6,downsampled_current_frame);
   step(logger7,upsampled_current_frame);
   step(logger8,resampled_current_frame);
   toc
end

%Plot results:
list = hsv(9);

%get delays:
fir_decimator_delay = fir_decimator_object.grpdelay;
fir_rate_converter_delay = fir_rate_converter_object.grpdelay;
fir_interpolator_delay = fir_interpolator_object.grpdelay;
cic_decimator_delay = cic_decimator_object.grpdelay;
cic_interpolator_delay = cic_interpolator_object.grpdelay;

x1 = linspace(1+fir_decimator_delay(1),N,length(logger1.Buffer));
x2 = linspace(1+fir_rate_converter_delay(1),N,length(logger2.Buffer));
x3 = linspace(1+fir_interpolator_delay(1),N,length(logger3.Buffer));
x4 = linspace(1+cic_decimator_delay(1),N,length(logger4.Buffer));
x5 = linspace(1+cic_interpolator_delay(1),N,length(logger5.Buffer));
x6 = linspace(1,N,length(logger6.Buffer));
x7 = linspace(1,N,length(logger7.Buffer));
x8 = linspace(1,N,length(logger8.Buffer));
x_original = linspace(1,N,N);

%normalize:
normalization = max(abs(y));

%plot without normaliztion:
figure(1)
plot(x1,logger1.Buffer+1,'col',list(1,:));
hold on;
plot(x2,logger2.Buffer+2,'col',list(2,:));
hold on;
plot(x3,logger3.Buffer+3,'col',list(3,:));
hold on;
plot(x4,logger4.Buffer+4,'col',list(4,:));
hold on;
plot(x5,logger5.Buffer+5,'col',list(5,:));
hold on;
plot(x6,logger6.Buffer+6,'col',list(6,:));
hold on;
plot(x7,logger7.Buffer+7,'col',list(7,:));
hold on;
plot(x8,logger8.Buffer+8,'col',list(8,:));
hold on;
plot(x_original+9,y,'col',list(9,:));
legend('FIR decimator','FIR rate converter','FIR interpolator','CIC decimator','CIC interpolator','downsample','upsample','resample','original');


%plot with normalization:
figure(2)
plot(x1,normalize_signal(logger1.Buffer,normalization)+1,'col',list(1,:));
hold on;
plot(x2,normalize_signal(logger2.Buffer,normalization)+2,'col',list(2,:));
hold on;
plot(x3,normalize_signal(logger3.Buffer,normalization)+3,'col',list(3,:));
hold on;
plot(x4,normalize_signal(double(logger4.Buffer),normalization)+4,'col',list(4,:));
hold on;
plot(x5,normalize_signal(double(logger5.Buffer),normalization)+5,'col',list(5,:));
hold on;
plot(x6,normalize_signal(logger6.Buffer,normalization)+6,'col',list(6,:));
hold on;
plot(x7,normalize_signal(logger7.Buffer,normalization)+7,'col',list(7,:));
hold on;
plot(x8,normalize_signal(logger8.Buffer,normalization)+8,'col',list(8,:));
hold on;
plot(x_original+9,y,'col',list(9,:));
legend('FIR decimator','FIR rate converter','FIR interpolator','CIC decimator','CIC interpolator','downsample','upsample','resample','original');


