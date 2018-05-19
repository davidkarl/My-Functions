function [y] = make_sinus_with_phase_shift(Fs,sin_frequency,t_duration,dB_down,phase_shift)

t_vec=0:1/Fs:t_duration; %make time vector
A=10^(-dB_down/20);
y=A*sin(2*pi*sin_frequency*t_vec+phase_shift);
y1(:,1)=y;
y1(:,2)=y;
% wavwrite(y1,Fs,strcat(file_name,'.wav'));