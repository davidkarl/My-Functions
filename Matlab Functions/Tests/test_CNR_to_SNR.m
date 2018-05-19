%test CNR to SNR:

N = 100000;
Fs = 44100;
Fc = 12000;
BW = 5000; 
t_vec = my_linspace(0,1/Fs,N)';
x = sin(2*pi*Fc*t_vec);
filter_BW = 1000;
bp_filter = get_filter_1D('hann',10,1024*1,Fs,Fc-filter_BW/2,Fc+filter_BW/2,'bandpass');

CNR_vec = 1:1:20;
y_phase_difference_rms = zeros(size(CNR_vec));
for k=1:length(CNR_vec)
    tic
    CNR = CNR_vec(k);
    [y,noise] = add_noise_of_certain_SNR_over_certain_BW(x,CNR,Fs,BW);
    y_filtered = filtfilt(bp_filter.numerator,1,y);
    y_analytic = hilbert(y_filtered);
    y_analytic = y_analytic .* exp(-1i*2*pi*Fc*t_vec);
    y_phase_difference = angle(y_analytic(2:end).*conj(y_analytic(1:end-1)));
    y_phase_difference_rms(k) = rms(y_phase_difference);
    figure
    plot(y_phase_difference);
    toc
end

% ft = fittype('a*1/(b+x)^3');
% fitted = fit(CNR_vec(:),y_phase_difference_rms(:),ft);
% figure;
% scatter(CNR_vec,y_phase_difference_rms);
% hold on;
% plot(CNR_vec,feval(fitted,CNR_vec));

% close('all');