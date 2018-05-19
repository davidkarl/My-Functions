%WAVELET TUTORIALS:
addpath('C:\Users\master\Desktop\matlab\data mathworks files');

% %(1). haar, db, sym, coif, bior, rbio, meyr, dmey, gaus, mexh, morl, cgau,
% %shan, fbsp, cmor, fk
% waveinfo('sym'); 
% 
% %(2). there is no scaling function associated with the morlet wavelet
% %the psi length is 7169 samples long....
% [psi_wavelet,x_axis] = wavefun('morl',10);
% plot(x_axis,psi_wavelet);
% title('Morlet wavelet');
% 
% %(3).
% %both the phi_scaling and psi_wavelet is 7169 samples long....
% % 'haar'   : Haar wavelet.
% % 'db'     : Daubechies wavelets.
% % 'sym'    : Symlets.
% % 'coif'   : Coiflets.
% % 'bior'   : Biorthogonal wavelets.
% % 'rbio'   : Reverse biorthogonal wavelets.
% % 'meyr'   : Meyer wavelet.
% % 'dmey'   : Discrete Meyer wavelet.
% % 'gaus'   : Gaussian wavelets.
% % 'mexh'   : Mexican hat wavelet.
% % 'morl'   : Morlet wavelet.
% % 'cgau'   : Complex Gaussian wavelets.
% % 'cmor'   : Complex Morlet wavelets.
% % 'shan'   : Complex Shannon wavelets.
% % 'fbsp'   : Complex Frequency B-spline wavelets.
% % 'fk'     : Fejer-Korovkin orthogonal wavelets
% wavelet_strings = {'db4','sym4','coif4','bior3.5','meyr4','dmey4','gaus4','morl4','cgau4','cmor4','shan4','fbsp4','fk4'};
% [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{1},10,'plot'); %'db4' %orthogonal filter for db4, bior is biorthogonal
% [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{2},10,'plot'); %sym4
% [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{3},10,'plot'); %coif4
% % [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{4},10,'plot'); %'bior3.5' - need more outputs
% % [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{5},10,'plot'); %meyr4 - incorrect use
% % [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{6},10,'plot'); %dmey4 - incorrect use
% % [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{7},10,'plot'); %gaus4 - need more outputs
% % [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{8},10,'plot'); %morl4 - incorrect use morl4
% % [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{9},10,'plot'); %cgau4 - need more outputs
% % [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{10},10,'plot'); %cmor4 - need more INPUTS
% % [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{11},10,'plot'); %shan4 - invalid wavelet number
% % [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{12},10,'plot'); %fbsp4 - invalid wavelet number
% [phi_scaling,psi_wavelet,x_axis] = wavefun(wavelet_strings{13},10,'plot'); %fk4
% subplot(2,1,1);
% plot(x_axis,phi_scaling);
% title('db4 scaling function');
% subplot(2,1,2);
% plot(x_axis,psi_wavelet);
% title('db4 wavelet');
% 
% %(4).
% % Available wavelet names 'wname' are:
% % Daubechies: 'db1' or 'haar', 'db2', ... ,'db45'
% %     Coiflets  : 'coif1', ... ,  'coif5'
% %     Symlets   : 'sym2' , ... ,  'sym8', ... ,'sym45'
% %     Discrete Meyer wavelet: 'dmey'
% % Biorthogonal:
% % 'bior1.1', 'bior1.3' , 'bior1.5'
% % 'bior2.2', 'bior2.4' , 'bior2.6', 'bior2.8'
% % 'bior3.1', 'bior3.3' , 'bior3.5', 'bior3.7'
% % 'bior3.9', 'bior4.4' , 'bior5.5', 'bior6.8'.
% % Reverse Biorthogonal:
% % 'rbio1.1', 'rbio1.3' , 'rbio1.5'
% % 'rbio2.2', 'rbio2.4' , 'rbio2.6', 'rbio2.8'
% % 'rbio3.1', 'rbio3.3' , 'rbio3.5', 'rbio3.7'
% % 'rbio3.9', 'rbio4.4' , 'rbio5.5', 'rbio6.8'.
% [low_pass_analysis_filter,high_pass_analysis_filter,low_pass_synthesis_filter,high_pass_synthesis_filter] ...
%                                                                                         = wfilters('bior3.5');
% %each filter is 12 samples long (understand why)
% subplot(2,2,1);
% plot(low_pass_analysis_filter);
% title('low pass analysis filter');
% subplot(2,2,2);
% plot(high_pass_analysis_filter);
% title('high pass analysis filter');
% subplot(2,2,3);
% plot(low_pass_synthesis_filter);
% title('low pass synthesis filter');
% subplot(2,2,4);
% plot(high_pass_synthesis_filter);
% title('high pass synthesis filter');
% 
% %(5).
% load wecg;
% ecg_signal = wecg;
% plot(ecg_signal);
% title('ECG Signal');
% %for db2 the analysis and synthesis filters are just time reversals
% [low_pass_analysis_filter,high_pass_analysis_filter,low_pass_synthesis_filter,high_pass_synthesis_filter] ...
%                                                                                                 = wfilters('db2');
% %set the padding mode for the DWT to periodization. this does not extend the signal:
% original_dwt_mode = dwtmode('status','nodisplay'); %original_dwt_mode = 'sym' , that is, symmetric extention
% dwtmode('per','nodisplay');  %periodic extention
% 
% %obtain the level one DWT of the ECG signal. the is equivalent to the
% %analysis branch with downsampling of the two channel filter bank:
% [low_pass_level1_coefficients,high_pass_level1_coefficients] = ...
%     dwt(ecg_signal,low_pass_analysis_filter,high_pass_analysis_filter); %notice each output is half the size of the signal!!!! downsampling
% 
% %upsample and interpolate the lowpass (scaling coefficients) and highpass
% %(wavelet coefficients) subbands with the synthesis filters and demonstrate perfect reconstruction:
% ecg_reconstructed = ...
%     idwt(low_pass_level1_coefficients, ...
%          high_pass_level1_coefficients,...
%          low_pass_synthesis_filter,...
%          high_pass_synthesis_filter);
% 
% 
% %demonstrate perfect reconstruction:
% max_reconstruction_error = max(abs(ecg_signal-ecg_reconstructed));
% subplot(2,1,1);
% plot(wecg);
% title('original ecg waveform');
% subplot(2,1,2);
% plot(ecg_reconstructed);
% title('reconstructed ecg waveform');
% 
% 
% %(6). 
% %instead of providing dwt with the analysis filters in the previous example, you can
% %use the string 'db2' instead. 
% [low_pass_level1_coefficients,high_pass_level1_coefficients] = dwt(ecg_signal,'db2');
% ecg_reconstructed = idwt(low_pass_level1_coefficients,high_pass_level1_coefficients,'db2');
% 
% 
% %(7).
% % the filter number refers to the number of vanishing moments. basically, a
% % wavelet with N vanishing moments removes a polynomial of order N-1 in the
% % wavelet coefficients(???).
% %the linear trend is preserved in the scaling coefficients and the wavelet
% %coefficients can be regarded as consisting of only noise:
% t_vec = (0:511)/512;
% y_straight_line_with_noise = (2*t_vec + 0.2*randn(size(t_vec)));
% plot(t_vec,y_straight_line_with_noise);
% [scaling_coefficients_level1,wavelet_coefficients_level1] = dwt(y_straight_line_with_noise,'db2');
% subplot(2,1,1);
% plot(scaling_coefficients_level1);
% title('scaling coefficients');
% subplot(2,1,2);
% plot(wavelet_coefficients_level1);
% title('wavelet coefficients');
% 
% 
% %(8).
% %obtain the level three DWT of the ecg signal using the sym4 orthogonal filter bank.
% %the number of coefficients by level is contained in the vector levels_vec.
% %(*)the first elements of levels_vec is equal to the number of scaling coefficients at level 3 (final level)
% %(*)the second element of levels_vec is the number of wavelet coefficients at level 3
% %(*)subsequent elements give the number of wavelet coefficients at higher
% %levels UNTIL YOU REACH THE FINAL ELEMENT OF levels_vec which is equal to
% %the number of samples in the original signal.
% decomposition_level = 3;
% [coefficients_vec,levels_vec] = wavedec(y_straight_line_with_noise,decomposition_level,'sym4');
% [coefficients_vec,levels_vec] = wavedec(ecg_signal,decomposition_level,'sym4');
% 
% %(9).
% %the scaling and wavelet coefficients are stored in the vector
% %coefficients_vec in the same order. to extract the scaling or wavelet
% %coefficients use 'appcoef' or 'detcoef'. extract all the wavelet
% %coefficients in a cell array and final level scaling coefficients:
% wavelet_coefficients_cell_array = ...
%               detcoef(coefficients_vec,levels_vec,'dcells'); %'dcells' returns cell array - 
% approximation_coefficients_level3 = ...
%               appcoef(coefficients_vec,levels_vec,'sym4'); %extract final level approximation coefficients
% 
% 
% %you can plot the wavelet and scaling coefficients at their approximate positions:
% coefficients_matrix = zeros(numel(ecg_signal),4);
% coefficients_matrix(1:2:end,1) = wavelet_coefficients_cell_array{1}(1:1024); %why is it 1027 from the beginning?!?!
% coefficients_matrix(1:4:end,2) = wavelet_coefficients_cell_array{2}(1:512);
% coefficients_matrix(1:8:end,3) = wavelet_coefficients_cell_array{3}(1:256);
% coefficients_matrix(1:8:end,4) = approximation_coefficients_level3(1:256);
% subplot(5,1,1);
% plot(wecg);
% title('original signal');
% axis tight;
% for kk = 2:4
%    subplot(5,1,kk);
%    plot(coefficients_matrix(:,kk-1), 'marker', 'none');
%    ylabel(['D', num2str(kk-1)]);
%    axis tight;
% end
% subplot(5,1,5);
% plot(coefficients_matrix(:,end),'marker','non');
% ylabel('A3');
% xlabel('Sample');
% axis tight;


% %(10).
% %if you wish to implement an orthogonal wavelet filter bank WITHOUT
% %DOWNSAMPLING, you can use modwt (MAXIMAL OVERLAP DWT):
% %modwt = maximal overlap dwt
% %modwtmra = maximal overlap dwt multi-resolution analysis
% decomposition_level = 3;
% ecg_modwt_coefficients = modwt(ecg_signal,'sym4',decomposition_level); 
% ecg_modwt_approximations_by_level = modwtmra(ecg_modwt_coefficients,'sym4'); %so called imodwt
% subplot(5,1,1);
% plot(wecg);
% title('original signal');
% 
% title('MODWT-based multi-resolution analysis');
% for kk=2:4
%    subplot(5,1,kk);
%    plot(ecg_modwt_approximations_by_level(kk-1,:));
%    ylabel(['D',num2str(kk-1)]);
% end
% subplot(5,1,5);
% plot(ecg_modwt_approximations_by_level(end,:));
% ylabel('A3');
% xlabel('sample');
% 
% 
% %(11).
% %in a biorthogonal filter bank, the synthesis filters are not simply time
% %reversed versions of the analysis filters. the family of biorthogonal
% %spline wavelet filters are an example of such filter banks:
% [low_pass_analysis_filter,high_pass_analysis_filter,low_pass_synthesis_filter,high_pass_synthesis_filter] = ...
%     wfilters('bior3.5');
% 
% %if you examine the analysis filter and the synthesis filters, you can see
% %that they are very different. thesis filter banks still provide perfect reconstruction though:
% [approximation_coefficients,detail_coefficients] = ...
%                 dwt(ecg_signal,low_pass_analysis_filter,high_pass_analysis_filter);
% ecg_reconstructed = ...
%     idwt(approximation_coefficients,detail_coefficients,low_pass_synthesis_filter,high_pass_synthesis_filter);
% max(abs(ecg_signal-ecg_reconstructed));
% 
% %biorthogonal filters are useful when linear phase is a requirement for
% %your filter bank. orthgonal filters cannot have linear phase with the
% %exception of the Haar wavelet filter:
% [low_pass_analysis_filter,high_pass_db6] = wfilters('db6');
% [phase_db6,wavelet_name] = phasez(high_pass_db6,1,512);
% phase_bior35 = phasez(high_pass_analysis_filter);
% figure;
% subplot(2,1,1);
% plot(wavelet_name./(2*pi),phase_db6);
% title('phase response for db6 wavelet');
% grid on;
% xlabel('cycles/sample');
% ylabel('radians');
% subplot(2,1,2);
% plot(wavelet_name./(2*pi),phase_bior35);
% title('phase response for bior3.5 wavelet');
% grid on;
% xlabel('cycles/sample');
% ylabel('radians');
% 
% 
% %(12). ADD WAVELET OF YOUR OWN:
% % First, you must have some way of obtaining the coefficients. In this case, here are the
% % coefficients for the lowpass (scaling) Beylkin(18) filter. You only need a valid scaling
% % filter, wfilters creates the corresponding wavelet filter for you.
% beyl = [9.93057653743539270E-02
% 4.24215360812961410E-01
% 6.99825214056600590E-01
% 4.49718251149468670E-01
% -1.10927598348234300E-01
% -2.64497231446384820E-01
% 2.69003088036903200E-02
% 1.55538731877093800E-01
% -1.75207462665296490E-02
% -8.85436306229248350E-02
% 1.96798660443221200E-02
% 4.29163872741922730E-02
% -1.74604086960288290E-02
% -1.43658079688526110E-02
% 1.00404118446319900E-02
% 1.48423478247234610E-03
% -2.73603162625860610E-03
% 6.40485328521245350E-04];
% Current_DIR = pwd;
% cd(tempdir);
% save beyl beyl;
% familyName = 'beylkin';
% familyShortName = 'beyl';
% familyWaveType = 1;
% familyNums = '';
% fileWaveName = 'beyl.mat';
% wavemngr('add',familyName,familyShortName,familyWaveType,familyNums,fileWaveName)

% load xbox;
% xnox_image = xbox;
% decomposition_level = 1;
% reconstruction_level = 1;
% %     The output wavelet 2-D decomposition structure [C,S]
% %     contains the wavelet decomposition vector C and the 
% %     corresponding bookeeping matrix S. 
% %     Vector C is organized as:
% %       C = [ A(N)   | H(N)   | V(N)   | D(N) | ... 
% %     H(N-1) | V(N-1) | D(N-1) | ...  | H(1) | V(1) | D(1) ].
% %       where A, H, V, D, are row vectors such that: 
% %     A = approximation coefficients, 
% %     H = hori. detail coefficients,
% %     V = vert. detail coefficients,
% %     D = diag. detail coefficients,
% %     each vector is the vector column-wise storage of a matrix. 
% %     Matrix S is such that:
% %       S(1,:) = size of app. coef.(N)
% %       S(i,:) = size of det. coef.(N-i+2) for i = 2,...,N+1
% %       and S(N+2,:) = size(X).
% [coefficients_mat,index_matrix] = wavedec2(xnox_image,decomposition_level,'beyl'); %beyl is not implemented yet
% % [C,S] = wavedec2(xnox_image,1,'sym4');
% [H,V,D] = detcoef2('all',coefficients_mat,index_matrix,reconstruction_level); %WHY IS EACH MATRIX 72X72????
% subplot(2,1,1);
% imagesc(xnox_image);
% axis off;
% title('original image');
% subplot(2,1,2);
% imagesc(D);
% axis off;
% title('level one diagonal coefficients');
% 
% %(13).
% %verify that the new filter satisfies the conditions for an orthogonal QMF
% %pair. obtain the scaling and wavelet filters:
% [low_pass,high_pass] = wfilters('beyl');
% 
% %sub the lowpass filter coefficients to verify that the su equals sqrt(2),
% %then sum the wavelet filter coefficients and verify that the sum is zero:
% sum(low_pass)
% sum(high_pass)
% 
% %to understand why these filters are called quadrature mirror filters,
% %visualize the squared magnitude frequency responses of the scaling and
% %wavelet filters:
% nfft = 512;
% F = 0:1/nfft:1/2;
% low_pass_fft = fft(low_pass,nfft);
% high_pass_fft = fft(high_pass,nfft);
% plot(F,abs(low_pass_fft(1:nfft/2+1)).^2);
% hold on;
% plot(F,abs(high_pass_fft(1:nfft/2+1)).^2,'r');
% legend('scaling filter','wavelet filter');
% xlabel('frequency');
% ylabel('squared magnitude');
% grid on;
% plot([1/4,1/4],[0,2],'k');
% 
% 
% %(14).
% %CWT - continuous wavelet transform.
% %use the bump wavelet- the bump wavelet is a good choice for the cwt when
% %your signals are oscillatory and you are more intereseted in time
% %frequency analysis than localization of transient. (THAN WHAT IS GOOD FOR
% %THAT STUFF????).
% % cwtft outputs a structure array with seven fields:
% % cfs:         CWT coefficients
% % scales:      vector of scales.
% % frequencies: frequencies in cycles per unit time (or space)
% % corresponding to the scales. If the sampling period
% % units are seconds, the frequencies are in hertz.
% % The elements of frequencies are in decreasing order to
% % correspond to the elements in the scales vector. Use
% % this field to examine the CWT in the time-frequency
% % plane.
% % wav:         wavelet used for the analysis (see WAV below).
% % omega:       angular frequencies used in the Fourier transform of
% % the wavelet. This field is used in ICWTFT and ICWTLIN
% % for the inversion of the CWT.
% %     meanSIG:     mean of SIG
% %     dt:          sampling period
% %     
% %     CWTSTRUCT = cwtft(SIG,'scales',SCA,'wavelet',WAV) lets you
% %     define the scales and the wavelet. Supported analyzing wavelets are:
% %     
% %     'morl' -    Morlet wavelet (analytic)
% %     'morlex' -  Morlet wavelet (nonanalytic)
% %     'morl0' -   Exact zero-mean Morlet wavelet (nonanalytic)
% %     'bump' -    Bump wavelet (analytic)
% %     'paul' -    Paul wavelet (analytic)
% %     'dog'  -    N-th order derivative of Gaussian (nonanalytic)
% %     'mexh' -    Second derivative of Gaussian (nonanalytic)
% load quadchirp;
% quad_chirp_signal = quadchirp;
% dt = 1/1000;
% max_possible_frequency = (1/dt) / 2;
% number_of_points_per_octave = 32; %voices???
% a0 = 2^(1/number_of_points_per_octave); %?
% %where do i get different wavelet's center frequencies???
% wavelet_center_frequency = 5/(2*pi); %bump wavelet center frequency
% min_frequency = 20; %Hz
% max_frequency = 500; %Hz
% min_scale = wavelet_center_frequency/(max_frequency*dt); %UNDERSTAND THIS!!!
% max_scale = wavelet_center_frequency/(min_frequency*dt);
% min_scale = floor(number_of_points_per_octave*log2(min_scale)); %number of time units
% max_scale = ceil(number_of_points_per_octave*log2(max_scale));
% scales = a0.^(min_scale:max_scale) .* dt; %use this to contstruct a function which returns a scales_vec using min and max frequencies
% cwt_quad_chirp = cwtft({quad_chirp_signal,dt},'wavelet','bump','scales',scales);
% helperCWTTimeFreqPlot(cwt_quad_chirp.cfs , tquad, cwt_quad_chirp.frequencies, ...
%              'suft','cwt of quadratic chirp - bump wavelet','seconds','Hz'); %what is this and what is tquad
% 
% 
% 
% 
% 
% %(15). REMOVE TIME LOCALIZED FREQUENCY COMPONENTS:
% dt = 1/2000;
% t_vec = 0 : dt : 1-dt;
% x1 = sin(50*pi*t_vec).*exp(-50*pi*(t_vec-0.2).^2);
% x2 = sin(50*pi*t_vec).*exp(-100*pi*(t_vec-0.5).^2);
% x3 = 2*cos(140*pi*t_vec).*exp(-50*pi*(t_vec-0.2).^2);
% x4 = 2*sin(140*pi*t_vec).*exp(-80*pi*(t_vec-0.2).^2);
% super_imposed_sine_signal = x1+x2+x3+x4;
% plot(t_vec,super_imposed_sine_signal);
% grid on;
% title('superimposed signal');
% 
% %obtain the CWT using the bump wavelet and display the result as a function of time and frequency:
% s0 = 2; %first scale as a multiple of (2*dt) (or is it simply dt????)
% a0 = 2^(1/32); %2^(1/number_of_points_per_decade)
% scales = (s0*a0.^(32:7*32)).*dt; %2*2^7*dt = final scale
% cwt_coefficients = cwtft({super_imposed_sine_signal,dt},'scales',scales,'wavelet',{'bump',[4,0.9]}); %what does the [4,0.9] stand for?!?!!?
% figure;
% contour(t_vec,cwt_coefficients.frequencies,abs(cwt_coefficients.cfs));
% xlabel('seconds');
% ylabel('Hz');
% grid on;
% title('analytic cwt using bump wavelet');
% hcol = colorbar;
% hcol.Label.String = 'magnitude';
% 
% %remove the 25Hz component which occurs from approximately 0.05 to 0.35
% %seconds by zeroing out the CWT coefficients. use the inverse CWT (icwtft)
% %to reconstruct an approximation to the signal:
% frequency_indices = (cwt_coefficients.frequencies>19 & cwt_coefficients.frequencies<31); %Hz
% time_indices = 100:700; 
% cwt_coefficients2 = cwt_coefficients;
% cwt_coefficients2.cfs(frequency_indices,time_indices) = 0; %coefficients_mat(frequency_indices,time_indices)
% ecg_reconstructed = icwtft(cwt_coefficients2);
% 
% subplot(2,1,1);
% plot(t_vec,super_imposed_sine_signal);
% grid on;
% title('original signal');
% subplot(2,1,2);
% plot(t_vec,ecg_reconstructed);
% grid on;
% title('signal with the first 25Hz component removed');
% 
% %compare the reonstructed signal with the original signal without the 25Hz
% %component centered at 0.2 seconds:
% y = x2+x3+x4;
% figure
% plot(t_vec,ecg_reconstructed);
% hold on;
% plot(t_vec,y,'r--');
% grid on;
% legend('inverse cwt approximation','original signal without 25Hz');
% hold off;
% 
% 
% %(16). WAVELET COHERENCE
% load NIRSData;
% figure;
% plot(tm,NIRSData(:,1));
% hold on;
% plot(tm,NIRSData(:,2),'r');
% legend('subject 1', 'subject 2','location','northwest');
% xlabel('seconds');
% title('NIRS Data');
% 
% %examining the time-domain data, it is not clear what oscillations are
% %present in the individual time series, or what oscillations are common to
% %both data sets. use wavelet analysis to answer both questions:
% scalesCWT = helperCWTTimeFreqVector(0.03,2,5/(2*pi),1/10,32);
% cwtsubj1 = cwtft({NIRSData(:,1),1/10},'wavelet','bump','scales',scalesCWT,'padmode','symw');
% cwtsubj2 = cwtft({NIRSData(:,2),1/10},'wavelet','bump','scales',scalesCWT,'padmode','symw');
% subplot(2,1,1);
% helperCWTTimeFreqPlot(cwtsubj1.cfs,tm,cwtsubj1.frequencies,'surf','subject 1','seconds','hz');
% set(gca,'ytick',[0.15,0.6,1.2,1.8]);
% subplot(2,1,2);
% helperCWTTimeFreqPlot(cwtsubj2.cfs,tm,cwtsubj2.frequencies,'surf','subject 2','seconds','hz');
% set(gca,'ytick',[0.15,0.6,1.2,1.8]);
% 
% scales = helperCWTTimeFreqVector(0.01,2,5,1/10,12); %???
% dt = 1/10;
% 
% %USE MATLAB'S wcoher!!!:
% %WCOHER Wavelet coherence.
% %	For two signals S1 and S2, WCOH = WCOHER(S1,S2,SCALES,WAME)
% %   returns the Wavelet Coherence (WCOH).
% %   SCALES is a vector which contains the scales, and WNAME is a 
% %   string containing the name of the wavelet used for the continuous 
% %   wavelet transform.
% %
% %   In addition, [WCOH,WCS] = WCOHER(...) returns also the
% %   Wavelet Cross Spectrum (WCS).
% %
% %   In addition, [WCOH,WCS,CWT_S1,CWT_S2] = WCOHER(...) returns 
% %   also the continuous wavelet transforms of S1 and S2.
% %
% %   [...] = WCOHER(...,'ntw',VAL,'nsw',VAL) allows to smooth the 
% %   CWT coefficients before computing WCOH and WCS. Smoothing
% %   can be done in time or scale, specifying in each case the width 
% %   of the window using positive integers:
% %       'ntw' : N-point time window  (defaut is min[20,0.05*length(S1)])
% %       'nsw' : N-point scale window (default is 1).
% %
% %   [...] = WCOHER(...,'plot') displays the modulus and phase 
% %   of the Wavelet Coherence (WCOH).
% %
% %   [...] = WCOHER(...,'plot',TYPEPLOT) allows to display other plots.
% %	The valid values for TYPEPLOT are:
% %       'wcoh' : More on WCOH phase is displayed.
% %       'wcs'  : WCS is displayed.
% %       'cwt'  : Continuous wavelet transforms are displayed.
% %       'all'  : All the outputs are displayed.
% %
% %   Arrows representing the phase are displayed on the Wavelet
% %   Coherence plots. 
% %   [...] = WCOHER(...,'nat',VAL,'nas',VAL,'ars',ARS) allows to 
% %   change the number and the scale factor for the arrows (see QUIVER):
% %       'nat' : number of arrows in time.
% %       'nas' : number of arrows in scale.
% %       'asc' : scale factor for the arrows.
% %       ARS = 2 doubles their relative length, and ARS = 0.5 
% %       halves the length.
% wavelet_coherence = wcoher(NIRSData(:,1),NIRSData(:,2),scales./dt,'cmor0.5-5','nsw',8); %???
% 
% %USE MATLAB'S scal2freq!!!!:
% frequencies = scal2frq(scales./dt,'cmor0.5-5',dt);
% 
% figure;
% surf(tm,frequencies,abs(wavelet_coherence).^2);
% view(0,90);
% shading interp;
% axis tight;
% hc = colorbar;
% hc.Label.String = 'coherence';
% title('wavelet coherence');
% xlabel('seconds');
% ylabel('Hz');
% set(gca,'ytick',[0.15,0.6,1.2,1.8]);
% 
% 
% %(17).
% %continuous wavelet transform complex gaus:
% load cuspamax;
% % c = cwt(cuspamax, 1:2:64, 'cgau4');
% coefficients_mat = cwt(cuspamax, 1:2:64, 'cgau4', 'plot');
% 
% %(18).
% %cwt of sum of disjoint sinusoids:
% N = 1024;
% t_vec = linspace(0,1,N);
% dt = 1/(N-1);
% Y = sin(8*pi*t_vec).*(t_vec<=0.5) + sin(16*pi*t_vec).*(t_vec>0.5);
% 
% %obtain the continuous wvaelet transform using the default analytic morlet
% %wavelet, and plot the results. WHAT'S THE DIFFERENCE BETWEEN cwt and cwtft?????
% signal = {Y,dt};
% cwtS1 = cwtft(signal,'plot'); %cool, entering cell into cwtft
% 
% %specify the analyzing wavelet as the Paul wavelet of order 8. specify the
% %initial scale, the spacing between scales, and the number of scales. by
% %degault, the scale vector is logarithmic to the base 2:
% 
% %smallest scale, spacing between scales, number of scales:
% dt = 1/1023;
% s0 = 2*dt; %smallest scale
% ds = 0.5; %spacing between scales
% number_of_scales = 20; %number of scales
% %scale vector is: scales = s0*2.^((0:NbSc-1)*ds);
% wavelet_name = 'paul';
% SIG = {Y,dt};
% %create SCA input as cell array:
% % SCA can be a vector, a structure array, or a cell array.
% % If SCA is a vector, it contains the scales.
% % If SCA is a structure, it may contain at most five fields
% % (s0,ds,nb,type,pow). The last two fields are optional.
% % s0, ds, and nb are respectively the smallest scale, the spacing
% % between scales, and the number of scales. The field, type, determines
% % the spacing used for the scales vector: 'pow' (logarithmic spacing)
% % which is the default or 'lin' (linear spacing).
% % For 'pow' : scales = s0*pow.^((0:nb-1)*ds);
% % For 'lin' : scales = s0 + (0:nb-1)*ds;
% % When type is 'pow', if SCA.pow exists SCA.pow = pow. Otherwise, pow is
% %     set to 2.
% %     If SCA is a cell array, SCA{1}, SCA{2}, and SCA{3} contain
% %     the smallest scale, the spacing between scales,
% %     and the number of scales. If SCA{4} and SCA{5} exist, they
% %     contain the type of scaling and the power.
% SCA = {s0,ds,number_of_scales}; %understand
% %specify wavelet and parameters:
% WAV = {wavelet_name,8};
% %compute and plot the CWT:
% cwtS2 = cwtft(SIG,'scales',SCA,'wavelet',WAV,'plot');


% %(18). APPROXIMATE SCALE-FREQUENCY CONVERSIONS:
% %there is not a direct correspondence between fourier wavelength and scale.
% %however, you can find conversion factors for select wavelets that yield an
% %approximate scale-frequency correspondence. you can find these factors for
% %wavelets supported by cwtft listed on the reference page.
% 
% %this example shows you how to change the scale axis to an approximate
% %frequency axis for analysis. use the sum of disjint sinusoids as the input
% %signal. set the initial scale to 6*dt. the scale increment to 0.15, and
% %the number of scales to 50. use the fourier factor for the morlet wavelet
% %to convert the scale vector to an approximate frequency vector in Hz. plot
% %the result:
% figure;
% s0 = 6*dt; %initial scale
% ds = 0.15; %scale increment
% number_of_scales = 50;
% wavelet_name = 'morl';
% SCA = {s0,ds,number_of_scales};
% cwtsig = cwtft({Y,dt},'scales',SCA,'wavelet',wavelet_name);
% MorletFourierFactor = 4*pi/(6+sqrt(2+6^2));
% Scales = cwtsig.scales .* MorletFourierFactor;
% Freq = 1./Scales;
% imagesc(t_vec,[],abs(cwtsig.cfs));
% indices = get(gca,'ytick');
% set(gca,'yticklabel',Freq(indices));
% xlabel('Time');
% ylabel('Hz');
% title('Time-Frequency Analysis with CWT');
% 
% %Repeat the above example by using the Paul analyzing wavelet with order m
% %equal to 8, use a contour plot of the real part of the CWT to visualize
% %the sine waves at 4 and 8 Hz.
% %The real part exhibits oscillations in the sign of the wavelet
% %coefficients at those frequencies:
% s0 = 6*dt;
% ds = 0.15;
% number_of_scales = 50; %number of scales
% m = 8; %wavelet order 
% %scale vector is: scales = s0*2.^((0:NbSc-1)*ds);
% wavelet_name = 'paul';
% SIG = {Y,dt};
% %create SCA input as cell array:
% SCA = {s0,ds,number_of_scales};
% %specify wavelet and parameters:
% WAV = {wavelet_name,m}; %wavelet cell array'
% cwtPaul = cwtft(SIG,'scales',SCA,'wavelet',WAV);
% scales = cwtPaul.scales;
% PaulFourierFactor = 4*pi/(2*m+1); %how do i get the conversion factors for other wavelets?!!!?
% Freq = 1./(PaulFourierFactor.*scales);
% contour(1:4096,Freq,real(cwtPaul.cfs)); %????
% xlabel('time');
% ylabel('Hz');
% colorbar;
% title('Real part of CWT using Paul wavelet (m=8)');
% axis([0,1,1,15]);
% grid on;


% %(19). SIGNAL RECONSTRUCTION FOR CWT COEFFICIENTS:
% %the inversion of the CWT is not as straightforward as the DWT. the
% %simplest CWT inversion utilizes the single integral dormula due to Morlet,
% %which employs a diract delta function as the synthesizing wavelet. icwtft
% %and icwtlin both implement the single integral formula. because of
% %nevessary approximation in the implementation of the single integral
% %inverse CWT, you cannot expect to obtain perfect reconstruction. however,
% %you can use the inverse CWT to obtain useful position and scale-dependent
% %approximation to the input signal.
% 
% %implement the inverse CWT with logarithmically-spaced scales:
% N = 1024;
% t_vec = linspace(0,1,N);
% dt = 1/(N-1);
% Y = sin(8*pi*t_vec).*(t_vec<=0.5) + sin(16*pi*t_vec).*(t_vec>0.5);
% dt = 1/1023;
% s0 = 2*dt;
% ds = 0.5;
% number_of_scales = 20;
% wavelet_name = 'paul';
% SIG = {Y,dt};
% SCA = {s0,ds,number_of_scales}; %it doesn't seem as if there are only NbSc=20 scales in the plot but there are 20 in the coefficients returns in cwts2
% WAV = {wavelet_name,8};
% cwts2 = cwtft(SIG,'scales',SCA,'wavelet',WAV);
% YR1 = icwtft(cwtS2,'plot','signal',SIG); %Y_reconstructed
% norm(Y-YR1,2);
% 
% %obtain the CWT of a noisy doppler signal using the analytic morlet
% %wavelet. reconstruct an approximation by selecting a subset of theCWT
% %coefficients. 
% %BY ELIMINATING THE SMALLEST SCALES, YOU OBTAIN A LOWPASS APPROXIMATION TO
% %THE SIGNAL.
% %the lowpass approximation produces a smooth approximation to the lower
% %frequency features of the noisy doopler signal. the high frequency (small
% %scale) features at the beginning of the signal are lost.
% load noisdopp;
% noisy_doppler_signal = noisdopp;
% N = length(noisy_doppler_signal);
% dt = 1;
% s0 = 2*dt; %check whether i can use simply dt
% ds = 0.4875; 
% number_of_scales = 20;
% wavelet_name = 'morl';
% SIG = {noisy_doppler_signal,dt};
% SCA = {s0,ds,number_of_scales};
% WAV = {wavelet_name,[]}; %????
% cwtS4 = cwtft(SIG,'scales',SCA,'wavelet',WAV);
% %Thresholding step building the new structure:
% cwtS5 = cwtS4;
% newCFS = zeros(size(cwtS4.cfs));
% newCFS(11:end,:) = cwtS4.cfs(11:end,:);
% cwtS5.cfs = newCFS;
% %reconstruction from the modified structure:
% YRDen = icwtft(cwtS5,'signal',SIG);
% plot(noisy_doppler_signal,'k-.');
% hold on;
% plot(YRDen,'r','linewidth',3);
% axis tight;
% legend('original signal','selective inverse DWT');
% title('signal approximation based on a subset of CWT coefficients');


% %(20). 2D CWT of noisy pattern:
% %use the isotropic (non-directional) mexican hat wavelet and the
% %anisotropic (directional) Morlet wavelet. demonstrate that the real-valued
% %maxican hat wavelet does not depend on the angle:
% Y = zeros(32,32); 
% Y(16,16) = 1;
% cwtmexh = cwtft2(Y,'wavelet','mexh','scales',1,'angles',[0,pi/2]); %angles?????
% surf(real(cwtmexh.cfs(:,:,1,1,1))); %what does each dimension here symbolize????
% shading interp;
% title('angle=0 radians');
% %extract the wavelet corresponding to an angle of pi/2 radians, it's the same as 0 radians:
% surf(real(cwtmexh.cfs(:,:,1,1,2)));
% shading interp;
% title('angle = pi/2 radians'); 
% %now use the complex valued morlet wavelet:
% Y = zeros(64,64);
% Y(32,32) = 1;
% cwtmorl = cwtft2(Y,'wavelet','morl','scales',1,'angles',[0,pi/2]);
% surf(abs(cwtmorl.cfs(:,:,1,1,1)));
% shading interp;
% title('angle=0 radians');
% surf(abs(cwtmorl.cfs(:,:,1,1,2)));
% shading interp;
% title('angle=pi/2 radians');

% %apply the mexican hat and morlet wavelets to the detection of a pattern in
% %noise. create a pattern consisting of line segments joined at a 90-degree
% %angle. the amplitude of the pattern is 3 and it occurs in additive N(0,1)
% %white gaussian noise:
% x = zeros(256,256);
% x(100:200,100:102) = 3;
% x(200:202,100:125) = 3;
% x = x + randn(size(x));
% imagesc(x);
% axis xy;
% cwtmexh = cwtft2(x,'wavelet','mexh','scales',3:0.5:8);
% figure;
% surf(abs(cwtmexh.cfs(:,:,1,3,1)).^2); %as i put a larger number (scale) in the next to last argument the result is more lowpassed
% view(0,90);
% shading interp;
% axis tight;
% %use a directional morlet wavelet to seperately extract the vertical and
% %horizontal line segments:
% cwtmorl = cwtft2(x,'wavelet','morl','scales',3:0.5:8,'angles',[0,pi/2]);
% surf(abs(cwtmorl.cfs(:,:,1,4,1)).^2);
% view(0,90);
% shading interp;
% axis tight;
% %as you can see the vertical line segment is extracted by one angle and the
% %horizontal line segment by the other one.



% %(21). DOUBLE DENSITY WAVELET TRANSFORM:
% % dtfilters Dual-Tree filters
% % [DF,RF] = dtfilters(NAME) returns the decomposition filters
% % and the associated reconstruction filters which name
% % is given by the string NAME.
% % 
% % The valid values for NAME are:
% %         - 'dtfP': returns the filters for the first step and the following steps, for dual-tree ('FSfarras' for the first
% %             step and  'qshiftN' with P = 1, 2, 3 or 4 corresponding to N = 6, 10, 14 or 18 for the following steps) .
% %         - 'dddtf1': returns the filters for the first step and the following steps, for double density dual-tree ('FSdoubledualfilt' and 'doubledualfilt').
% %         - 'self1','self2': I W. Selesnick - 3-channel perfect reconstruction  filter bank
% %         - 'filters1','filters2': NOT DOCUMENTED [] 3-channel perfect reconstruction filter bank
% %         - 'farras','FSfarras' (first step filter): nearly symmetric filters for orthogonal 2-channel perfect reconstruction filter bank
% %         - 'qshiftN' with N = 6, 10, 14 or 18 Kingsbury Q-filters for the dual-tree complex DWT
% %         - 'doubledualfilt','FSdoubledualfilt' (first step filter): DF and RF are cell arrays of 3-channel filter bank
% %         - 'AntonB': NOT DOCUMENTED []
% %         - Orthogonal or biorthogonal wavelet names (see WFILTERS)
% %             F = dtfilters(NAME,TYPE) returns the following filters:
% %             Decomposition filters if type = 'd'
% %             Reconstruction filters if type = 'r'
% x = zeros(256,1);
% decomposition_level = 5;
% decomposition_filters = dtfilters('filters1'); %returns 3 filters: low pass & 2 high passes
% wt1 = dddtree('ddt',x,decomposition_level,decomposition_filters,decomposition_filters); %Double Density Dual tree
% wt2 = dddtree('ddt',x,decomposition_level,decomposition_filters,decomposition_filters);
% wt1.cfs{5}(5,1,1) = 1;
% wt2.cfs{5}(5,1,2) = 1;
% wav1 = idddtree(wt1);
% wav2 = idddtree(wt2);
% plot(wav1);
% hold on;
% plot(wav2,'r');
% axis tight;
% legend('\psi_1(t)','\psi_2(t).');
% 
% %you can obtain wavelet analysis and synthesis frames for the double
% %density wavelet transform with 6 and 12 taps using dtfilters:
% [df1,sf1] = dtfilters('filters1');
% [df2,sf2] = dtfilters('filters2');
% %df1 and df2 are three column matrices containing the analaysis filters.
% %the first column contains the scaling filters and columns two and three
% %contain the wavelet filters. the corresponding synthesis filters are in sf1 and sf2.



%(22). 1D DECIMATED WAVELET TRANSFORMS:
% dwt = single level decomposition
% wavedec = decomposition
% wmaxlev = maximum wavelet decomposition level
% idwt = single level reconstruction
% waverec = full reconstruction
% wrcoef = selective reconstruction
% upcoef = single reconstruction 
% detcoef = extraction of detail coefficients
% appcoef = extraction of approximation coefficients
% upwlev = reconposition of decomposition structure
% ddencmp = provide default values for denoising and compression
% wbmpen = penalized threshold for wavelet 1D and 2D denoising
% wdcbm = thresholds for wavelet 1D using Birge-Massart strategy ??????
% wdencmp = wavelet denoising and compression
% wden = automatic wavelet denoising
% wthrmngr = threshold settings manager


% %(23). 1D analysis using the command line:
% load leleccum;
% electricity_use_signal = leleccum(1:3920);
% signal_length = length(electricity_use_signal);
% %perform a single level wavelet decomposition using the db1 wavelet:
% [approximation_coefficients_level1,details_coefficients_level1] = dwt(electricity_use_signal,'db1'); %cA1 = approximation coefficients, cD1 = detail coefficients
% %reconstruct approximations and details from the coefficients:
% %possibility 1
% signal_approximation_using_approximation_coefficients = upcoef('a',approximation_coefficients_level1,'db1',1,signal_length); %dont i need both the approximation and detail coefficients for reconstruction
% signal_approximation_using_detail_coefficients = upcoef('d',details_coefficients_level1,'db1',1,signal_length);
% %possibility 2
% signal_approximation_using_approximation_coefficients = idwt(approximation_coefficients_level1,[],'db1',signal_length);
% signal_approximation_using_detail_coefficients = idwt([],details_coefficients_level1,'db1',signal_length);
% %disply the approximation and detail:
% subplot(1,2,1);
% plot(signal_approximation_using_approximation_coefficients);
% title('approximation A1');
% subplot(1,2,2);
% plot(signal_approximation_using_detail_coefficients);
% title('detail D1');
% %regenerate a signal by using the inverse wavelet transform:
% electricity_use_signal_reconstructed = idwt(approximation_coefficients_level1,details_coefficients_level1,'db1',signal_length);
% error = max(abs(electricity_use_signal-electricity_use_signal_reconstructed));
% %perform a multilevel wavelet decomposition of a signal:
% %perform a level 3 decomposition of a signal using the db1 wavelet:
% [coefficients_mat,levels_vec] = wavedec(electricity_use_signal,3,'db1');
% %extract the level 3 approximation coefficients from C:
% cA3 = appcoef(coefficients_mat,levels_vec,'db1',3);
% %extract the levels 3,2, and 1 detail coefficients from C:
% details_coefficients_level3 = detcoef(coefficients_mat,levels_vec,3);
% details_coefficients_level2 = detcoef(coefficients_mat,levels_vec,2);
% details_coefficients_level1 = detcoef(coefficients_mat,levels_vec,1);
% %another way of doing it:
% [details_coefficients_level1,details_coefficients_level2,details_coefficients_level3] = detcoef(coefficients_mat,levels_vec,[1,2,3]);
% %reconstruct the level 3 approximation and the level 1,2, and 3 detail:
% %reconstruct the level 3 approximation from C:
% A3 = wrcoef('a',coefficients_mat,levels_vec,'db1',3); %wavelet reconstruct from coefficients = wrcoef
% %reconstruct the details at level 1,2,3 from C:
% signal_approximation_using_detail_coefficients = wrcoef('d',coefficients_mat,levels_vec,'db1',1);
% D2 = wrcoef('d',coefficients_mat,levels_vec,'db1',2);
% D3 = wrcoef('d',coefficients_mat,levels_vec,'db1',3);
% %display the results of the level 3 decomposition:
% subplot(2,2,1);
% plot(A3);
% title('approximation A3');
% subplot(2,2,2);
% plot(signal_approximation_using_detail_coefficients);
% title('detail D1');
% subplot(2,2,3);
% plot(D2);
% title('detail D2');
% subplot(2,2,4);
% plot(D3);
% title('detail D3');
% %reconstruct the original signal from the level 3 decomposition:
% %to reconstruct the original signal from the wavelet decomposition
% %structure:
% electricity_use_signal_reconstructed = waverec(coefficients_mat,levels_vec,'db1');
% max_reconstruction_error = max(abs(electricity_use_signal-electricity_use_signal_reconstructed));
% %remove noise by thresholding:
% %to denoise the signal, use the ddencmp command to calculate the default
% %parameters and the wdencmp command to perform the actual denoising:
% % ddencmp Default values for de-noising or compression.
% %     [THR,SORH,KEEPAPP,CRIT] = ddencmp(IN1,IN2,X)
% %     returns default values for de-noising or compression,
% %     using wavelets or wavelet packets, of an input vector
% %     or matrix X which can be a 1-D or 2-D signal.
% %     THR is the threshold, SORH is for soft or hard
% %     thresholding, KEEPAPP allows you to keep approximation
% %     coefficients, and CRIT (used only for wavelet packets)
% %     is the entropy name (see WENTROPY).
% %     IN1 is 'den' or'cmp' and IN2 is 'wv' or 'wp'.
% [threshold,soft_or_hard_threshold,keep_approximation] = ddencmp('den','wv',electricity_use_signal);
% %  wdencmp De-noising or compression using wavelets.
% %     wdencmp performs a de-noising or compression process
% %     of a signal or an image using wavelets.
% %  
% %     [XC,CXC,LXC,PERF0,PERFL2] = 
% %     wdencmp('gbl',X,'wname',N,THR,SORH,KEEPAPP)
% %     returns a de-noised or compressed version XC of input
% %     signal X (1-D or 2-D) obtained by wavelet coefficients
% %     thresholding using global positive threshold THR.
% %     Additional output arguments [CXC,LXC] are the
% %     wavelet decomposition structure of XC, 
% %     PERFL2 and PERF0 are L^2 recovery and compression
% %     scores in percentages.
% %     PERFL2 = 100*(vector-norm of CXC/vector-norm of C)^2
% %     where [C,L] denotes the wavelet decomposition structure
% %     of X.
% %     Wavelet decomposition is performed at level N and
% %     'wname' is a string containing the wavelet name.
% %     SORH ('s' or 'h') is for soft or hard thresholding
% %     (see WTHRESH for more details).
% %     If KEEPAPP = 1, approximation coefficients cannot be
% %     thresholded, otherwise it is possible.
% clean = wdencmp('gbl',coefficients_mat,levels_vec,'db1',3,threshold,soft_or_hard_threshold,keep_approximation);
% %not that wdencmp uses the results of the decomposition (C and L) that we
% %calculated. we also specify that we used the db1 wavelet to perform the
% %original analysis, and we specify the global thresholding option 'gbl'. 
% subplot(2,1,1);
% plot(index_matrix(2000:3920));
% title('original');
% subplot(2,1,2);
% plto(clean(2000:3920));
% title('denoised');


% %(24). 
% %it is interesting to notice that if arbitrary extension is used and
% %decomposiiton performed using the convolution downsampling schene,
% %perferct reconstruction is recovered using idwt or idwt2.:
% x = sin(0.3*[1:451]);
% wavelet_name = 'db9';
% [low_pass_decomposition,high_pass_decomposition,low_pass_reconstruction,high_pass_reconstruction] = wfilters(wavelet_name);
% signal_length = length(x);
% filter_length = length(low_pass_decomposition);
% extended_x = [randn(1,filter_length),x,randn(1,filter_length)];
% axis([1,signal_length+2*filter_length,-2,3]);
% subplot(2,1,1);
% plot(filter_length+1:filter_length+signal_length,x);
% title('original signal');
% axis([1,signal_length+2*filter_length,-2,3]);
% subplot(2,1,2);
% plot(extended_x);
% title('extended signal');
% axis([1,signal_length+2*filter_length,-2,3]);
% 
% %decomposition:
% full_convolved_signal_half_size = floor((signal_length+filter_length-1)/2);
% %  wkeep  Keep part of a vector or a matrix.
% %     For a vector:
% %     Y = wkeep(X,L,OPT) extracts the vector Y 
% %     from the vector X. The length of Y is L.
% %     If OPT = 'c' ('l' , 'r', respectively), Y is the central
% %     (left, right, respectively) part of X.
% %     Y = wkeep(X,L,FIRST) returns the vector X(FIRST:FIRST+L-1).
% %  
% %     Y = wkeep(X,L) is equivalent to Y = wkeep(X,L,'c').
% %  
% %     For a matrix:
% %     Y = wkeep(X,S) extracts the central part of the matrix X. 
% %     S is the size of Y.
% %     Y = wkeep(X,S,[FIRSTR,FIRSTC]) extracts the submatrix of 
% %     matrix X, of size S and starting from X(FIRSTR,FIRSTC).
% approximation1 = wkeep(dyaddown(conv(extended_x,low_pass_decomposition)),full_convolved_signal_half_size); %dyaddown??? wkeep???
% details1 = wkeep(dyaddown(conv(extended_x,high_pass_decomposition)),full_convolved_signal_half_size);
% %reconstruction:
% xr = idwt(approximation1,details1,wavelet_name,signal_length);
% err0 = max(abs(x-xr));


% %(25).
% load geometry;
% weird_pattern = X;
% %x contains the loaded image and map contains the loaded colormap:
% number_of_colors = size(map,1);
% colormap(pink(number_of_colors));
% %  wcodemat Extended pseudocolor matrix scaling.
% %     Y = wcodemat(X,NBCODES,OPT,ABSOL) returns a coded version
% %     of input matrix X if ABSOL=0, or ABS(X) if ABSOL is 
% %     nonzero, using the first NBCODES integers.
% %     Coding can be done row-wise (OPT='row' or 'r'), columnwise 
% %     (OPT='col' or 'c'), or globally (OPT='mat' or 'm'). 
% %     Coding uses a regular grid between the minimum and 
% %     the maximum values of each row (column or matrix,
% %     respectively).
% image(wcodemat(weird_pattern,number_of_colors)); %wcodemat????
% decomposition_level = 3;
% wavelet_name = 'sym4';
% %zero padding extention
% dwtmode('zpd'); 
% [coefficients_mat,index_matrix] = wavedec2(weird_pattern,decomposition_level,wavelet_name);
% % wrcoef2 Reconstruct single branch from 2-D wavelet coefficients.
% %     wrcoef2 reconstructs the coefficients of an image
% approximation_coefficients_level3 = wrcoef2('a',coefficients_mat,index_matrix,wavelet_name,decomposition_level); %??? summarize this using the 1D version
% image(wcodemat(approximation_coefficients_level3,number_of_colors));
% %symmetric extension:
% dwtmode('sym');
% [coefficients_mat,index_matrix] = wavedec2(weird_pattern,decomposition_level,wavelet_name);
% approximation_coefficients_level3 = wrcoef2('a',coefficients_mat,index_matrix,wavelet_name,decomposition_level);
% image(wcodemat(approximation_coefficients_level3,number_of_colors));
% %smooth padding extention:
% dwtmode('spd');
% [coefficients_mat,index_matrix] = wavedec2(weird_pattern,decomposition_level,wavelet_name);
% approximation_coefficients_level3 = wrcoef2('a',coefficients_mat,index_matrix,wavelet_name,decomposition_level);
% image(wcodemat(approximation_coefficients_level3,number_of_colors));
% 
% 
% %(26).
% %epsilon-decimated DWT / nondecimated discrete stationary wavelet transform (SWTs):
% e = 0;
% [A,D] = dwt(weird_pattern,wavelet_name,'mode','per','shift',e); %e={0/1}
% %1D SWT using the command line:
% load noisdopp;
% noisy_doppler_signal = noisdopp;
% [sw_approximation_coefficients,sw_detail_coefficients] = swt(noisy_doppler_signal,1,'db1');
% electricity_use_signal_reconstructed = iswt(sw_approximation_coefficients,sw_detail_coefficients,'db1');
% max_reconstruction_error = norm(noisy_doppler_signal-electricity_use_signal_reconstructed);
% %construct the level 1 approximation and detail (A1 and D1) from the coefficients swa and swd:
% null_coefficients = zeros(size(sw_approximation_coefficients));
% signal_approximation_using_approximation_coefficients = iswt(sw_approximation_coefficients,null_coefficients,'db1'); 
% signal_approximation_using_detail_coefficients = iswt(null_coefficients,sw_detail_coefficients,'db1'); %make order with the parameter order and meaning
% subplot(1,2,1);
% plot(signal_approximation_using_approximation_coefficients);
% title('approximation A1');
% subplot(1,2,2);
% plot(signal_approximation_using_detail_coefficients);
% title('detail D1');
% %perform a decomposition at level 3 of the signal using the db1 wavelet:
% [sw_approximation_coefficients,sw_detail_coefficients] = swt(noisy_doppler_signal,1,'db1');
% %display the coefficients of approximations and details:
% kp = 0; 
% for i=1:3
%    subplot(3,2,kp+1);
%    plot(sw_approximation_coefficients(i,:));
%    title(['approximation coefficients level', num2str(i)]);
%    subplot(3,2,kp+2);
%    plot(sw_detail_coefficients(i,:));
%    title(['detail coefficients level', num2str(i)]);
%    kp = kp + 2;
% end
% %reconstruct approximation at level 3 from coefficients:
% null_coefficients = zeros(size(sw_detail_coefficients));
% A = null_coefficients;
% %reconstruct the approximation at level 3:
% A(3,:) = iswt(sw_approximation_coefficients,null_coefficients,'db1'); %??? 
% %reconstruct details from coefficients
% D = null_coefficients;
% for i=1:3
%    swcfs = null_coefficients;
%    swcfs(i,:) = sw_detail_coefficients(i,:);
%    D(i,:) = iswt(null_coefficients,swcfs,'db1'); %since i enter zero for approximation coefficients i get the details only
% end
% %reconstruct and display approximations at level 1 and 2 from approximation
% %at level 3 and details at levels 2 and 3:
% %to reconstruct the approximations at levels 2 and 3:
% A(2,:) = A(3,:) + D(3,:);
% A(1,:) = A(2,:) + D(2,:);
% %to display the approximations and details at levels 1,2 and 3:
% kp = 0;
% for i=1:3
%    subplot(3,2,kp+1);
%    plot(A(i,:));
%    title(['approximation level',num2str(i)]);
%    subplot(3,2,kp+2);
%    plot(D(i,:));
%    title(['detail level',num2str(i)]);
%    kp = kp + 2;
% end
% %to denoise the signal, use the ddencmp command to calculate a default
% %global threshold. use the wthresh command to perform the actual
% %thresholding of the detail coefficients, and then use the iswt command to
% %obtain the denoised signal:
% [threshold,soft_or_hard_threshold] = ddencmp('den','wv',index_matrix); %what is sorh????? - soft or hard
% dswd = wthresh(sw_detail_coefficients,soft_or_hard_threshold,threshold);
% clean = iswt(sw_approximation_coefficients,dswd,'db1');
% subplot(2,1,1);
% plot(noisy_doppler_signal);
% title('original signal');
% subplot(2,1,2);
% plot(clean);
% title('denoised signal');
% %a second syntax can be used for the swt and iswt functions, giving the same results:
% decomposition_level = 5;
% swc = swt(index_matrix,decomposition_level,'db1');
% swcden = swc;
% swcden(1:end-1,:) = wthresh(swcden(1:end-1,:),soft_or_hard_threshold,threshold);


% %(26). WAVELET CHANGEPOINT DETECTION:
% load nileriverminima;
% % load river_dataset
% river_height_signal = nileriverminima;
% years = 622:1284;
% figure;
% plot(years,river_height_signal);
% title('nile river minimum levels');
% AX = gca;
% AX.XTick = 622:100:1222;
% grid on;
% xlabel('year');
% ylabel('cubits');
% %obtain a multiresolution analysis (MRA) of the data using the Haar wavelet:
% wt = modwt(river_height_signal,'haar',4); %????
% %  modwtmra Multiresolution analysis using the MODWT
% %     MRA = modwtmra(W) returns the multiresolution analysis (MRA) of the
% %     matrix W. W is a LEV+1-by-N matrix containing the MODWT of an N-point 
% %     input signal down to level LEV. By default, MODWMRA assumes that you 
% %     used the 'sym4' wavelet with periodic boundary handling to obtain
% %     the MODWT. MRA is a LEV+1-by-N matrix where LEV is the level of the
% %     MODWT and N is the length of the analyzed signal. The k-th row of MRA
% %     contains the details for the k-th level. The LEV+1-th row of MRA 
% %     contains the LEV-th level smooth.
% mra = modwtmra(wt,'haar'); %????
% figure;
% subplot(2,1,1);
% plot(years,mra(1,:));
% title('level 1 details');
% subplot(2,1,2);
% plot(years,mra(2,:));
% title('level 2 details');
% AX = gca;
% AX.XTick = 622:100:1222;
% xlabel('years');
% %apply an overall change of variance test to the wavelet coefficients:
% %HOW DOES IT WORK?!?!?!?!?
% for JJ=1:5
% %      wvarchg Find variance change points. 
% %     [PTS_OPT,KOPT,T_EST] = wvarchg(Y,K,D) computes the estimated 
% %     change points of the variance of signal Y for j change 
% %     points, with j = 0, 1, 2,..., K.
% %     Integer D is the minimum delay between two change points. 
% %  
% %     Integer KOPT is the proposed number of change points
% %     (0 <= KOPT <= K).
% %     The vector PTS_OPT contains the corresponding change points. 
% %     For 1 <= k <= K, T_EST(k+1,1:k) contains the k instants
% %     of the variance change points and then, 
% %     if KOPT > 0, PTS_OPT = T_EST(KOPT+1,1:KOPT) 
% %     else PTS_OPT = [].
% %  
% %     K and D must be integers such that 1 < K << length(Y) and
% %     1 <= D << length(Y).
% %     Signal Y should be zero mean.
% %     
% %     wvarchg(Y,K) is equivalent to wvarchg(Y,K,10).
% %     wvarchg(Y)   is equivalent to wvarchg(Y,6,10).
%    pts_opt = wvarchg(wt(JJ,:),2); %????????
%    changepoints{JJ} = pts_opt;
% end
% cellfun(@(x) ~isempty(x),changepoints,'uni',0) %????
% %determine the year corresponding to the detected change of variance:
% years(cell2mat(changepoints))
% %split the data into two segments. the first segment includes the years 622
% %to 721 when the fine scale wavelet coefficients indicate a change in
% %variance. the second segment cotains the years 722 to 1284. obtain
% %unbiased estimates of the wavelet variance by scale:
% tspre = river_height_signal(1:100);
% tspost = river_height_signal(101:end);
% wpre = modwt(tspre,'haar',4); %modwt = maximal overlap discrete wavelet transform!!!!!
% wpost = modwt(tspost,'haar',4);
% % modwtvar Maximal overlap discrete wavelet transform multiscale variance.
% %     WVAR = modwtvar(W) returns unbiased estimates of the wavelet variance
% %     by scale for the maximal overlap discrete wavelet transform (MODWT) in
% %     the LEV+1-by-N matrix W where LEV is the level of the input MODWT. For
% %     unbiased estimates, modwtvar returns variance estimates only where
% %     there are nonboundary coefficients. This condition is satisfied when
% %     the transform level is not greater than floor(log2(N/(L-1)+1)) where N
% %     is the length of the input. If there are sufficient nonboundary
% %     coefficients at the final level, modwtvar returns the scaling
% %     variance in the final element of WVAR. By default, modwtvar uses the
% %     'sym4' wavelet to determine the boundary coefficients.
% %         WVAR = modwtvar(...,'table') outputs a MATLAB table with the following
% %     variables:
% %         NJ          The number of MODWT coefficients by level. For unbiased
% %                     estimates, NJ represents the number of nonboundary
% %                     coefficients. For biased estimates, NJ is the number of
% %                     coefficients in the MODWT.
% %         Lower       The lower confidence bound for the variance estimate.
% %         Variance    The variance estimate by level. 
% %         Upper       The upper confidence bound for the variance estimate.
% wvarpre = modwtvar(wpre,'haar',0.95,'table'); %????
% wvarpost = modwtvar(wpost,'haar',0.95,'table') %understand all mod-something functions!!!?
% %compare the results:
% vpre = table2array(wvarpre);
% vpost = table2array(wvarpost);
% vpre = vpre(1:end-1,2:end);
% vpost = vpost(1:end-1,2:end);
% vpre(:,1) = vpre(:,2)-vpre(:,1); %vpre(:,2) used as mean and change 1 and 3 to the difference from them to 2-->
% vpre(:,3) = vpre(:,3)-vpre(:,2); %--> and enter those into the errorbar function
% vpost(:,1) = vpost(:,2)-vpost(:,1); %???
% vpost(:,3) = vpost(:,3)-vpost(:,2); %???
% figure;
% errorbar(1:4,vpre(:,2),vpre(:,1),vpre(:,3),'ko',markerfacecolor',[0,0,0]);
% hold on;
% errorbar(1.5:4.5,vpost(:,2),vpost(:,1),vpost(:,3),'b^','markerfacecolor',[0,0,1]);
% set(gca,'xtick',1.25:4.25);
% set(gca,'xticklabel',{'2 year','4 years','8 years','16 years','32 years'});
% grid on;
% ylabel('variance');
% title('wavelet variance 622-721 and 722-1284 by scale','fontsize',14);
% legend('years 622-721','years 722-1284','location','northeast');



% %(27). SCALE LOCALIZED VOLATILITY AND CORRELATION:
% load GDPcomponents;
% realgdpwt = modwt(realgdp,'db2',6);
% vardata = var(realgdp,1);
% varwt = var(realgdpwt,1,2);
% totalMODWTvar = sum(varwt);
% bar(varwt(1:end-1,:));
% AX = gca;
% AC.XTickLabels = {'[2 4)','[4 8)','[8 16)','[16 32)','[32 64)','[64 128)'};
% xlabel('quarters');
% ylabel('variance');
% title('wavelet variance by scale');
% %examining the aggregate data, it is not clear that there is in fact
% %reduced volatility in this period. use wavelets to investigate this by
% %first obtaining a multiresolution analysis of the real GDP data using the
% %'db2' wavelet down to level 6:
% realgdpwt = modwt(realgdp,'db2',6,'reflection');
% gdpmra = modwtmra(realgdpwt,'db2','reflection');
% %plot the level one details D1 - which capture oscillations in the data
% %between two and four quarters in duration:
% helperFinancialDataExample1(gdpmra(1,:),years,'year over year real us gdp - D1');
% %test the level one wavelet coefficients for significant variance changepoints:
% [pts_opt,kopt,t_est] = wvarchg(realgdpwt(1,1:numel(realgdp)),2);
% years(pts_opt)
% 
% 
% %(28). correlation between two datasets by scale:
% %examine the correlation between the aggregate data on government spending
% %and private investment. the data cover the same period as the real GDP
% %data and are transformed in the exact same way"
% [rho,pval] = corrcoef(privateinvest,govtexp);
% %now repeat this analysis using the MODWT:
% wtPI = modwt(privateinvest,'db2',5,'reflection');
% wtGE = modwt(govtexp,'db2',5,'reflection');
% wcorrtable = modwtcorr(wtPI,wtGE,'db2',0.95,'reflection','table');
% display(wcorrtable)
% %with financial data, there is often a leading or lagging relationship
% %between variables. in those cases, it is useful to examine the
% %cross-correlation sequence to determine if lagging one variable with
% %respect to another maximizes their cross correlation. 
% %to illustrate this, consider the correlation between two components of the
% %GDP personal consumption expenditures and gross private domenstic investment
% piwt = modwt(privateinvest,'fk8',5);
% pcwt = modwt(pc,'fk8',5);
% figure;
% modwtcorr(piwt,pcwt,'fk8');
% %examine the wavelet cross correlation sequence at the scale representing 2-4 quarter cycles:
% [xcseq,xcseqci,lags] = modwtxcorr(piwt,pcwt,'fk8'); %cross correlation between coefficients
% zerolag = floor(numel(xcseq{1})/2)+1;
% plot(lags{1}(zerolag:zerolag+20),xcseq{1}(zerolag:zerolag+20));
% hold on;
% plot(lags{1}(zerolag:zerolag+20),xcseqci{1}(zerolag:zerolag+20,:),'r--');
% xlabel('lag (quarters)');
% grid on;
% title('wavelet cross correlation sequence -- [2Q,4Q)');



% %(29). R wave detection in the ECG
% load mit200;
% figure;
% plot(tm,ecgsig);
% hold on;
% plot(tm(ann),ecgsig(ann),'ro');
% xlabel('seconds');
% ylabel('amplitude');
% title('subject MIT-BIH 200');
% %the 'sym4' wavelet resembles the QRS compex, which makes it a good choice
% %for QRS detection. to illustrate this more clearly, extract a QRS complex
% %and plot the result with a dilated and translated 'sym4' wavleet for comparison:
% qrsEx = ecgsig(4560:4810);
% [mpdict,~,~,longs] = wmpdictionary(numel(qrsEx),'lstcpt',{'sym4',3}); %????
% figure;
% plot(qrsEx);
% hold on;
% plot(2*circshift(mpdict(:,11),[-2,0]),'r');
% axis tight;
% legend('qrs complex','sym4 wavelet');
% title('comparison of sym4 wavelet and QRS complex');
% %first, decompose the ECG waveform down to level 5 using the default 'sym4'
% %wavelet. then, reconstruct a frequency localized version of the ECG
% %waveform using only the wavelet coefficients at scales 4 and 5.
% wt = modwt(ecgsig,5);
% wtrec = zeros(size(wt));
% wtrec(4:5,:) = wt(4:5,:);
% y = imodwt(wtrec,'sym4');
% %use the squared absolute values of the signal approximation built from the
% %wavelet coefficients and employ a peak finding algorithm to identify the R peaks:
% y = abs(y).^2;
% [qrspeaks,locs] = findpeaks(y,tm,'minpeakheight',0.35,'minpeakdistance',0.15);
% figure;
% plot(tm,y);
% hold on;
% plot(locs,qrspeaks,'ro');
% xlabel('seconds');
% title('R peaks localized by wavelet transform with automatic annotations');



% %(30). 1D multisignal wavelet analysis:
% % mdtdec = multisignal wavelet decomposition
% % mdwtrec = multisignal wavelet reconstruction and extraction of approximation and detail coefficients
% % chgwdeccfs = change multisignal 1D decomposition coefficients
% % wdecenrgy = multisignal 1D decomposition energy repartition
% % mswcmp = multisignal 1D compression using wavelets
% % mswcmpscr = multisignal 1D wavelet compression scores
% % mswcmptp = multisignal 1D  compression thresholds and performance
% % mswden = multisignal 1D denoising using wavelets
% % mswthresh = perform multisignal 1D thresholding
% load thinker;
% multiple_signals = X; %each signal along columns (192 signals of length 96)
% plot(multiple_signals(1:5,:)','r');
% hold on;
% plot(multiple_signals(21:25,:)','b');
% plot(multiple_signals(31:35,:)','g');
% set(gca,'Xlim',[1,96]);
% grid;
% %perform wavelet decomposition of signals at level 2 of row signals using the db2 wavelet:
% dec = mdwtdec('r',multiple_signals,2,'db2');
% %change wavelet coefficients:
% decBIS = chgwdeccfs(dec,'cd',0,1); %this generates a new decomposition structure decBIS
% %perform a wavelet reconstruction of signals and plot some of the new signals:
% xbis = mdwtrec(decBIS);
% figure;
% plot(xbis(1:5,:)','r');
% hold on;
% plot(xbis(21:25,:)','b');
% plot(xbis(31:35,:)','g');
% grid;
% set(gca,'xlim',[1,96]);
% %compare old and new signals by plotting them together:
% figure;
% idxSIG = [1,31];
% plot(multiple_signals(idxSIG,:)','r','linewidth',2);
% hold on;
% plot(xbis(idxSIG,:)','b','linewidth',2);
% grid;
% set(gca,'xlim',[1,96]);
% %set the wavelet coefficients at level 1 and 2 for signals 31 to 35 to the
% %value zero, perform a wavleet reconstruction of signal 31, and compare
% %some of the old and new signals:
% decTER = chgwdeccfs(dec,'cd',0,1:2,31:35);
% Y = mdwtrec(decTER,'a',0,31);
% figure;
% plot(X([1,31],:)','r','linewidth',2);
% hold on;
% plot([xbis(1,:);Y]','b',linewidth',2);
% grid;
% set(gca,'xlim',[1,96]);
% %compute the energy of signals and the percentage of energy for wavelet components:
% [E,PEC,PECFS] = wdecenergy(dec);
% %energy of signals 1 and 31:
% ener_1_31 = E([1,31]);
% %compute the percentage of energy for wavelet components of signals 1 and 31:
% pec_1_31 = PEC([1,31],:);
% %compress the signals to obtain a percentage of zeros near 95% for the wavelet coefficients:
% [XC,decCMP,THRESH] = mswcmp('cmp',dec,'N0_perf',05);
% [Ecmp,PECcmp,PECFScmp] = wdecenergy(decCMP);
% %plot the original signals 1 and 31, and the corresponding compressed
% %signals:
% figure;
% plot(multiple_signals([1,31],:)','r','linewidth',2);
% hold on;
% plot(XC([1,31],:),'b','linewidth',2);
% grid;
% set(gca,'xlim',[1,96]);
% %compute thresholds, percentage of energy preserved and percentage of zeros
% %associated with the L2_perf method preserving at least 95% of energy:
% [thr_val,l2_perf,n0_perf] = mswcmptp(dec,'L2_perf',95);
% idxSIG = [1,31];
% threshold = thr_val(idxSIG);
% l2per = l2_perf(idxSIG);
% n0per = n0_perf(idxSIG);
% %compress the signals to obtain a percentage of zeros near 60% for the
% %wavelet:
% [XC,decCMP,THRESH] = mswcmp('cmp',dec,'N0_perf',60); 
% %now XC signals are the compressed versions of the original signals in the row direction
% %now compress the XC signals in the column direction:
% XX = mswcmp('cmpsig','c',XC,'db2',2,'N0_perf',60);
% %plot original signals  and the compressed signals xx as images:
% figure;
% subplot(1,2,1);
% image(X);
% subplot(1,2,2);
% image(XX);
% colormap(pink(222));
% %Denoise the signals using the universal threshold:
% %XD signals are the denoised versions of the original signals in the row direction
% [XD,decDEN,THRESH] = mswden('den',dec,'sqtwolog','sln');
% figure;
% plot(multiple_signals([1,31],:)','r','linewidth',2);
% hold on;
% plot(XD([1,31],:)','b','linewidth',2);
% grid;
% set(gca,'xlim',[1,96]);
% %now denoise the XD signals in the column direction:
% XX = mswden('densig','c',XD,'db2',2,'sqtwolog','sln');
% %plot original signals X and the denoised signals XX as images:
% figure;
% subplot(1,2,1);
% image(X);
% subplot(1,2,2);
% image(XX);
% colormap(pink(222));



%(31). 2D discrete wavelet analysis:
load wbarb;
woman_image = X;
imagesc(woman_image);
colormap(map);
colorbar;
[approximation_coefficients_level1,cH1,cV1,details_coefficients_level1] = dwt2(woman_image,'bior3.7');
%construct and display approximations and details from the coefficients:
signal_approximation_using_approximation_coefficients = upcoef2('a',approximation_coefficients_level1,'bior3.7',1);
H1 = upcoef2('h',cH1,'bior3.7',1);
V1 = upcoef2('v',cV1,'bior3.7',1);
signal_approximation_using_detail_coefficients = upcoef2('d',details_coefficients_level1,'bior3.7',1);
%or:
sx = size(woman_image);
signal_approximation_using_approximation_coefficients = idwt2(approximation_coefficients_level1,[],[],[],'bior3.7',sx);
H1 = idwt2([],cH1,[],[],'bior3.7',sx);
V1 = idwt2([],[],cV1,[],'bior3.7',sx);
signal_approximation_using_detail_coefficients = idwt2([],[],[],details_coefficients_level1,'bior3.7',sx);
%to display the results of the level 1 decomposition:
colormap(map);
subplot(2,2,1);
image(wcodemat(signal_approximation_using_approximation_coefficients,192)); %wcodemat??!?!?!
title('approximation A1');
subplot(2,2,2);
image(wcodemat(H1,192));
title('horizontal detail H1');
subplot(2,2,3);
image(wcodemat(V1,192));
title('vertical detail V1');
subplot(2,2,4);
image(wcodemat(signal_approximation_using_detail_coefficients,192));
title('diagonal detail D1');
%to find the inverse transform:
Xsyn = idwt2(approximation_coefficients_level1,cH1,cV1,details_coefficients_level1,'bior3.7');
%perform a multilevel wavelet decomposition:
%to perform a level 2 decomposition of the image:
[coefficients_mat,index_matrix] = wavedec2(woman_image,2,'bior3.7');
%extract the level 2 approximation coefficients from C:
cA2 = appcoef2(coefficients_mat,index_matrix,'bior3.7',2);
%extract the first and second level detail coefficients from C:
cH2 = detcoef2('h',coefficients_mat,index_matrix,2);
cV2 = detcoef2('v',coefficients_mat,index_matrix,2);
details_coefficients_level2 = detcoef2('d',coefficients_mat,index_matrix,2);
cH1 = detcoef2('h',coefficients_mat,index_matrix,1);
cV1 = detcoef2('v',coefficients_mat,index_matrix,1);
details_coefficients_level1 = detcoef2('d',coefficients_mat,index_matrix,1);
%or
[cH2,cV2,details_coefficients_level2] = detcoef2('all',coefficients_mat,index_matrix,2);
[cH1,cV1,details_coefficients_level1] = detcoef2('all',coefficients_mat,index_matrix,1);
%reconstruct the level 2 approximation and the level 1 and 2 details:
A2 = wrcoef2('a',coefficients_mat,index_matrix,'bior3.7',2); %reconstruct the level 2 approximation from C
%to reconstruct the level 1 and 2 detilas from C:
H1 = wrcoef2('h',coefficients_mat,index_matrix,'bior3.7',1);
V1 = wrcoef2('v',coefficients_mat,index_matrix,'bior3.7',1);
signal_approximation_using_detail_coefficients = wrcoef2('d',coefficients_mat,index_matrix,'bior3.7',1);
H2 = wrcoef2('h',coefficients_mat,index_matrix,'bior3.7',2);
V2 = wrcoef2('v',coefficients_mat,index_matrix,'bior3.7',2);
D2 = wrcoef2('d',coefficients_mat,index_matrix,'bior3.7',2);
%display the results of the level 2 decomposition:
colormap(map);
subplot(2,4,1);
image(wcodemat(signal_approximation_using_approximation_coefficients,192)); %why wcodemat!?!?!?!!??
title('approximation A1');
subplot(2,4,2);
image(wcodemat(H1,192));
title('horizontal detail H1');
subplot(2,4,3);
image(wcodemat(V1,192));
title('vertical detail V1');
subplot(2,4,4);
image(wcodemat(signal_approximation_using_detail_coefficients,192));
title('diagonal detail D1');
subplot(2,4,5);
image(wcodemat(A2,192));
title('approximation A2');
subplot(2,4,6);
image(wcodemat(H2,192));
title('horizontal detail H2');
subplot(2,4,7);
image(wcodemat(V2,192));
title('vertical detail V2');
subplot(2,4,8);
image(wcodemat(D2,192));
title('diagonal detail D2');
%reconstruct the original image from the multilevel decomposition:
X0 = waverec2(coefficients_mat,index_matrix,'bior3.7');
%compress the image and display it:
%to compress the original image X, use the ddencmp command to calculate the
%default parameters and the wdencmp command to perform the actual compression:
[threshold,soft_or_hard_threshold,keep_approximation] = ddencmp('cmp','wv',woman_image);
[Xcomp,CXC,LXC,PERF0,PERFL2] = wdencmp('gbl',coefficients_mat,index_matrix,'bior3.7',2,threshold,soft_or_hard_threshold,keep_approximation);
colormap(map);
subplot(1,2,1);
imagesc(woman_image);
title('original image');
axis square;
subplot(1,2,2);
image(Xcomp);
title('compressed image');
axis square;


%(32). 2D discrete stationary wavelet analysis:
% load noisewom;
load wbarb;
woman_image = X; %maybe add noise
%perform single level decomposition of the image using the db1 wavelet:
[sw_approximation_coefficients,swh,swv,sw_detail_coefficients] = swt2(woman_image,1,'db1');
map = pink(size(map,1));
colormap(map);
subplot(2,2,1);
image(wcodemat(sw_approximation_coefficients,192)); %what is the 192?!!?!
title('approximation swa'); 
subplot(2,2,2);
image(wcodemat(swh,192));
title('horizontal detail swh');
subplot(2,2,3);
image(wcodemat(swv,192));
title('vertical detail swv');
subplot(2,2,4);
image(wcodemat(sw_detail_coefficients,192));
title('diagonal detail swd');
%find the inverse transform:
electricity_use_signal_reconstructed = iswt2(sw_approximation_coefficients,swh,swv,sw_detail_coefficients,'db1');
max_reconstruction_error = max(max(abs(woman_image-electricity_use_signal_reconstructed)));
%to construct the level 1 approximation and details (A1,H1,V1 and D1) from
%the coefficients swa,swh,swv,swd:
null_coefficients = zeros(size(sw_approximation_coefficients));
signal_approximation_using_approximation_coefficients = iswt2(sw_approximation_coefficients,null_coefficients,null_coefficients,null_coefficients,'db1');
H1 = iswt2(null_coefficients,swh,null_coefficients,null_coefficients,'db1');
V1 = iswt2(null_coefficients,null_coefficients,swv,null_coefficients,'db1');
signal_approximation_using_detail_coefficients = iswt2(null_coefficients,null_coefficients,null_coefficients,sw_detail_coefficients,'db1');
%to display the approximation and details at level 1:
colormap(map);
subplot(2,2,1);
image(wcodemat(signal_approximation_using_approximation_coefficients,192));
title('approximation A1');
subplot(2,2,2);
image(wcodemat(H1,192));
title('horizontal detail H1');
subplot(2,2,3);
image(wcodemat(V1,192));
title('vertical detail V1');
subplot(2,2,4);
image(wcodemat(signal_approximation_using_detail_coefficients,192));
title('diagonal detail D1');
%remove noise by thresholding:
threshold = 44.5;
soft_or_hard_threshold = 's';
dswh = wthresh(swh,soft_or_hard_threshold,threshold);
dswv = wthresh(swv,soft_or_hard_threshold,threshold);
dswd = wthresh(sw_detail_coefficients,soft_or_hard_threshold,threshold);
clean = iswt2(sw_approximation_coefficients,dswh,dswv,dswd,'db1');
colormap(map);
subplot(1,2,1);
image(wcodemat(woman_image,192));
title('original image');
subplot(1,2,2);
image(wcodemat(clean,192));
title('denoised image');
%a second syntax can be used for the swt2 and iswt2 function, giving the same results:
decomposition_level = 4;
swc = swt2(woman_image,decomposition_level,'db1');
swcden = swc;
swcden(:,:,1:end-1) = wthresh(swcden(:,:,1:end-1),soft_or_hard_threshold,threshold);
clean = iswt2(swcden,'db1');



%(33).Dual Tree wavelet transforms:
%the DWT suffers from shift variance, meaning that small shifts in the
%input signal or image can cause significant changes in the distribution of
%the signal/image energy across scales in the DWT coefficients. the complex
%dual tree DWT is approximately shift invariant.
kronDelta1 = zeros(128,1);
kronDelta1(60) = 1;
kronDelta2 = zeros(128,1);
kronDelta2(64) = 1;
J = 3;
dwt1 = dddtree('dwt',kronDelta1,J,'sym7'); %if i use 'dwt' i think it just the regular dwt, CHECK THIS!!!
dwt2 = dddtree('dwt',kronDelta2,J,'sym7'); %double density dual tree
dwt1cfs = dwt1.cfs{J};
dwt2cfs = dwt2.cfs{J};
dt1 = dddtree('cplxdt',kronDelta1,J,'dtf3'); 
dt2 = dddtree('cplxdt',kronDelta2,J,'dtf3'); %complex discrete transform
dt1cfs = dt1.cfs{J}(:,:,1) + 1i*dt1.cfs{J}(:,:,2);
dt2cfs = dt2.cfs{J}(:,:,1) + 1i*dt2.cfs{J}(:,:,2);
figure;
subplot(1,2,1);
plot(abs(dwt1cfs),'markerfacecolor',[0,0,1]);
title(['dwt squared 2-norm = ',num2str(norm(dwt1cfs,2)^2)]);
subplot(1,2,2);
plot(abs(dwt2cfs),'markerfacecolor',[0,0,1]);
title(['dwt squared 2-norm = ',num2str(norm(dwt2cfs,2)^2)]);
figure;
subplot(1,2,1);
plot(abs(dt1cfs),'markerfacecolor',[0,0,1]);
title(['dual tree dwt squared 2 norm = ',num2str(norm(dt1cfs,2)^2)]);
subplot(1,2,2);
plot(abs(dt2cfs),'markerfacecolor',[0,0,1]);
title(['dual tree dwt squared 2 norm = ',num2str(norm(dt2cfs,2)^2)]);


load wecg;
ecg_signal = wecg;
dt = 1/180;
t_vec = 0:dt:(length(ecg_signal)*dt)-dt;
figure;
plot(t_vec,ecg_signal);
xlabel('seconds');
ylabel('millivolts');
J = 6;
dtDWT1 = dddtree('dwt',ecg_signal,J,'farras');
details = zeros(2048,3); 
details(2:4:end,2) = dtDWT1.cfs{2}; %what above level 1????
details(4:8:end,3) = dtDWT1.cfs{3};
subplot(3,1,1);
plot(t_vec,details(:,2),'marker','none','showbaseline','off');
title('level 2');
ylabel('mV');
subplot(3,1,2);
plot(t_vec,details(:,3),'marker','none','showbaseline','off');
title('level 3');
ylabel('mV');
subplot(3,1,3);
plot(t_vec,ecg_signal);
title('original signal');
xlabel('seconds');
ylabel('mV');
%repeat the above analysis for the dual tree transform. in this case, just
%plot the real part of the dual tree coefficients at levels 2 and 3:
dtcplx1 = dddtree('cplxdt',ecg_signal,J,'dtf3');
details = zeros(2048,3);
details(2:4:end,2) = dtcplx1.cfs{2}(:,1,1) + 1i*dtcplx1.cfs{2}(:,1,2);
details(4:8:end,3) = dtcplx1.cfs{3}(:,1,1) + 1i*dtcplx1.cfs{3}(:,1,2);
subplot(3,1,1);
plot(t_vec,details(:,2),'marker','none','showbaseline','off');
title('level 2');
ylabel('mV');
subplot(3,1,2);
plot(t_vec,details(:,3),'marker','none','showbaseline','off');
title('level 3');
ylabel('mV');
subplot(3,1,3);
plot(t_vec,wecg);
title('original signal');
xlabel('seconds');
ylabel('mV');


%(34).
%both the critically sampled and dual tree wavelet transforms localize an
%important feature of the ECG waveform to similar scales. an important
%application of wavelets in 1D signals is to obtain an analysis of
%variance by scale. it stands to reason that this analysis of variance
%should not be sensitive to circular shifts in the input signal.
%unfortunately, this is not the case with the critically sampled DWT. to
%demonstrate this, we circularly shift the ECG signal by 4 samples, analyze
%the unshifted and shifted signals with the critically sampled DWT, and
%calculate the distribution of energy across scales:
wecgShift = circshift(ecg_signal,4);
dtDWT2 = dddtree('dwt',wecgShift,J,'farrs');
sigenergy = norm(ecg_signal,2)^2;
enr1 = cell2mat(cellfun(@(x)(norm(x,2)^2/sigenergy)*100,dtDWT1.cfs,'uni',0)); %weird way to use cellfun and cell2mat
enr2 = cell2mat(cellfun(@(x)(norm(x,2)^2/sigenergy)*100,dtDWT2.cfs,'uni',0));
levels = {'D1';'D2';'D3';'D4';'D5';'D6';'A6'};
enr1 = enr1{:};
enr2 = enr2{:};
table(levels,enr1,enr2,'variablenames',{'levels','enr1','enr2'});
%now use the complex dual tree wavelet transform:
dtcplx1 = dddtree('cplxdt',ecg_signal,J,'dtf3');
dtcplx2 = dddtree('cplxdt',wecgShift,J,'dtf3');
cfs1 = cellfun(@squeeze,dtcplx1.cfs,'uni',0); %what is the 'uni' ????
cfs2 = cellfun(@squeeze,dtcplx2.cfs,'uni',0);
cfs1 = cellfun(@(x) complex(x(:,1),x(:,2)),cfs1,'uni',0);
cfs2 = cellfun(@(x) complex(x(:,1),x(:,2)),cfs2,'uni',0);
dtenr1 = cell2mat(cellfun(@(x)(norm(x,2)^2/sigenergy)*100,cfs1,'uni',0));
dtenr2 = cell2mat(cellfun(@(x)(norm(x,2)^2/sigenergy)*100,cfs2,'uni',0));
dtenr1 = dtenr1(:);
dtenr2 = dtenr2(:);
table(levels,dtenr1,dtenr2,'variablenames',{'level','dtenr1','dtenr2'});



%(35). DIRECTIONAL SELECTIVITY IN IMAGE PROCESSING:
%the standard implementation of the DWT uses separable filtering of the
%columns and rows of the image. the LH,HL,and HH wavelets for Daubechies'
%least-asymmetric phase wavelet with 4 vanishing moments (sym4) are shown
%in the following figure:
figure;
J = 5;
levels_vec = 3*2^(J+1); %number of possible paths (3 at the beginning then two for each branch)????
N = levels_vec/2^J;
Y = zeros(levels_vec,3*levels_vec); %3 orientations
dt = dddtree2('dwt',Y,J,'sym4');
dt.cfs{J}(N/2,N/2,1) = 1; 
dt.cfs{J}(N/2,N/2+N,2) = 1; %third argument is the different {LH,HL,HH}
dt.cfs{J}(N/2,N/2+2*N,3) = 1;
dwtImage = idddtree2(dt);
imagesc(dwtImage);
axis xy;
axis off;
title({'critically sampled DWT';'2D separable wavelets (sym4)-LH,HL,HH'});
%the dual tree DWT achieves directional selectivity by using wavelets that
%are approximately analytic, meaning that they have support on only half of
%the frequency axis. in the dual tree DWT, there are six subbands for both
%the real and imaginary parts. the six real parts are formed by adding the
%outputs of column filtering followed by row filtering of the input image
%in the two trees. the six imaginary parts are formed by subtracting the
%outputs of column filtering followed by row filtering(?????):

%the filters applied to the columns and rows may be from the same filter
%pair, {h0,h1} or {g0,g1}, or from different filter pairs, {h0,g1},{g0,h1}.
%the following code shows the orientation of the 12 wavelets corresponding
%to the real and imaginary parts of the complex oriented dual tree DWT:
J = 4;
levels_vec = 3*2^(J+1);  
N = levels_vec/2^J;
Y = zeros(2*levels_vec,6*levels_vec); %6 orientations and two parts of the signal (real and imaginary)
wt = dddtree2('cplxdt',Y,J,'dtf3');
wt.cfs{J}(N/2,N/2+0*N,2,2,1) = 1;
wt.cfs{J}(N/2,N/2+1*N,3,1,1) = 1;
wt.cfs{J}(N/2,N/2+2*N,1,2,1) = 1;
wt.cfs{J}(N/2,N/2+3*N,1,1,1) = 1;
wt.cfs{J}(N/2,N/2+4*N,3,2,1) = 1;
wt.cfs{J}(N/2,N/2+5*N,2,1,1) = 1;

wt.cfs{J}(N/2+N,N/2+0*N,2,2,2) = 1;
wt.cfs{J}(N/2+N,N/2+1*N,3,1,2) = 1;
wt.cfs{J}(N/2+N,N/2+2*N,1,2,2) = 1;
wt.cfs{J}(N/2+N,N/2+3*N,1,1,2) = 1;
wt.cfs{J}(N/2+N,N/2+4*N,3,2,2) = 1;
wt.cfs{J}(N/2+N,N/2+5*N,2,1,2) = 1;

waveIm = idddtree2(wt);
imagesc(waveIm);
axis off;
title('complex oriented dual tree 2D wavelets');
%you can use the dddtree2 with the 'realdt' option to obtain the real
%oriented dual tree DWT, which uses only the real parts. using the real
%oriented dual tree wavelet transform you can achieve directional
%selectivity, but you do not gain the full benefit of using analytic
%wavelets usch as approximate shift invariance.


%(36). IMAGE DENOISING:
%the dual tree DWT isolates distinct orientations is separate subbands and
%outperforms the standard separable DWT in applications like image denoising.



%(37). EDGE REPRESENTATION IN TWO DIMENSIONS:
%the approximation analyticity and selective orientation of the complex
%dual tree wavelets provide superior performance over the standard 2D DWT
%in the representation of edges in images. the illustrate this, we analyze
%test images with edges consisting of line and curve singularities in
%multiple directions using the critically sampled 2D DWT and the 2D complex
%oriented dual tree transform. first analyze an image of an octagon, which
%consists of line singularities:
load woctagon;
% hexagon = imread('hexagon.jpg');
hexagon = woctagon;
figure;
imagesc(hexagon);
colormap gray;
title('original image');
axis equal;
axis off;
%decompose the image down to level 4 and reconstruct an image approximation
%based on the level 4 dtail coefficients:
dtcplx = dddtree2('cplxdt',hexagon,4,'dtf3');
dtDWT = dddtree2('dwt',hexagon,4,'farras');
dtcplx.cfs{1} = zeros(size(dtcplx.cfs{1}));
dtcplx.cfs{2} = zeros(size(dtcplx.cfs{2}));
dtcplx.cfs{3} = zeros(size(dtcplx.cfs{3}));
dtcplx.cfs{5} = zeros(size(dtcplx.cfs{5})); %where is the 4-th element?!?!?!?
dtDWT.cfs{1} = zeros(size(dtDWT.cfs{1}));
dtDWT.cfs{2} = zeros(size(dtDWT.cfs{2}));
dtDWT.cfs{3} = zeros(size(dtDWT.cfs{3}));
dtDWT.cfs{5} = zeros(size(dtDWT.cfs{5}));
dtImage = idddtree2(dtcplx);
dwtImage = idddtree2(dtDWT);
subplot(1,2,1);
imagesc(dtImage);
axis equal;
axis of;
colormap gray;
title('complex oriented dual tree');
subplot(1,2,2);
imagesc(dwtImage);
axis equal;
axis off;
colormap gray;
title('DWT');
%we have shown that the dual tree DWT possess the desirable properties of
%near shift invariance and directional seleectivity not achievable with the
%critically sampled DWT. we have demonstrated how these properties can
%result in improved prerformance in signal analysis, the representation of
%singularities in image, and image denoising. in the addition to the real
%oriented and complex oriented dual tree DWT, dddtree and dddtree2 also
%support the double density wavelet transform and dual tree double density
%wasvelet transforms, which are additional examples of overcomplete wavelet
%filter banks (frames) with advantage over the standard DWT.



%(38). ANALYTIC WAVELETS USING THE DUAL TREE WAVELET TRANSFORM
%this example shows how to create approximately analytic wavelets using the
%dual tree complex wavelet transform. the example demonstrates that you
%cannot arbitrarily choose the analysis (decomposition) and snthesis
%(reconstruction) filters to obtain an approximately analytic wavelet. the
%FIR filters in the two filter banks must be carefully constructed in order
%to obtain an approximately analytic wavelet transform and derive the
%benefits of the dual tree transform.
%obtain the lowpass and highpass analysis filters:
DF = dtfilters('dtf1');
%DF is a 1X2 cell array of NX2 matrices containing the first stage lowpass
%and highpass filters, DF{1}, and the lowpass and highpass filters for
%subsequent stages, DF{2}.

%create the zero signal 256 samples in length. obtain two dual tree
%transforms of the zero signal dwon to level 5.
x = zeros(256,1);
wt1 = dddtree('cplxdt',x,5,DF{1},DF{2});
wt2 = dddtree('cplxdt',x,5,DF{1},DF{2});
%set a single level 5 detail coefficient in each of the two trees to 1 and
%invert the transform to obtain the wavelets:
wt1.cfs{5}(5,1,1) = 1; %what's the difference between the {5} and the (5,....) = 1???
wt2.cfs{5}(5,1,2) = 1;
wav1 = idddtree(wt1);
wav2 = idddtree(wt2);
%form the compplex wavelet using the first tree as the real part and the
%second tree as the imaginary part. plot the real and imaginary parts of the wavelet:
analwav = wav1 + 1i*wav2;
plot(real(analwav));
hold on;
plot(imag(analwav),'r');
plot(abs(analwav),'k','linewidth',2);
axis tight;
legend('real part','imaginary part','magnitude','location','northwest');
%fourier transform the analytic wavelet and plot the magnitude:
zdft = fft(analwav);
domega = (2*pi)/length(analwav);
omega = 0:domega:(2*pi)-domega;
clf;
plot(omega,abs(zdft));
xlabel('radians/sample');
%the fourier transform of the wavelet has support on essentialy only half
%of the frequency axis (not completely).
%now repeat the preceding procedure with two arbitrarily chosen orthogonal
%wavelets,'db4' and 'sym4':
[LoD1,HiD1] = wfilters('db4');
[LoD2,HiD2] = wfilters('sym4');
decomposition_filters = {[LoD1',HiD1'] , [LoD2',HiD2']};
wt1 = dddtree('cplxdt',x,5,decomposition_filters,decomposition_filters);
wt2 = dddtree('cplxdt',x,5,decomposition_filters,decomposition_filters); 
wt1.cfs{5}(5,1,1) = 1;
wt2.cfs{5}(5,1,2) = 1;
wav1 = idddtree(wt1);
wav2 = idddtree(wt2);
analwav = wav1 + 1i*wav2;
zdft = fft(analwav);
domega = (2*pi)/length(analwav);
omega = 0:domega:(2*pi)-domega;
clf;
plot(omega,abs(zdft));



%(39). WAVLEET PACKETS:
%the wavelet packet method is a generalization of wavelet decomposiiton
%that offers a richer signal analysis. wavelet packet atoms are waveforms
%indexed by three naturally interpreted parameters: position, scale (as in
%wavelet decomposition), and frequency.
%for a given orthogonal wavelet function, we generate a library of bases
%called wavelet packet bases. each of these bases offers a particular way
%of coding signals, preserving global energy and reconstructing exact
%features. we then select the most suitable decomposition of a given signal
%with respect to an entropy based criterion. 
%there exist simple and efficient algorithms for both wavelet packet
%decomposition and optimal decomposition selection. we can then produce
%adaptive filtering algorithms with direct applications in optimal coding
%and data compression.

%in the orthogonal wavelet decomposition procedure, the generic step splits
%the approximation coefficients into two parts. after splitting we obtain a
%vector of approximation coefficients and a vector of detail coefficients,
%both at a coarser scale. the information lost between two successive
%approximations is captured in the detail coefficients. then the next step
%consists of splitting the new APPROXIMATION coefficient vector, whilse
%successive details are never reanalyzed.
%in the corresponding wavelet packet situation, each detail coefficient
%vector is also decomposed into two parts using the same approach as in
%approimation vector splitting. this offers the richest analysis: the
%complete binary tree is produced as shown in the following figure.

% wpcoef = wavelet packet coefficients
% wpdec and wpdec2 = full decomposition
% wpsplt = decompose packet
% wprceof = reconstruct coefficients
% wprec and wprec2 = full reconstruction
% wpjoin = recompose packet
% bestrree = find best tree
% bestlevt = find best leve tree
% entrupd = update wavelet packets entropy
% get = get WPTREE object fields contents
% read = read values in WPTREE object fields
% wenergy = entropy
% wp2wtree = extract wavelet tree from wavelet packet tree
% wpcutree = cut wavelet packet tree
% ddencmp = default values for denoising and compression
% wpbmpen = penalized threshold for wavelet packet denoising
% wpdencmp = denoising and compression using wavelet packets
% wpthcoef = wavelet packets coefficients thresholding 
% wthrmngr = threshold settings manager

%because wavelets are localized in time and frequency, it is possible to
%use wavelet based counterparts to the STFT for the time frequency analysis
%of nonstationary signals. for example, it is possible to contruct the
%scalogram (wscalogram) based on the continuous wavelet transform (CWT).
%however, a potential drawback of using the CWT is that it is
%computationally expensive. 
%the discrete wavelet transform (DWT) permits a time frequency
%decomposition of the input sgnal, but the degree of frequency resolution
%in the DWT is typically considered too coarse for practical time frequency
%analysis.
%as a compromise between the DWT and CWT based techniques, wavelet packets
%provide a computationally efficient alternative with sufficient frequency
%resolution. you can use wpspectrum to perform a time frequency analysis of
%your signal using wavelet packets.
fs = 1000; 
t_vec = 0:1/fs:2;
y = sin(256*pi*t_vec);
level = 6;
wpt = wpdec(y,level,'sym8');
[spec,time,freq] = wpspectrum(wpt,fs,'plot');
%now compute the short time fourier transform:
figure;
windowsize = 128;
window = hanning(windowsize);
nfft = windowsize;
noverlap = windowsize - 1;
[index_matrix,F,T] = spectrogram(y,window,noverlap,nfft,fs);
imagesc(T,F,log10(abs(index_matrix)));
set(gca,'ydir','normal');
xlabel('time [secs]');
ylabel('frequency [Hz]');
title('short time fourier transform spectrum');
%sum of two sine waves with frequencies of 64 and 128 Hz:
fs = 1000;
t_vec = 0:1/fs:2;
y = sin(128*pi*t_vec) + sin(256*pi*t_vec);
level = 6;
wpt = wpdec(y,level,'sym8');
[spec,time,freq] = wpspectrum(wpt,fs,'plot');
figure;
windowsize = 128;
window = hanning(windowsize);
nfft = windowsize;
noverlap = windowsize - 1;
[index_matrix,F,T] = spectrogram(y,window,noverlap,nfft,fs);
imagesc(T,F,log10(abs(index_matrix)));
set(gca,'ydir','normal');
xlabel('time [secs]');
ylabel('frequency [Hz]');
title('short time fourier transform spectrum');
%signal with a abrupt change in frequency from 16 to 64 Hz at two seconds:
fs = 500;
t_vec = 0:1/fs:4;
y = sin(32*pi*6).*(t_vec<2) + sin(128*pi*t_vec).*(t_vec>=2);
level = 6;
wpt = wpdec(y,level,'sym8');
[spec,time,freq] = wpspectrum(wpt,fs,'plot');
figure;
windowsize = 128;
window = hanning(windowsize);
nfft = windowsize;
noverlap = windowsize - 1;
[index_matrix,F,T] = spectrogram(y,window,noverlap,nfft,fs);
imagesc(T,F,log10(abs(index_matrix)));
set(gca,'ydir','normal');
xlabel('time [secs]');
ylabel('frequency [Hz]');
title('short time fourier transform spectrum');
%wavelet packet spectrum of a linear chirp:
fs = 1000;
t_vec = 0:1/fs:2;
y = sin(256*pi*t_vec.^2);
level = 6;
wpt = wpdec(y,level,'sym8');
[spec,time,freq] = wpspectrum(wpt,fs,'plot');
figure;
windowsize = 128;
window = hanning(windowsize);
nfft = windowsize;
noverlap = windowsize - 1;
[index_matrix,F,T] = spectrogram(y,window,noverlap,nfft,fs);
imagesc(T,F,log10(abs(index_matrix)));
set(gca,'ydir','normal');
xlabel('time [secs]');
ylabel('frequency [Hz]');
title('short time fourier transform spectrum');
%wavelet packet spectrum of quadratic chirp:
y = wnoise('quadchirp',10); %WHAT MORE CAN I DO HERE!??!?!!?
len = length(y);
t_vec= linspace(0,5,len);
fs = 1/t_vec(2);
level = 6;
wpt = wpdec(y,level,'sym8');
[spec,time,freq] = wpspectrum(wpt,fs,'plot');
figure;
windowsize = 128;
window = hanning(windowsize);
nfft = windowsize;
noverlap = windowsize - 1;
[index_matrix,F,T] = spectrogram(y,window,noverlap,nfft,fs);
imagesc(T,F,log10(abs(index_matrix)));
set(gca,'ydir','normal');
xlabel('time [secs]');
ylabel('frequency [Hz]');
title('short time fourier transform spectrum');

%get in wfun the approximat values of Wn for n=0 to 7, cmputed on 1 1/2^5
%grid of the support xgrid:
[wfun,xgrid] = wpfun('db1',7,5);



%(40). RECONSTRUCTING A SIGNAL APPROXIMATION FROM A NODE
%you can use the function wprcoef to reconstruct an approximation to your
%signal from any node in the wavelet packet tree. this is true irrespective
%of whether you are working with a full wavelet packet tree, or a subtree
%determined by an optimality criterion. use wpcoef if you want to extract
%the wavelet packet coefficients form a node without reconstructing an
%approximation to the signal.
load noisdopp;
noisy_doppler_signal = noisedopp;
dwtmode('per'); %periodization mode
T = wpdec(noisy_doppler_signal,5,'sym4');
plot(T);
%exctract the wavelet packet coefficients from node 16:
wpc = wpcoef(T,16); %wpc is length 64
%obtain an approximation ot the signal from node 16:
rwpc = wprcoef(T,16); %rwpc is length 1024
plot(noisy_doppler_signal,'k');
hold on;
plot(rwpc,'b','linewidth',2);
axis tight;
%determine the optimum binary wavelet packet tree:
Topt = besttree(T);
plot(Topt);
%reconstruct an approxiomation to the signal from the (3,0) doublet (node 7):
rsig = wprcoef(Topt,7); %rsig is length 1024
plot(noisy_doppler_signal,'k');
hold on;
plot(rsig,'b','linewidth',2);
axis tight;
%if you know which doublet in the binary wavelet packet tree you want to
%extract, you can determine the node corresponding to that doublet with
%depo2ind. for example, to determine the node corresponding to the doublet(3,0), enter:
node = depo2ind(2,[3,0]);


%(41). THRESHOLD SELECTION RULES:
threshold = thselect(y,threshold_selection_string);
%tptr is a string:
%(1). 'rigrsure' - selection using principle of stein's unbiased risk estimate (SURE)
%(2). 'sqtwolog' - fixed form (universal) threshold equal to: sqrt(2*ln(N)), N=signal length
%(3). 'heursure' - selection using a mixture of the first two options
%(4). 'minimaxi' - selection using minimax principle

%(1).'rigruse' uses for the soft threshold estimator a threshold selection rule
%based on stein's unbiased estimate of risk (quadratic loss function). you
%get an estimate of the risk for a particular threshold value t. minimizing
%the risks in t gives a selection of the threshold value
%(2). 'sqtwolog' uses a fixed form threshold yielding minimax performance
%multiplied by a small factor proportional to log(length(s))
%(3). 'heursure' is a mixture of the two previous options. as a result, if
%the SNR is very small, the SURE estimate is very noisy. so if such a
%situation is detected, the fixed form threshold is used.
%(4). 'minimaxi' uses a fixed threshold chosen to yield minimax performance
%for mean square error against an ideal procedure. the minimax principle is
%used in statistics to design estimators. since the denoised signal can be
%assimilated to the estimator of the unknown regression function, the
%minimax estimator is the option that realizes the minimum, over a given
%set of functions, of the maximum mean square error

rng default;
signal = randn(1000,1);
thr_rigrsure = thselect(signal,'rigrsure');
thr_univthresh = thselect(signal,'sqtwolog');
thr_heursure = thselect(signal,'heursure');
thr_minimaxi = thselect(signal,'minimaxi');
histogram(signal);
h = findobj(gca,'type','patch');
set(h,'facecolor',[0.7,0.7,0.7],'edgecolor','w');
hold on;
plot([thr_rigrsure,thr_rigrsure],[0,300],'linewidth',2);
plot([thr_univthresh,thr_univthresh],[0,300],'r','linewidth',2);
plot([thr_minimaxi,thr_minimaxi],[0,300],'k','linewidth',2);
plot([-thr_rigrsure,-thr_rigrsure],[0,300],'linewidth',2);
plot([-thr_univthresh,-thr_univthresh],[0,300],'r','linewidth',2);
plot([-thr_minimaxi,-thr_minimaxi],[0,300],'k','linewidth',2);


%(42). dealing with unscaled noise and nonwhite noise:
%usually in practice the basic model cannot be used directly. we examine
%here the options available to deal with model deviations in the main
%denoising function wden. the simplest use of wden is:
sd = wden(index_matrix,threshold_selection_string,soft_or_hard_threshold,scal,n,wav);
%the parameter sorh specifies the thresholding of detils coefficients of
%the decomposition at leven n of s by the wavelet called wav. the remaining
%parameter scal is to be specified. it corresponds to threshold's rescaling methods:
%'one' = basic model
%'sln' = basic model with unscaled noise ???
%'mln' = basic model with nonwhite noise
%-the 'sln' handles threshold rescaling using a single estimation of level
%noise based on the first level coefficients
%-when you suspect a nonwhite noise e, thresholds must be rescaled by a
%level dependent estimation of the level noise. the same kind of strategy
%as in the previous option is used by estimating sigma_level level by level.
%-for a more general procedure, the wdencmp function performs wavelet
%coefficients thresholding for both denoising and compression purposes,
%while directly handling one-dimensional and two-dimensional data. it
%allows you to define your own thresholding strategy selecting in:
xd = wdencmp(opt,x,wav,n,threshold,soft_or_hard_threshold,keep_approximation);
%opt = 'gbl' and thr is a positive real number for uniform threshold
%opt = 'lvd' and thr is a vector for level dependent threshold
%keepapp = 1 to keep approximation coefficients
%keepapp = 0 to allow approximation coefficients thresholding
%x is the signal to be denoised and wav,n,sorh are the same as above.

%denoising in action:
sqrt_snr = 4;
init = 2055615866;
%generate original signal xref and a noisy version x adding a standard gaussian white noise:
[xref,x] = wnoise(1,11,sqrt_snr,init);
%[X,XN] = wnoise(FUN,N,SQRT_SNR,INIT)
% FUN = 1 or FUN = 'blocks'     
% FUN = 2 or FUN = 'bumps'      
% FUN = 3 or FUN = 'heavy sine'
% FUN = 4 or FUN = 'doppler'   
% FUN = 5 or FUN = 'quadchirp'   
% FUN = 6 or FUN = 'mishmash'  


%denoise noisy signal using soft heuristic SURE thresholding and scaled
%noise option, on detail coefficients obtained from the decomposition ox,
%at level 3 by sym8 wavelet.
xd = wden(x,'heursure','s','one',3,'sym8');

%
load leleccum;
electricity_use_signal = leleccum;
indx = 2000:3450;
x = electricity_use_signal(indx);
%find first value in order to avoid edge effects (REMEMBER THIS!!!!):
deb = x(1);
%denoise signal using soft fixed form thresholding and unknown noise option:
xd = wden(x-deb,'sqtwolog','s','mln',3,'db3') + deb;



%(43). IMAGE DENOISING:
load woman;
womag_image = X;
%generate noisy image:
x = womag_image + 15*randn(size(womag_image));
%find default values. in this case fixed form threshold is used with
%estimation of level noise, thresholding mode is soft and the
%approximation coefficients are kept:
[threshold,soft_or_hard_threshold,keep_approximation] = ddencmp('den','wv',x);
%thr is equal to estimated_sigma*sqrt(log(prod(size(X))))!!!!!!!

%denoise image using global thresholding option:
xd = wdencmp('gbl',x,'sym4',2,threshold,soft_or_hard_threshold,keep_approximation);
colormap(pink(255));
sm = size(map,1);
subplot(2,2,1);
image(wcodemat(womag_image,sm));
title('original image');
subplot(2,2,2);
image(wcodemat(x,sm));
title('noisy image');
subpot(2,2,3);
image(wcodemat(xd,sm));
title('denoised image');



%(44). 1D wavelet variance adaptive thresholding:
%the idea is to define level by level time dependent thresholds, and then
%increase the capability of the denoising strategies to handle
%nonstationary variance noise models.

%let us generate a signal from a fixed design regression model with two
%noise variance change points located at positions 200 and 600:
x = wnoise(1,10);
%generate noisy blocks with change points:
bb = randn(1,length(x));
cp1 = 200; %cp = change point
cp2 = 600;
x = x + [bb(1:cp1),bb(cp1+1:cp2)/4,bb(cp2+1:end)];
%perform a single level wavelet decomposition of the signal using db3:
wavelet_name = 'db3';
decomposition_level = 1;
[coefficients_mat,levels_vec] = wavedec(x,decomposition_level,wavelet_name);
%reconstruct detail at level 1:
det = wrcoef('d',coefficients_mat,levels_vec,wavelet_name,1);
%the reconstructed detail at level 1 reconvered at this stage is almost
%signal free. it captures the main features of the noise from a change
%points detection viewpoint if the interesting part of the signal has a
%sparse wavelet representation. to remove almost all the signal, we replace
%the biggest values by the mean:

%to remove almost all the signal, replace 2% of biggest values by the mean:
x = sort(abs(det));
v2p100 = x(fix(length(x)*0.98));
ind = find(abs(det)>v2p100);
det(ind) = mean(det);
%use the wvarchg function to estimate the change points with the following
%parameters: 
%1. the minimum delay between two change points is d=10 - where is this???
%2. the maximum number of change points is 5:  
[cp_est,kopt,t_est] = wvarchg(det,5); 
%kopt is the proposed number of change points
%for 2<=i<=6, t_est(i,1:i-1) contains the i-1 instants of the variance change points

%replace the estimated change points.
cp_est = t_est(kopt+1,1:kopt);




%(45). WAVELET DENOISING:
rng default; %????
[X,XN] = wnoise('bumps',10,sqrt(6));
subplot(2,1,1);
plot(X);
title('original signal');
AX = gca;
AX.YLim = [0,12];
subplot(2,1,2);
plot(XN);
title('noisy signal');
AX = gca;
AX.YLim = [0,12];
%denoise the signal down to level 4 using the undecimated wavelet transform. 
xdMODWT = wden(XN,'modwtsqtwolog','s','mln',4,'sym4');
figure;
plot(X,'r');
hold on;
plot(xdMODWT);
legend('original signal','denoised signal','location','northeastoutside');
aixs tight;
hold off;

%you can also use wavelets to denoise signals in which the noise is nonuniform:
load leleccum;
electricity_use_signal = leleccum;
indx = 2000:3450;
x = electricity_use_signal(indx);
plot(x);
grid on;
%you want to use different thresholding in the initial part of the signal then the latter.
%you can use cmddenoise to determine the optimal number of intervals to
%denoise and denoise the signal. in this example, use the 'db3' wavelet and
%decompose the data down to level 3.
%  cmddenoise Interval dependent denoising 
%     SIGDEN = cmddenoise(SIG,WNAME,LEVEL) performs an interval  
%     dependent denoising of the signal SIG, using a wavelet 
%     decomposition at the level LEVEL with a wavelet which 
%     name is WNAME. SIGDEN is the denoised signal.
%  
%     General Syntax:
%         [SIGDEN,COEFS,thrParams,int_DepThr_Cell,BestNbOfInt] =
%             cmddenoise(SIG,WNAME,LEVEL)
%         [...] = cmddenoise(SIG,WNAME,LEVEL,SORH)
%         [...] = cmddenoise(SIG,WNAME,LEVEL,SORH,NB_INTER)
%         [...] = cmddenoise(SIG,WNAME,LEVEL,SORH,NB_INTER,thrParams)
%   
%         - SORH ('s' or 'h') stands for soft or hard thresholding
%           (see WTHRESH for more details). Default is SORH = 's'.
%         - NB_INTER is an integer giving the number of intervals
%           used for denoising. NB_INTER must be such that 0<=NB_INTER
%           and NB_INTER<=6. The default is computed automatically.
%   
%         - COEFS is a vector containing the wavelet decomposition
%           of SIGDEN.
%         - thrParams is a cell array of length LEVEL such that
%           thrParams{j} is a NB_INTER by 3 array. In this array, each  
%           row contains the lower and upper bounds of the thresholding 
%           interval and the threshold value.
%         - BestNbOfInt is the best number of intervals (computed  
%           automatically).
%         - thrParams{j} is equal to int_DepThr_Cell{NB_INTER} or to
%           int_DepThr_Cell{BestNbOfInt} depending on the inputs of
%           cmddenoise.
[SIGDEN,~,thrParams,~,BestNbOfInt] = cmddenoise(x,'db3',3);
%display the number of intervals and the sample values that delimit the intervals:
BestNbOfInt
thrParams{1}(:,1:2)
plot(SIDEN);
title('denoised signal');

%denoise signals:
load(fullfile(matlabroot,'examples','wavelet','jump.mat')); 
wavelet_name = 'bior3.5';
level = 5;
[coefficients_mat,index_matrix] = wavedec2(jump,level,wavelet_name);
threshold = wthrmngr('dw2ddenoLVL','penalhi',coefficients_mat,index_matrix,3);
soft_or_hard_threshold = 'S';
[XDEN,cfsDEN,dimCFS] = wdencmp('lvd',coefficients_mat,index_matrix,wavelet_name,level,threshold,soft_or_hard_threshold);
figure;
subplot(1,2,1);
imagesc(jump);
colormap gray;
axis off;
title('noisy image');
subplot(1,2,2);
imagesc(XDEN);
colormap gray;
axis off;
title('denoised image');



%(46). MULTIVARIATE WAVELET DENOISING:
load ex4mwden
original_signals = x_orig;
noisy_signals = x;
covar %see covariance matrix
%remove noise by simple multivariate thresholding
%the denoising strategy combines univariate wavelet denoising in the basis
%where the estimated noise covariance matrix is diagonal with noncentered
%PCA on approximations in the wavelet domain or with final PCA.

%first perform univariate denoising by typing the following to set the
%denoising paramteers:
level = 5;
wavelet_name = 'sym4';
threshold_selection_string = 'sqtwolog';
soft_or_hard_threshold = 's';
%set the PCA parameters by retaining all the principal components:
npc_app = 4;
npc_fin = 4;
%finally perform multivariate denoising by typing:
% wmulden Wavelet multivariate denoising.
%     [X_DEN,NPC,NESTCOV,DEC_DEN,PCA_Params,DEN_Params] = ...
%                 wmulden(X,LEVEL,WNAME,NPC_APP,NPC_FIN,TPTR,SORH)
%     or  [...] = wmulden(X,LEVEL,WNAME,'mode',EXTMODE,NPC_APP,...)
%     returns a denoised version X_DEN of the input matrix X.
%     The strategy combines univariate wavelet denoising in the  
%     basis where the estimated noise covariance matrix is  
%     diagonal with non-centered Principal Component 
%     Analysis (PCA) on approximations in the wavelet domain or
%     with final PCA.
%  
%     Input matrix X contains P signals of length N stored
%     columnwise where N > P. 
%  
%     Wavelet Decomposition Parameters.
%     ---------------------------------
%     The Wavelet decomposition is performed using the 
%     decomposition level LEVEL and the wavelet WNAME. 
%     EXTMODE is the extended mode for the DWT (default 
%     is returned by DWTMODE).
%  
%     If a decomposition DEC obtained using MDWTDEC is available, 
%     then you can use [...] = wmulden(DEC,NPC_APP) instead of
%     [...] = wmulden(X,LEVEL,WNAME,'mode',EXTMODE,NPC_APP).
%  
%     Principal Components Parameters NPC_APP and NPC_FIN.
%     ----------------------------------------------------
%     Input selection methods NPC_APP and NPC_FIN define the way  
%     to select principal components for approximations at level 
%     LEVEL in the wavelet domain and for final PCA after 
%     wavelet reconstruction respectively.
%  
%     If NPC_APP (resp. NPC_FIN) is an integer, it contains the number 
%     of retained principal components for approximations at  
%     level LEVEL (resp. for final PCA after wavelet reconstruction).
%     NPC_XXX must be such that:   0 <= NPC_XXX <= P.
%  
%     NPC_APP or NPC_FIN = 'kais' (resp. 'heur') selects
%     automatically the number of retained principal components
%     using the Kaiser's rule (resp. the heuristic rule).
%        - Kaiser's rule keeps the components associated with 
%          eigenvalues exceeding the mean of all eigenvalues.
%        - heuristic rule keeps the components associated with 
%          eigenvalues exceeding 0.05 times the sum of all 
%          eigenvalues.
%     NPC_APP or NPC_FIN = 'none' is equivalent to NPC_APP or 
%     NPC_FIN = P.
%  
%     De-Noising Parameters TPTR, SORH (See WDEN and WBMPEN).
%     -------------------------------------------------------
%     Default values are: TPTR = 'sqtwolog' and SORH = 's'.
%     Valid values for TPTR are:
%         'rigrsure','heursure','sqtwolog','minimaxi'
%         'penalhi','penalme','penallo'
%     Valid values for SORH are: 's' (soft) or 'h' (hard)
%  
%     Outputs.
%     --------
%     X_DEN is a denoised version of the input matrix X.
%     NPC is the vector of selected numbers of retained
%     principal components.
%     NESTCOV is the estimated noise covariance matrix obtained 
%     using the minimum covariance determinant (MCD) estimator.
%     DEC_DEN is the wavelet decomposition of X_DEN. See MDWTDEC
%     for more information on decomposition structure. 
%     PCA_Params is a structure such that:
%         PCA_Params.NEST = {pc_NEST,var_NEST,NESTCOV}
%         PCA_Params.APP  = {pc_APP,var_APP,npc_APP}
%         PCA_Params.FIN  = {pc_FIN,var_FIN,npc_FIN}
%     where: 
%         - pc_XXX is a P-by-P matrix of principal components.
%           The columns are stored according to the descending 
%           order of the variances.
%         - var_XXX is the principal component variances vector. 
%         - NESTCOV is the covariance matrix estimate for detail 
%           at level 1.
%     DEN_Params is a structure such that:
%         DEN_Params.thrVAL is a vector of length LEVEL which 
%         contains the threshold values for each level. 
%         DEN_Params.thrMETH is a string containing the name of 
%         denoising method (TPTR)
%         DEN_Params.thrTYPE  is a char containing the type of 
%         thresholding (SORH)
%  
%     Special cases.
%     --------------
%     [DEC,PCA_Params] = wmulden('estimate',DEC,NPC_APP,NPC_FIN)
%     returns the wavelet decomposition DEC and the Principal 
%     Components Estimates PCA_Params.
%  
%     [X_DEN,NPC,DEC_DEN,PCA_Params] = wmulden('execute',DEC,PCA_Params)
%     or  [...] = wmulden('execute',DEC,PCA_Params,TPTR,SORH) uses the
%     Principal Components Estimates PCA_Params previously computed.
%     
%     The input value DEC can be replaced by X, LEVEL and WNAME.
x_den = wmulden(noisy_signals,level,wavelet_name,npc_app,npc_fin,threshold_selection_string,soft_or_hard_threshold);

%now improve the first result by retaining fewer principal components:
%the results can be improved by taking advantage of the relationships
%between the signals, leading to an additional denosiing effect.
%to automatically selecet the numbers of retained principal components by
%Kaiser's rule (which keeps the components associated with eigenvalues
%exceeding the mean of all eigenvalues):
npc_app = 'kais';
npc_fin = 'kais';
[x_den,npc,nestco] = wmulden(noisy_signals,level,wavelet_name,npc_app,npc_fin,threshold_selection_string,soft_or_hard_threshold);
%display the number of retained principal components:
npc
%display the estimated noise covariance matrix
nestco

%MULTISCALE PRINCIPAL COMPONENTS ANALYSIS:
load ex4mwden;
%the multiscale PCA combines noncentered PCA on approximations and details
%in the wavelet domain and a final PCA. at each level, the most significant
%principal components are selected:
level = 5;
wavelet_name = 'sym4';
npc = 'kais';
% wmspca Multiscale Principal Component Analysis.
%     [X_SIM,QUAL,NPC,DEC_SIM,PCA_Params] = wmspca(X,LEVEL,WNAME,NPC) 
%     or [...] = wmspca(X,LEVEL,WNAME,'mode',EXTMODE,NPC)
%     returns a simplified version X_SIM of the input matrix X
%     obtained from the wavelet based multiscale PCA.
%  
%     Input matrix X contains P signals of length N stored
%     columnwise where N > P.
%  
%     Wavelet Decomposition Parameters.
%     ---------------------------------
%     The Wavelet decomposition is performed using the 
%     decomposition level LEVEL and the wavelet WNAME. 
%     EXTMODE is the extended mode for the DWT (default is
%     returned by DWTMODE).
%  
%     If a decomposition DEC obtained using MDWTDEC is available, 
%     then you can use [...] = wmspca(DEC,NPC) instead of
%     [...] = wmspca(X,LEVEL,WNAME,'mode',EXTMODE,NPC).
%  
%     Principal Components Parameter NPC.
%     -----------------------------------
%     If NPC is a vector, it must be of length LEVEL+2. It contains the   
%     number of retained principal components for each PCA performed:
%        - NPC(d) is the number of retained non-centered principal 
%  	     components for details at level d, for 1 <= d <= LEVEL,
%        - NPC(LEVEL+1) is the number of retained non-centered
%          principal components for approximations at level LEVEL, 
%        - NPC(LEVEL+2) is the number of retained principal components
%          for final PCA after wavelet reconstruction, 
%        NPC must be such that 0 <= NPC(d) <= P for 1 <= d <= LEVEL+2.
%  
%     If NPC = 'kais' (respectively 'heur'), the numbers of retained 
%     principal components are selected automatically using the  
%     Kaiser's rule (respectively the heuristic rule).
%        - Kaiser's rule keeps the components associated with 
%          eigenvalues exceeding the mean of all eigenvalues.
%        - heuristic rule keeps the components associated with 
%          eigenvalues exceeding 0.05 times the sum of all 
%          eigenvalues.
%     If NPC = 'nodet', the details are "killed" and all the
%     approximations are retained.
%  
%     Outputs.
%     --------
%     X_SIM is a simplified version of the matrix X.
%     QUAL is a vector of length P containing the quality of
%     column reconstructions given by the relative mean square   
%     errors in percent.
%     NPC is the vector of selected numbers of retained 
%     principal components.
%     DEC_SIM is the wavelet decomposition of X_SIM. See MDWTDEC
%     for more information on the decomposition structure. 
%     PCA_Params is a structure array of length LEVEL+2 such that:
%       PCA_Params(d).pc = PC where:
%          PC is a P-by-P matrix of principal components.
%          The columns are stored according to the descending    
%          order of the variances. 
%       PCA_Params(d).variances = VAR where: 
%          VAR is the principal component variances vector.   
%       PCA_Params(d).npc = NPC.
[x_sim,qual,npc] = wmspca(noisy_signals,level,wavelet_name,npc);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 1 - Basic filters, upsampling and downsampling.
N = 1024;
W = (-N/2:N/2-1) / (N/2);
% Low pass filter.
h0 = [0.5, 0.5];
H0 = fftshift(fft(h0,N));
plot(W,abs(H0))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('Frequency response of Haar lowpass filter: [1/2 1/2]')
% High pass filter.
h1 = [0.5, -0.5];
H1 = fftshift(fft(h1,N));
plot(W,abs(H1))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('Frequency response of Haar highpass filter [1/2 -1/2]')
% Linear interpolating lowpass filter.
hlin = [0.5 1 0.5];
H = fftshift(fft(hlin,N));
plot(W,abs(H))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('Frequency response of lowpass filter [1/2 1 1/2]')
% Upsampling.
u0 = [0.5 0 0.5 0];
U0 = fftshift(fft(u0,N));
plot(W,abs(U0))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('Fourier transform of [1/2 0 1/2 0]')
% Downsampling.
x = [-1 0 9 16 9 0 -1] / 16;
X = fftshift(fft(x,N));
plot(W,abs(X))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('Fourier transform of x = [-1 0 9 16 9 0 -1] / 16')

x2 = [-1 9 9 -1] / 16;
X2 = fftshift(fft(x2,N));
plot(W,abs(X2))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('Fourier transform of [-1 9 9 -1] / 16')

XX = fftshift(fft(x,2*N));   % X(w)
XX2 = XX(N/2+1:3*N/2);       % X(w/2)
XXPi = fftshift(XX);         % X(w+pi)
XX2Pi = XXPi(N/2+1:3*N/2);   % X(w/2+pi)
Y = (XX2 + XX2Pi) / 2;
plot(W,abs(Y))
xlabel('Angular frequency (normalized by \pi)')
ylabel('Fourier transform magnitude')
title('[X(\omega/2) + X(\omega/2+pi)]/2')




% Product filter examples
%p = 2
switch p
case 1,
  % Degree 2
  b = [1 2 1];  % (1 + z^-1)^2
  q = 1 / 2;
  p0 = [1 2 1] / 2;   % conv(b, q)

case 2,
  % Degree 6
  b = [1 4 6 4 1];  % (1 + z^-1)^4
  q = [-1 4 -1] / 16;
  p0 = [-1 0 9 16 9 0 -1] / 16;  % conv(b, q)

case 3,
  % Degree 10
  b = [1 6 15 20 15 6 1];   % (1 + z^-1)^6
  q = [3 -18 38 -18 3] / 256;
  p0 = [3 0 -25 0 150 256 150 0 -25 0 3] / 256;  % conv(b,q)

case 4,
  % Degree 14
  b = [1 8 28 56 70 56 28 8 1];  % (1 + z^-1)^8
  q = [-5 40 -131 208 -131 40 -5] / 2048;
  p0 = [-5 0 49 0 -245 0 1225 2048 1225 0 -245 0 49 0 -5] / 2048;  % conv(b,q)

otherwise,
  % Degree 4p-2
  [p0,b,q] = prodfilt(p);
end
zplane(p0);
title(sprintf('Zeros of the product filter with degree %d', 4*p-2))
pause
[P,W] = dtft(p0,512);
plot(W/pi, abs(P))
xlabel('Angular frequency (normalized by pi)')
ylabel('Frequency response magnitude')
title(sprintf('Frequency response of the product filter with degree %d', 4*p-2))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1-D signal analysis

% biorwavf generates symmetric biorthogonal wavelet filters.
% The argument has the form biorNr.Nd, where
%    Nr = number of zeros at pi in the synthesis lowpass filter, s[n].
%    Nd = number of zeros at pi in the analysis lowpass filter, a[n].
% We find the famous Daubechies 9/7 pair, which have Nr = Nd = 4.  
% Note: In earlier versions of the Matlab Wavelet Toolbox (v2.1 and 
% below,) the vectors s and a are zero-padded to make their lengths equal:
%    a[-4] a[-3] a[-2] a[-1] a[0] a[1] a[2] a[3] a[4]
%      0   s[-3] s[-2] s[-1] s[0] s[1] s[2] s[3]   0
[index_matrix,approximation_coefficients_level3]= biorwavf('bior4.4');  

% Find the zeros and plot them.
close all
clf
fprintf(1,'Zeros of H0(z)')
roots(approximation_coefficients_level3)
subplot(1,2,1)
zplane(approximation_coefficients_level3)
title('Zeros of H0(z)')

fprintf(1,'Zeros of F0(z)')
roots(index_matrix)
subplot(1,2,2)
zplane(index_matrix)          % Note: there are actually 4 zeros clustered at z = -1.
title('Zeros of F0(z)')
pause


% Determine the complete set of filters, with proper alignment.
% Note: Matlab uses the convention that a[n] is the flip of h0[n].
%    h0[n] = flip of a[n], with the sum normalized to sqrt(2).
%    f0[n] = s[n], with the sum normalized to sqrt(2).
%    h1[n] = f0[n], with alternating signs reversed (starting with the first.)
%    f1[n] = h0[n], with alternating signs reversed (starting with the second.)
[h0,h1,f0,f1] = biorfilt(approximation_coefficients_level3, index_matrix);

clf
subplot(2,2,1)
stem(0:8,h0(2:10))
ylabel('h0[n]')
xlabel('n')
subplot(2,2,2)
stem(0:6,f0(2:8))
ylabel('f0[n]')
xlabel('n')
v = axis; axis([v(1) 8 v(3) v(4)])
subplot(2,2,3)
stem(0:6,h1(2:8))
ylabel('h1[n]')
xlabel('n')
v = axis; axis([v(1) 8 v(3) v(4)])
subplot(2,2,4)
stem(0:8,f1(2:10))
ylabel('f1[n]')
xlabel('n')
pause


% Examine the Frequency response of the filters.
N = 512;
W = 2/N*(-N/2:N/2-1);
H0 = fftshift(fft(h0,N));
H1 = fftshift(fft(h1,N));
F0 = fftshift(fft(f0,N));
F1 = fftshift(fft(f1,N));
clf
plot(W, abs(H0), '-', W, abs(H1), '--', W, abs(F0), '-.', W, abs(F1), ':')
title('Frequency responses of Daubechies 9/7 filters')
xlabel('Angular frequency (normalized by pi)')
ylabel('Frequency response magnitude')
legend('H0', 'H1', 'F0', 'F1', 0)
pause


% Load a test signal. 
load noisdopp

x = noisdopp;
levels_vec = length(x);
clear noisdopp

% Compute the lowpass and highpass coefficients using convolution and
% downsampling.
y0 = dyaddown(conv(x,h0));
y1 = dyaddown(conv(x,h1));

% The function dwt provides a direct way to get the same result.
[yy0,yy1] = dwt(x,'bior4.4');

% Now, reconstruct the signal using upsamping and convolution.  We only
% keep the middle L coefficients of the reconstructed signal i.e. the ones
% that correspond to the original signal.
xhat = conv(dyadup(y0),f0) + conv(dyadup(y1),f1);
xhat = wkeep(xhat,levels_vec);

% The function idwt provides a direct way to get the same result.
xxhat = idwt(y0,y1,'bior4.4');

% Plot the results.
subplot(4,1,1);
plot(x)
axis([0 1024 -12 12])
title('Single stage wavelet decomposition')
ylabel('x')
subplot(4,1,2);
plot(y0)
axis([0 1024 -12 12])
ylabel('y0')
subplot(4,1,3);
plot(y1)
axis([0 1024 -12 12])
ylabel('y1')
subplot(4,1,4);
plot(xhat)
axis([0 1024 -12 12])
ylabel('xhat')
pause

% Next, we perform a three level decomposition.  The following
% code draws the structure of the iterated analysis filter bank.
clf
t_vec = wtree(x,3,'bior4.4');
plot(t_vec)
pause
close(2)

% For a multilevel decomposition, we use wavedec instead of dwt.
% Here we do 3 levels.  wc is the vector of wavelet transform
% coefficients.  l is a vector of lengths that describes the
% structure of wc.
[wc,levels_vec] = wavedec(x,3,'bior4.4');

% We now need to extract the lowpass coefficients and the various
% highpass coefficients from wc.
a3 = appcoef(wc,levels_vec,'bior4.4',3);
d3 = detcoef(wc,levels_vec,3);
d2 = detcoef(wc,levels_vec,2);
d1 = detcoef(wc,levels_vec,1);

clf
subplot(5,1,1)
plot(x)
axis([0 1024 -22 22])
ylabel('x')
title('Three stage wavelet decomposition')
subplot(5,1,2)
plot(a3)
axis([0 1024 -22 22])
ylabel('a3')
subplot(5,1,3)
plot(d3)
axis([0 1024 -22 22])
ylabel('d3')
subplot(5,1,4)
plot(d2)
axis([0 1024 -22 22])
ylabel('d2')
subplot(5,1,5)
plot(d1)
axis([0 1024 -22 22])
ylabel('d1')
pause

% We can reconstruct each branch of the tree separately from the individual
% vectors of transform coefficients using upcoef.
ra3 = upcoef('a',a3,'bior4.4',3,1024);
rd3 = upcoef('d',d3,'bior4.4',3,1024);
rd2 = upcoef('d',d2,'bior4.4',2,1024);
rd1 = upcoef('d',d1,'bior4.4',1,1024);

% The sum of these reconstructed branches gives the full recontructed
% signal.
xhat = ra3 + rd3 + rd2 + rd1;

clf
subplot(5,1,1)
plot(x)
axis([0 1024 -10 10])
ylabel('x')
title('Individually reconstructed branches')
subplot(5,1,2)
plot(ra3)
axis([0 1024 -10 10])
ylabel('ra3')
subplot(5,1,3)
plot(rd3)
axis([0 1024 -10 10])
ylabel('rd3')
subplot(5,1,4)
plot(rd2)
axis([0 1024 -10 10])
ylabel('rd2')
subplot(5,1,5)
plot(rd1)
axis([0 1024 -10 10])
ylabel('rd1')
pause

clf
plot(xhat-x)
title('Reconstruction error (using upcoef)')
axis tight
pause

% We can also reconstruct individual branches from the full vector of
% transform coefficients, wc.
rra3 = wrcoef('a',wc,levels_vec,'bior4.4',3);
rrd3 = wrcoef('d',wc,levels_vec,'bior4.4',3);
rrd2 = wrcoef('d',wc,levels_vec,'bior4.4',2);
rrd1 = wrcoef('d',wc,levels_vec,'bior4.4',1);
xxhat = rra3 + rrd3 + rrd2 + rrd1;

clf
plot(xxhat-x)
title('Reconstruction error (using wrcoef)')
axis tight
pause

% To reconstruct all branches at once, use waverec.
xxxhat = waverec(wc,levels_vec,'bior4.4');

clf
plot(xxxhat-x)
axis tight
title('Reconstruction error (using waverec)')
pause
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 2D image analysis.

% Load a test image.  Matlab test images consist of a matrix, X,
% color palette, map, which maps each value of the matrix to a
% color.  Here, we will apply the Discrete Wavelet Transform to X.
load woman2
%load detfingr; X = X(1:200,51:250);

close all
clf
image(X)
colormap(map)
axis image; set(gca,'XTick',[],'YTick',[]); title('Original')
pause

% We will use the 9/7 filters with symmetric extension at the
% boundaries.
dwtmode('sym')
wavelet_name = 'bior4.4'

% Plot the structure of a two stage filter bank.
t_vec = wtree(X,2,'bior4.4');
plot(t_vec)
pause
close(2)

% Compute a 2-level decomposition of the image using the 9/7 filters.
[wc,index_matrix] = wavedec2(X,2,wavelet_name);

% Extract the level 1 coefficients.
a1 = appcoef2(wc,index_matrix,wavelet_name,1);         
h1 = detcoef2('h',wc,index_matrix,1);           
v1 = detcoef2('v',wc,index_matrix,1);           
d1 = detcoef2('d',wc,index_matrix,1);           

% Extract the level 2 coefficients.
a2 = appcoef2(wc,index_matrix,wavelet_name,2);
h2 = detcoef2('h',wc,index_matrix,2);
v2 = detcoef2('v',wc,index_matrix,2);
d2 = detcoef2('d',wc,index_matrix,2);

% Display the decomposition up to level 1 only.
ncolors = size(map,1);              % Number of colors.
sz = size(X);
cod_a1 = wcodemat(a1,ncolors); cod_a1 = wkeep(cod_a1, sz/2);
cod_h1 = wcodemat(h1,ncolors); cod_h1 = wkeep(cod_h1, sz/2);
cod_v1 = wcodemat(v1,ncolors); cod_v1 = wkeep(cod_v1, sz/2);
cod_d1 = wcodemat(d1,ncolors); cod_d1 = wkeep(cod_d1, sz/2);
image([cod_a1,cod_h1;cod_v1,cod_d1]);
axis image; set(gca,'XTick',[],'YTick',[]); title('Single stage decomposition')
colormap(map)
pause

% Display the entire decomposition upto level 2.
cod_a2 = wcodemat(a2,ncolors); cod_a2 = wkeep(cod_a2, sz/4);
cod_h2 = wcodemat(h2,ncolors); cod_h2 = wkeep(cod_h2, sz/4);
cod_v2 = wcodemat(v2,ncolors); cod_v2 = wkeep(cod_v2, sz/4);
cod_d2 = wcodemat(d2,ncolors); cod_d2 = wkeep(cod_d2, sz/4);
image([[cod_a2,cod_h2;cod_v2,cod_d2],cod_h1;cod_v1,cod_d1]);
axis image; set(gca,'XTick',[],'YTick',[]); title('Two stage decomposition')
colormap(map)
pause

% Here are the reconstructed branches
ra2 = wrcoef2('a',wc,index_matrix,wavelet_name,2);
rh2 = wrcoef2('h',wc,index_matrix,wavelet_name,2);
rv2 = wrcoef2('v',wc,index_matrix,wavelet_name,2);
rd2 = wrcoef2('d',wc,index_matrix,wavelet_name,2);

ra1 = wrcoef2('a',wc,index_matrix,wavelet_name,1);
rh1 = wrcoef2('h',wc,index_matrix,wavelet_name,1);
rv1 = wrcoef2('v',wc,index_matrix,wavelet_name,1);
rd1 = wrcoef2('d',wc,index_matrix,wavelet_name,1);

cod_ra2 = wcodemat(ra2,ncolors);
cod_rh2 = wcodemat(rh2,ncolors);
cod_rv2 = wcodemat(rv2,ncolors);
cod_rd2 = wcodemat(rd2,ncolors);
cod_ra1 = wcodemat(ra1,ncolors);
cod_rh1 = wcodemat(rh1,ncolors);
cod_rv1 = wcodemat(rv1,ncolors);
cod_rd1 = wcodemat(rd1,ncolors);
subplot(3,4,1); image(X); axis image; set(gca,'XTick',[],'YTick',[]); title('Original')
subplot(3,4,5); image(cod_ra1); axis image; set(gca,'XTick',[],'YTick',[]); title('ra1')
subplot(3,4,6); image(cod_rh1); axis image; set(gca,'XTick',[],'YTick',[]); title('rh1')
subplot(3,4,7); image(cod_rv1); axis image; set(gca,'XTick',[],'YTick',[]); title('rv1')
subplot(3,4,8); image(cod_rd1); axis image; set(gca,'XTick',[],'YTick',[]); title('rd1')
subplot(3,4,9); image(cod_ra2); axis image; set(gca,'XTick',[],'YTick',[]); title('ra2')
subplot(3,4,10); image(cod_rh2); axis image; set(gca,'XTick',[],'YTick',[]); title('rh2')
subplot(3,4,11); image(cod_rv2); axis image; set(gca,'XTick',[],'YTick',[]); title('rv2')
subplot(3,4,12); image(cod_rd2); axis image; set(gca,'XTick',[],'YTick',[]); title('rd2')
pause

% Adding together the reconstructed average at level 2 and all of
% the reconstructed details gives the full reconstructed image.
Xhat = ra2 + rh2 + rv2 + rd2 + rh1 + rv1 + rd1;
sprintf('Reconstruction error (using wrcoef2) = %g', max(max(abs(X-Xhat))))

% Another way to reconstruct the image.
XXhat = waverec2(wc,index_matrix,wavelet_name);
sprintf('Reconstruction error (using waverec2) = %g', max(max(abs(X-XXhat))))

% Compression can be accomplished by applying a threshold to the
% wavelet coefficients.  wdencmp is the function that does this.
% 'h' means use hard thresholding. Last argument = 1 means do not
% threshold the approximation coefficients.
%    perfL2 = energy recovery = 100 * ||wc_comp||^2 / ||wc||^2.
%             ||.|| is the L2 vector norm.
%    perf0 = compression performance = Percentage of zeros in wc_comp.
threshold = 20;                                                    
[X_comp,wc_comp,s_comp,perf0,perfL2] = wdencmp('gbl',wc,index_matrix,wavelet_name,2,threshold,'h',1);

clf
subplot(1,2,1); image(X); axis image; set(gca,'XTick',[],'YTick',[]);
title('Original')
cod_X_comp = wcodemat(X_comp,ncolors);
subplot(1,2,2); image(cod_X_comp); axis image; set(gca,'XTick',[],'YTick',[]);
title('Compressed using global hard threshold')
xlabel(sprintf('Energy retained = %2.1f%% \nNull coefficients = %2.1f%%',perfL2,perf0))
pause

% Better compression can be often be obtained if different thresholds
% are allowed for different subbands.
thr_h = [21 17];        % horizontal thresholds.              
thr_d = [23 19];        % diagonal thresholds.                
thr_v = [21 17];        % vertical thresholds.                
threshold = [thr_h; thr_d; thr_v];
[X_comp,wc_comp,s_comp,perf0,perfL2] = wdencmp('lvd',X,wavelet_name,2,threshold,'h');

clf
subplot(1,2,1); image(X); axis image; set(gca,'XTick',[],'YTick',[]);
title('Original')
cod_X_comp = wcodemat(X_comp,ncolors);
subplot(1,2,2); image(cod_X_comp); axis image; set(gca,'XTick',[],'YTick',[]);
title('Compressed using variable hard thresholds')
xlabel(sprintf('Energy retained = %2.1f%% \nNull coefficients = %2.1f%%',perfL2,perf0))

% Return to default settings.
dwtmode('zpd')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Polyphase filter implementation

% Determine the filters.
[h0,h1,f0,f1] = orthfilt(dbwavf('db2'));
M = length(h0);

% Load a signal.
load noisdopp
noisy_doppler_signal = noisedopp;
N = length(noisy_doppler_signal);

% Change the signal into polyphase form.
xeven = dyaddown(noisy_doppler_signal,1);  % Even part
xodd = dyaddown(noisy_doppler_signal,0);   % Odd part
xodd = [0 xodd(1:N/2-1)];
X = [xeven; xodd];

% Construct the polyphase matrix.
H = zeros(2,2,M/2);
H(1,1,:) = dyaddown(h0,1);  % h0,even[n]
H(1,2,:) = dyaddown(h0,0);  % h0,odd[n]
H(2,1,:) = dyaddown(h1,1);  % h1,even[n]
H(2,2,:) = dyaddown(h1,0);  % h1,odd[n]

H(:,:,1)
H(:,:,2)

% Run the polyphase filter.
Y = polyfilt(H,X);

% Plot the results.
levels_vec = N/2;
n = 0:levels_vec-1;
clf
subplot(2,1,1)
plot(n,Y(1,1:levels_vec))
axis tight
xlabel('Sample number')
ylabel('Lowpass')
title('Output from polyphase filter')
subplot(2,1,2)
plot(n,Y(2,1:levels_vec))
axis tight
xlabel('Sample number')
ylabel('Highpass')
pause

% Compute the results using the direct approach.
y0 = dyaddown(conv(noisy_doppler_signal,h0),1);
y1 = dyaddown(conv(noisy_doppler_signal,h1),1);

% Now compare the results.
clf
subplot(2,1,1)
plot(n,Y(1,1:levels_vec)-y0(1:levels_vec))
axis tight
xlabel('Sample number')
ylabel('Lowpass difference')
title('Difference in outputs produced by polyphase and direct forms')
subplot(2,1,2)
plot(n,Y(2,1:levels_vec)-y1(1:levels_vec))
axis tight
xlabel('Sample number')
ylabel('Highpass difference')
pause

% Plot the determinant of the polyphase matrix as a function of frequency.
R = 32;
W = 2/R*(-R/2:R/2-1);
H0even = fftshift(fft(H(1,1,:),R));
H0odd = fftshift(fft(H(1,2,:),R));
H1even = fftshift(fft(H(2,1,:),R));
H1odd = fftshift(fft(H(2,2,:),R));
delta = zeros(1,R);
delta(:) = H0even .* H1odd - H0odd .* H1even;
clf
plot(W,abs(delta),'x-')
axis([-1 1 0 1.5])
xlabel('Angular frequency (normalized by pi)')
ylabel('Magnitude of determinant')
title('Determinant of the polyphase matrix')
pause

% Verify that the filter is orthogonal i.e. Hp'(w*) Hp(w) = I
A11 = zeros(1,R);
A11(:) = abs(H0even).^2 + abs(H1even).^2;
A12 = zeros(1,R);
A12(:) = conj(H0even).*H0odd + conj(H1even).*H1odd;
A21 = zeros(1,R);
A21(:) = conj(H0odd).*H0even + conj(H1odd).*H1even;
A22 = zeros(1,R);
A22(:) = abs(H0odd).^2 + abs(H1odd).^2;
clf
subplot(4,1,1)
plot(W,A11,'x-');
axis([-1 1 0 1.5])
ylabel('A11')
title('Variation of A = Hp''(w*) Hp(w) with frequency')
subplot(4,1,2)
plot(W,A12,'x-')
axis([-1 1 0 1.5])
ylabel('A12')
subplot(4,1,3)
plot(W,A21,'x-')
axis([-1 1 0 1.5])
ylabel('A21')
subplot(4,1,4)
plot(W,A22,'x-')
axis([-1 1 0 1.5])
xlabel('Angular frequency (normalized by pi)')
ylabel('A22')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Example 6: Compute the samples of Daubechies scaling function and
% wavelet using the inverse DWT.

%p = 2                                     % Number of zeros at pi.
N = 2 * p - 1;                            % Support of the scaling function
numlevels = 5;                            % Number of iterations/levels.
M = 2^numlevels;
levels_vec = M * N;
f0 = daub(N+1) / 2;                       % Synthesis lowpass filter.
f1 = (-1).^[0:N]' .* flipud(f0);          % Synthesis highpass filter.

% For the scaling function, we need to compute the inverse DWT with a delta
% for the approximation coefficients.  (All detail coefficients are set
% to zero.)
y = upcoef('a',[1;0],f0,f1,numlevels);    % Inverse DWT.
phi_scaling = M * [0; y(1:levels_vec)];

% For the wavelet, we need to compute the inverse DWT with a delta for the
% detail coefficients.  (All approximation coefficients and all detail
% coefficients at finer scales are set to zero.)
y = upcoef('d',[1;0],f0,f1,numlevels);    % Inverse DWT.
wavelet_name = M * [0; y(1:levels_vec)];

% Determine the time vector.
t_vec = [0:levels_vec]' / M;

% Plot the results.
plot(t_vec,phi_scaling,'-',t_vec,wavelet_name,'--')
legend('Scaling function','Wavelet')
title('Scaling function and wavelet by iteration of synthesis filter bank.')
xlabel('t')
pause

% Now compute the scaling function and wavelet by recursion.
% phivals (not part of the Matlab toolbox) does this.
[t1,phi1,w1] = phivals(daub(2*p),numlevels);

% Plot the results.
plot(t1,phi1,'-',t1,w1,'--')
legend('Scaling function','Wavelet')
title('Scaling function and wavelet by recursion.')
xlabel('t')
pause

% View the scaling functions side by side.
plot(t_vec,phi_scaling,'-',t1,phi1,'--')
legend('Scaling function using iteration','Scaling function using recursion')
title('Comparison of the two methods (recursion is exact.)')
xlabel('t')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Example 3a: Compute the samples of the biorthogonal scaling functions
% and wavelets.

[index_matrix,approximation_coefficients_level3]= biorwavf('bior2.2');           % 9/7 filters
[h0,h1,f0,f1] = biorfilt(approximation_coefficients_level3, index_matrix);

[x,phi_scaling,phitilde,psi_wavelet,psitilde] = biphivals(h0,h1,f0,f1,5);

plot(x,phi_scaling,'-',x,psi_wavelet,'-.')
legend('Primary scaling function', 'Primary wavelet')
pause

plot(x,phitilde,'--',x,psitilde,':')
legend('Dual scaling function', 'Dual wavelet')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Mallat pyramid decomposition.

p = 1;
p = input('Order of wavelet, p (defaults to 1) = ');
if isempty(p)
  p = 1;
end 
h = daub(2*p);
%J = 7;
J = input('Finest resolution, J (defaults to 7) = ');
if isempty(J)
  J = 7;
end 
%nlevels = 4;
nlevels = input('Number of levels, nlevels (defaults to 4) = ');
if isempty(nlevels)
  nlevels = 4;
end 

clf
J1 = J+2;
nx = 2^J1;
dx = 1/nx;
x = (0:nx-dx/2)'/nx;
f = exp(-50*(x-0.5+dx/2).^2);
%f = ones(nx,1);
%f = x;
%f = sin(3*pi*x);
plot(x,f);
title('Original function, f(x)')
axis([0 1 0 1.1])
xlabel('x')
ylabel('f(x)')
pause

% Projection on to V_J.
cJ = scalecoeffs(f,1,h,J,J1);
PJf = expand(cJ,1,h,J,J1);
plot(x,PJf)
axis([0 1 0 1.1])
title(sprintf('Projection on to finest scale, V_%d',J))
xlabel('x')
ylabel(sprintf('P_%df(x)',J))
pause

% Create the Mallat pyramid.
cJdec = fwt(cJ,nlevels,h,0);

% Extract the wavelet coefficients and compute the
% projections on to V_J0, W_J0, W_J0+1, ..., W_J
J0 = J - nlevels;
n = 2^J0;
cJ0 = cJdec(1:n);
PJ0f = expand(cJ0,1,h,J0,J1);
subplot(nlevels+2,1,1)
plot(x,PJ0f)
xlabel('x')
str = sprintf('P_%df(x)',J0);
ylabel(str)


i = 2;
total = PJ0f;
subplot(nlevels+2,1,nlevels+2);
plot(x,total)
ylabel('Total')
xlabel('x')
pause
strtot = str;
strtot2 = sprintf('Projections on to V_%d',J0);
for j = J0:J-1
  dj = cJdec(n+1:2*n);
  Qjf = wexpand(dj,1,h,j,J1);
  subplot(nlevels+2,1,i)
  plot(x,Qjf)
  xlabel('x')
  str = sprintf('Q_%df(x)',j);
  ylabel(str)
  strtot = strcat(strtot, ' + ', str);
  strtot2 = strcat(strtot2, sprintf(', W_%d',j));
  total = total + Qjf;
  subplot(nlevels+2,1,nlevels+2);
  plot(x,total)
  ylabel('Total')
  xlabel('x')
  if (j < J-1)
    pause
  end
  n = n*2;
  i = i+ 1;
end
subplot(nlevels+2,1,1)
title(strtot2)
pause

clf
subplot(211)
plot(x,total,'-',x,f,':')
xlabel('x')
ylabel(strtot)
title('Multiple scales combined are equivalent to single scale projection')
legend('Projection','Original function')

subplot(212)
plot(x,PJf,'-',x,f,':')
xlabel('x')
ylabel(sprintf('P_%df(x)',J))
title('Single scale projection')
legend('Projection','Original function')

display(sprintf('Approximation error in either case is %e',norm(total-f)'))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Approximation of a functions by the scaling function and its translates. 

p = input('Order of wavelet, p (defaults to 1) = ');
if isempty(p)
  p = 1;
end
h = daub(2*p);
%h = [-1 0 9 16 9 0 -1 0]'/16;

N = length(h);
[x1,phi_scaling] = phivals(h,4);

nx = 2^4;
levels_vec = 12;
n = levels_vec * nx;
x = (0:n)/n*6;
sf = [phi_scaling; zeros((levels_vec-N+1)*2^4,1)];

for num = 1:5

  y = zeros(size(sf));
  tmp = ((-N+2)*nx:(-N+2+32)*nx-1)'/n;
  v = x'/6;
  if num == 1
    ck = [zeros(N-2,1); h; zeros(levels_vec,1)];
  elseif num == 2
    f = ones(n+1,1);
  elseif num == 3
    f = v;
    ck = scalecoeffs(tmp,32,h,0,4);
  elseif num == 4
    f = 4*v.*v-4*v+1;
    ck = scalecoeffs(4*tmp.*tmp-4*tmp+1,32,h,0,4);
  elseif num == 5
    f = -6*v.*v.*v+9*v.*v-3*v;
    ck = scalecoeffs(-6*tmp.*tmp.*tmp+9*tmp.*tmp-3*tmp,32,h,0,4);
  end

  clf
  minval = 0;
  maxval = 0;
  for k = -N+2:levels_vec-1
    if num == 2
      g = eoshift(sf,k*nx);
    else
      g = ck(k+N-1)*eoshift(sf,k*nx);
    end
    hold on
    plot(x,g,':')
    hold off
    y = y + g;
    minval = min(minval,min(g));   
    maxval = max(maxval,max(g));    
  end
  hold on
  plot(x,y)
  hold off

  minval = min(0,min(y));
  maxval = max(y);
  index_matrix = maxval - minval;
  minval = minval - 0.2 * index_matrix;
  maxval = maxval + 0.2 * index_matrix;
  axis([min(x) max(x) minval maxval])
  xlabel('x')
  ylabel('f(x)')
  if num == 1
    title('Representation of a scaling function by its translates')
    v = axis;
    v(2) = 6;
    axis(v);
  elseif num == 2
    title('Representation of a constant function by translates of a scaling function')
  elseif num == 3
    title('Representation of a linear function by translates of a scaling function')
  elseif num == 4
    title('Representation of a quadratic by translates of a scaling function')
  elseif num == 5
    title('Representation of a cubic by translates of a scaling function')
  end

  if num > 1
    pause
    plot(x(1:n),f(1:n)-y(1:n))
    xlabel('x')
    ylabel('f(x)-f_{approx}(x)')
    title('Approximation error')
  end
  pause

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Example 4: Examine how polynomial data behaves in the filter bank
% when the lowpass filter has p zeros at pi.

p = 3;              % number of zeros at pi.
%a = [1 1]/128;
approximation_coefficients_level3 = [1 3 3 1]/128;    % Coefficients of the polynomial (low order to high order.)
n = 0:127;
q = length(approximation_coefficients_level3) - 1;  % Degree of the polynomial.
len = length(n);
x = zeros(1,len);
nq = ones(1,len);
for k = 1:q+1
  x = x + approximation_coefficients_level3(k) * nq;
  nq = nq .* n;
end

% Compute the DWT.
N = 2 * p;
h0 = daub(N);
h1 = (-1).^[0:N-1]' .* flipud(h0);
[y0,y1] = dwt(x,h0,h1);

% Plot the results.
clf
subplot(3,1,1);
plot(x)
axis([0 len-1 min(x) max(x)])
title(sprintf('Wavelet transform of degree %d polynomial data.  H0(w) has %d zeros at pi.', q, p))
ylabel('x')
subplot(3,1,2);
plot(y0)
axis([0 len-1 min(y0) max(y0)])
ylabel('y0')
subplot(3,1,3);
plot(y1)
minval = min(0,min(y1(p:length(y1)-p+1)));
maxval = max(0,max(y1(p:length(y1)-p+1)));
index_matrix = maxval - minval;
minval = minval - 0.2*index_matrix;
maxval = maxval + 0.2*index_matrix;
axis([0 len-1 minval maxval])
ylabel('y1')

fprintf('Maximum value of y1 (excluding boundary effects) = %d.\n',max(abs(y1(p:length(y1)-p+1))))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Eigenvalues of the transition matrix, T = down2(2 H H')

%h = 1; p = 0;
%h = [1 1]' / 2; p = 1;  % smax = 0.5;
h = daub(6) / 2; p = 3; % smax = 1.0 
%h = [1 2 1]'/4; p = 2;  % smax = 1.5
%h = [1 4 6 4 1]' / 16; p = 4; 
%h = [2 1 -1]'/2; p = 1;
%[x,phi,psi] = phivals(h,5);

%[f,h] = biorwavf('bior2.2'); p = 2; h = h';  % 5/3
%[h,f] = biorwavf('bior2.2'); p = 2; h = h';  % 3/5
%[f,h] = biorwavf('bior4.4'); p = 4; h = h'; % 9/7
%[h,f] = biorwavf('bior4.4'); p = 4; h = h'; % 7/9
%h = [-1 0 9 16 9 0 -1]' / 32; f = 1; p = 4; % Halfband filter
%[h0,h1,f0,f1] = biorfilt(h',f);
%[x,phi,psi,phitilde,psitilde] = biphivals(h0,h1,f0,f1,5);

%plot(x,phi);
%pause

approximation_coefficients_level3 = conv(h, flipud(h));

% Method 1: Use the function given in Ch 7, p221.
T = down(approximation_coefficients_level3');

% Method 2: Use basic Matlab commands.
N = length(approximation_coefficients_level3);
v = [approximation_coefficients_level3; zeros(N,1)];
V = toeplitz(v, [v(1) zeros(1, N-1)]);
TT = 2 * dyaddown(V, 'r', 1);

if norm(T - TT) ~= 0
  error('Something is wrong!')
end

lambda = flipud(sort(eig(T)))


pause

hs = 1;
for i = 1:p
  hs = conv(hs, [1,1]/2);
end
hq = deconv(h, hs);
aq = conv(hq, flipud(hq));
TQ = down(aq');
lambdaQ = flipud(sort(eig(TQ)));
lambdaMax = max(lambdaQ) / 4^p;

disp(sprintf('Largest nonspecial eigenvalue = %0.14f', lambdaMax));
disp(sprintf('Smallest special eigenvalue = %0.14f', 1/2^(2*p-1)));
disp(sprintf('Smoothness, s_max, = %0.14f', -log2(lambdaMax)/2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Treatment of boundaries.

clf

fd = fopen('barbara.raw', 'r');
X = fread(fd, [512,512], 'uchar');
X = X';
map = (0:255)'/255 * ones(1,3);
colormap(map)
imagesc(uint8(X))
axis image; set(gca,'XTick',[],'YTick',[]); title('Single stage decomposition')
title('Original image')
pause

% We will use the 9/7 filters with symmetric extension at the
% boundaries.
dwtmode('zpd')  % zpd, per, sym, sp0, sp1
%wname = 'db5'
wavelet_name = 'bior4.4'

% Compute a 2-level decomposition of the image using the 9/7 filters.
[wc,index_matrix] = wavedec2(X,2,wavelet_name);

% Extract the level 1 coefficients.
a1 = appcoef2(wc,index_matrix,wavelet_name,1);         
h1 = detcoef2('h',wc,index_matrix,1);           
v1 = detcoef2('v',wc,index_matrix,1);           
d1 = detcoef2('d',wc,index_matrix,1);           

% Extract the level 2 coefficients.
a2 = appcoef2(wc,index_matrix,wavelet_name,2);
h2 = detcoef2('h',wc,index_matrix,2);
v2 = detcoef2('v',wc,index_matrix,2);
d2 = detcoef2('d',wc,index_matrix,2);

% Display the decomposition up to level 1 only.
image(uint8([a1/2,h1*10;v1*10,d1*10]))
axis image; set(gca,'XTick',[],'YTick',[]);
title('Single stage decomposition')
pause

% Display the entire decomposition upto level 2.
st = dwtmode('status','nodisp');
if (st == 'per')
  image(uint8([[a2/4,h2*10;v2*10,d2*10],h1*10;v1*10,d1*10]));
  axis image; set(gca,'XTick',[],'YTick',[]);
  title('Two stage decomposition')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 
% 
% % function [p0,b,q] = prodfilt(p)
% %
% % Generate the halfband product filter of degree 4p-2.
% %
% %              Kevin Amaratunga
% %              19 February, 2003
% %  p0 = coefficients of product filter of degree 4p-2.
% %  b = coefficients of binomial (spline) filter of degree 2p
% %  q = coefficients of filter of degree 2p-2 that produces the halfband
% %      filter p0 when convolved with b.
% 
% function [p0,b,q] = prodfilt(p)
% 
% % Binomial filter (1 + z^-1)^2p
% tmp1 = [1 1];
% b = 1;
% for k = 0:2*p-1
%   b = conv(b, tmp1);
% end
% 
% %  Q(z)
% tmp2 = [-1 2 -1] / 4;
% q = zeros(1,2*p-1);
% vec = zeros(1,2*p-1);
% vec(p) = 1;
% for k=0:p-1
%   q = q + vec;
%   vec = conv(vec, tmp2) * (p + k) / (k + 1);
%   vec = wkeep(vec, 2*p-1);
% end
% q = q / 2^(2*p-1);
% 
% % Halfband filter, P0(z).  
% p0 = conv(b, q);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% %
% % Polyphase filter implementation (2 channels)
% % 
% %   X = input signal, separated into even and odd phases.
% %       first row = even phase
% %	second row = odd phase	
% %   Y = output signal, separated into even and odd phases.
% %   H = 2x2 polyphase matrix
% %       H(1,1,:) = h0,even[n]
% %       H(1,2,:) = h0,odd[n]
% %       H(2,1,:) = h1,even[n]
% %       H(2,2,:) = h1,odd[n]
% %
% function Y = polyfilt(H, X)
% y0 = conv(H(1,1,:),X(1,:)) + conv(H(1,2,:),X(2,:));
% y1 = conv(H(2,1,:),X(1,:)) + conv(H(2,2,:),X(2,:));
% Y = [y0; y1];
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% %       function h = daub(Nh)
% %
% %       Generate filter coefficients for the Daubechies orthogonal wavelets.
% %
% %                       Kevin Amaratunga
% %                       9 December, 1994.
% %
% %       h = filter coefficients of Daubechies orthonormal compactly supported
% %           wavelets
% %       Nh = length of filter.
% 
% function h = daub(Nh)
% K = Nh/2;
% L = Nh/2;
% N = 512;				% Use a 512 point FFT by default.
% k = 0:N-1;
% 
% % Determine samples of the z transform of Mz (= Mz1 Mz2) on the unit circle.
% % Mz2 = z.^L .* ((1 + z.^(-1)) / 2).^(2*L);
% 
% z = exp(j*2*pi*k/N);
% tmp1 = (1 + z.^(-1)) / 2;
% tmp2 = (-z + 2 - z.^(-1)) / 4;		% sin^2(w/2)
% 
% Mz1 = zeros(1,N);
% vec = ones(1,N);
% for l = 0:K-1
% %  Mz1 = Mz1 + binomial(L+l-1,l) * tmp2.^l;
%   Mz1 = Mz1 + vec;
%   vec = vec .* tmp2 * (L + l) / (l + 1);
% end
% Mz1 = 4 * Mz1;
% 
% % Mz1 has no zeros on the unit circle, so use the complex cepstrum to find
% % its minimum phase spectral factor.
% 
% Mz1hat = log(Mz1);
% m1hat = ifft(Mz1hat);			% Real cepstrum of ifft(Mz1). (= cmplx
%                                         % cepstrum since Mz1 real, +ve.)
% m1hat(N/2+1:N) = zeros(1,N/2);		% Retain just the causal part.
% m1hat(1) = m1hat(1) / 2;		% Value at zero is shared between
%                                         % the causal and anticausal part.
% G = exp(fft(m1hat,N));			% Min phase spectral factor of Mz1.
% 
% % Mz2 has zeros on the unit circle, but its minimum phase spectral factor
% % is just tmp1.^L.
% 
% Hz = G .* tmp1.^L;
% h = real(ifft(Hz));
% h = h(1:Nh)';
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %       function r = specfac(q)
% %
% %       Find the minimum phase spectral factor of the polynomial whose
% %       coefficients are q(n).
% %
% %                    Kevin Amaratunga
% %                    2 March, 1998
% 
% function r = specfac(q)
% L = length(q);
% 
% % Check that q is admissible.
% if rem(L,2) == 0
%   warning('q[n] must have odd length.')
% end
% for k = 1:(L+1)/2
%   if q(k) ~= q(L-k+1)
%      warning('q[n] must be symmetric.')
%   end
% end
% d = (L - 1) / 2;                        % Delay that needs to be applied
%                                         % in order to make q causal.
% 
% % Compute the DFT of q.
% M = 512;                                % Size of the DFT (make this even.)
% qp = zeros(1,M);
% qp(1:L-d) = q(d+1:L);
% qp(M-d+1:M) = q(1:d);
% Q = fft(qp);
% 
% % Q must be real, assuming that q is symmetric. Check that Q is also positive.
% if nnz(Q) < M
%   warning('Q(z) has zeros on the unit circle. Cannot take the logarithm of 0.')
% end
% if nnz(real(Q) < 0) > 1
%   warning('Negative Q(w) is a bad sign.  q[n] cannot be an autocorrelation.')
% end
% 
% % Compute the logarithm.
% Qhat = log(Q);
% 
% % Compute the cepstrum.
% qhat = ifft(Qhat);
% 
% % Find the causal part.
% rhat = zeros(1,M);
% rhat(1) = qhat(1) / 2;
% rhat(2:M/2) = qhat(2:M/2);
% 
% % Determine the DFT of the causal part.
% Rhat = fft(rhat);
% 
% % Take the exponent.
% R = exp(Rhat);
% 
% % Take the inverse DFT to get the spectral factor.
% r = ifft(R);
% 
% % Pick out the right number of coefficients.
% N = (L + 1) / 2;
% if abs(r(N+1:M)) > 1e-10
%   warning('Unexpected non-zero coefficients in r[n].')
% end
% r = r(1:N);
% 
% % Finally, r may have a small imaginary part due to roundoff error,
% % so just retain the real part.
% if (imag(r)) > 1e-10
%   warning('Unexpected imaginary part in r[n].')
% end
% r = real(r);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% % function [x,phi,psi] = phivals(h,i)
% % Generate a scaling function and its associated wavelet
% % using the given filter coefficients
% %               Kevin Amaratunga
% %               5 March, 1993
% %
% %     h = filter coefficients (sum(h)=2) 
% %     i = discretization parameter.  The number of points per integer
% %         step is 2^i.  Thus, setting i = 0 gives the scaling function
% %         and wavelet values at integer points.
% %
% function [x,phi,psi] = phivals(h,i)
% if i < 0
%   error('phivals: i must be non-negative')
% end
% [m,n] = size(h);
% g = (-1).^(0:m-1)' .* h(m:-1:1);
% 
% % The Haar filter produces a singular matrix, but since we know the solution
% % already we treat this as a special case.
% if m == 2 & h == [1;1]
%   phi = [ones(2^i,1);0];
%   if i > 0
%     psi = [ones(2^(i-1),1);-ones(2^(i-1),1);0];
%   elseif i == 0
%     psi = [1;0];
%   end
% else
%   ch = [h; zeros(m,1)];
%   rh = [h(1), zeros(1,m-1)];
%   tmp = toeplitz(ch,rh);
%   M = zeros(m,m);
%   M(:) = tmp(1:2:2*m*m-1);
%   M = M - eye(m);
%   M(m,:) = ones(1,m);
%   tmp = [zeros(m-1,1); 1];
%   phi = M \ tmp;		% Integer values of phi
% 
%   if i > 0
%     for k = 0:i-1
%       p = 2^(k+1) * (m-1) + 1;	% No of rows in toeplitz matrix
%       q = 2^k * (m-1) + 1;	% No of columns toeplitz matrix
%       if (k == 0)
%         ch0 = [h; zeros(p-1-m,1)];
%         ch = [ch0; 0];
%         cg0 = [g; zeros(p-1-m,1)];
%       else
%         ch = zeros(p-1,1);
%         ch(:) = [1; zeros(2^k-1,1)] * ch0';
%         ch = [ch; 0];
%       end
%       rh = [ch(1), zeros(1,q-1)];
%       Th = toeplitz(ch,rh);
%       if k == i-1
%         cg = [1; zeros(2^k-1,1)] * cg0';
% 	cg = cg(:);
%         cg = [cg; 0];
%         rg = [cg(1), zeros(1,q-1)];
%         Tg = toeplitz(cg,rg);
%         psi = Tg * phi;
%       end
%       phi = Th * phi;
%     end
%   elseif i == 0
%     cg0 = [g; zeros(m-2,1)];
%     cg = [cg0; 0];
%     rg = [cg(1), zeros(1,m-1)];
%     Tg = toeplitz(cg,rg);
%     psi = Tg * phi;
%     psi = psi(1:2:2*m-1);
%   end
% end
% 
% [a,b] = size(phi);
% x = (0:a-1)' / 2^i;
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 








