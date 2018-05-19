function yf=kalman_filter_speech_enhancement1(input_signal,strue)
% This function makes full processing of the signal x, e.g. segmentation,
% windowing, estimation and actual filtering
% Usage yf=procsig1(x)

% Set some constants first
w=256   ; % window length, should be power of two
ovrf=2  ; % overlap factor
ns=4 ;    % speech AR model order
nn=4 ;    % noise AR model order


s=w/ovrf ; % window shift
ntot=8 ; % AR model order for the input signal
fftw=w     ; % FFT window length

T=length(input_signal) ;

cnt=floor((T-w)/s) ; % how many windows
h=hanning1(w) ;       % window weights
h=w/ovrf/sum(h)*h ;  % adjust to proper scaling


input_signal=input_signal-sum(input_signal)/T ;       % remove the mean

for i=0:cnt,                  % cycle over windows
 ind1=1+i*s ; ind2=w+i*s ;    % start and end indexes
 ind1
 xw=input_signal(ind1:ind2) ;            % cut out the window
 [atot btot]=idarxb(xw,ntot) ;% estimate AR model of the signal
 stot=btot*a2s(atot,fftw) ;   % estimate the spectrum of the signal
 stot=stot(1:fftw/2) ;        % cut out the second half
 ssp=stot.^2-strue(1:fftw/2).^2 ; % calculate the noise spectrum
 ssp=sqrt(max([ssp ; zeros(1,fftw/2)])) ; % one way rectification
 %figure(2) ;
 % plot([stot(1:50)' strue(1:50)' ssp(1:50)']) ;
 csp=real(ifft([ssp ssp(fftw/2:-1:1).^2/fftw])) ;
 csp=csp(1:fftw/4) ;           % correlation function of the speech
 % figure(3) ; plot(csp) ;
 asp=c2a(csp,ns)             % AR parameters of the speech 
 [qsp rsp]=getnoise(csp,asp) ;
 bsp=sqrt(qsp)                 % b parameter of the speech
 pause ;
end ; 
 