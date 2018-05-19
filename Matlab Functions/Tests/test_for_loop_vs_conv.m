%test for loop vs. conv:
N=100;
M=3;
power_spectrum=randn(N,1);
smoother_window=randn(M,1);
logical_mask = randn(N,1);

tic
for k=1:100
   bla = conv(logical_mask,[1,1,1],'same'); 
end
toc

tic
ones_mat = ones(3,1)';
logical_mask_buffered = buffer(logical_mask,3,2);
for k=1:100
   blabla = ones_mat*logical_mask_buffered;
end
blabla = blabla';
toc

tic
ones_mat = ones(3,1);
logical_mask_buffered = buffer(logical_mask,3,2)';
for k=1:100
   blabla = logical_mask_buffered*ones_mat;
end
toc

% plot(bla);
% hold on;
% plot(blabla,'g');

% c = xcorr_fft(logical_mask,smoother_window);
% d = xcorr_fft(power_spectrum,c);
% d_matlab = xcorr(power_spectrum,c);
% e = d./c;
% 
% e_loop = zeros(N,1);
% c_loop = zeros(N,1);
% d_loop = zeros(N,1); 
% 
tic
logical_mask_buffered = buffer(logical_mask,3,2);
power_spectrum_buffered = buffer(power_spectrum,3,2);
smoother_window_buffered = repmat(smoother_window,[1,N]);
c_buffered = smoother_window_buffered.*logical_mask_buffered;
e_buffered = sum(c_buffered.*power_spectrum_buffered)./sum(c_buffered);
toc
% 
% tic
% c_fft = xcorr_fft(logical_mask,smoother_window);
% d_fft = xcorr_fft(logical_mask.*power_spectrum,smoother_window);
% e_fft = d_fft./c_fft;
% toc
% 
% tic
% for f=2:N-1
%     e_loop(f) = sum(smoother_window.*[logical_mask(f-1);logical_mask(f);logical_mask(f+1)].*[power_spectrum(f-1);power_spectrum(f);power_spectrum(f+1)]) / ...
%         sum(smoother_window.*[logical_mask(f-1);logical_mask(f);logical_mask(f+1)]);
%     
%     d_loop(f) = sum(smoother_window.*[logical_mask(f-1);logical_mask(f);logical_mask(f+1)].*[power_spectrum(f-1);power_spectrum(f);power_spectrum(f+1)]);
%     c_loop(f) = sum((smoother_window).*([logical_mask(f-1);logical_mask(f);logical_mask(f+1)]));
% end 
% toc
% 
% % close('all'); 
% figure 
% hold on;
% plot(e_loop(1:end),'b');
% plot(e_buffered(2:end),'g');
% % plot(e_fft(1:end),'r');
% 
% % hold on;
% % plot(d_loop(1:end),'b');
% % plot(d(1:end),'g');  
% % legend; 
% 
% 
