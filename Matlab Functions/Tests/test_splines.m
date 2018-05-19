%test splines:
N=512;
speckle_size=50;
spacing=1;
spline_order=5;
noise_std = 0.2;
sparse_factor = 8;
upsampling = 4;

%initialize things:
mat = abs(create_speckles_of_certain_size_in_pixels(speckle_size,N,1,0));
mat = mat/max(abs(mat(:)));
sparse_mat = mat(sort(randperm(size(mat,1),size(mat,1)/sparse_factor)),sort(randperm(size(mat,2),size(mat,2)/sparse_factor)));
noisy_mat = mat + noise_std*randn(size(mat));
mat_vec=mat(1,:)/max(abs(mat(1,:)));
sparse_mat_x=sort(randperm(length(mat_vec),length(mat_vec)/sparse_factor));
sparse_mat_vec = mat_vec(sparse_mat_x)/max(abs(mat_vec(sparse_mat_x)));
noisy_mat_vec = mat_vec + noise_std*randn(size(mat_vec));
%redifine things:
x_sparse = sparse_mat_x;
y_sparse = sparse_mat_vec;
x = 1:length(mat_vec);
y = mat_vec;
noisy_y = noisy_mat_vec;
[N,M] = size(mat); 
x = linspace(-N/2,N/2-1,N)*spacing;
y = linspace(-M/2,M/2-1,M)*spacing;
x_new = linspace(-N/2,N/2-1,upsampling*N)*spacing;
y_new = linspace(-M/2,M/2-1,upsampling*M)*spacing;
[X,Y]=meshgrid(x,y);
[X_new,Y_new]=meshgrid(x_new,y_new);

% %show graphs (1D):
% figure(1)
% plot(mat_vec,'r'); 
% hold on;
% scatter(sparse_mat_x,sparse_mat_vec,'fill');
% hold on;
% plot(noisy_mat_vec,'g');

%show images (2D):
figure(1)
subplot(3,1,1)
imagesc(mat);
subplot(3,1,2)
imagesc(sparse_mat);
subplot(3,1,3)
imagesc(noisy_mat);

%obtain spline result using various methods (1D):
spline_result_original = spapi(5,x,y);
spline_result_noisy = spapi(5,x,noisy_y);
spline_result_sparse = spapi(5,x_sparse,y_sparse);
spline_result_lsq_original = spap2(4,5,x,y);
spline_result_lsq_sparse = spap2(4,5,x_sparse,y_sparse);
spline_result_lsq_noisy = spap2(4,5,x,noisy_y);
spline_result_lsq_sparse_better = spap2(newknt(spline_result_lsq_sparse),5,x_sparse,y_sparse);
spline_result_lsq_noisy_better = spap2(newknt(spline_result_lsq_noisy),5,x,noisy_y);
spline_result_smooth_original = spaps(x,y,10^-3);
spline_result_smooth_sparse = spaps(x_sparse,y_sparse,10^-3);
spline_result_smooth_noisy1 = spaps(x,noisy_y,10^-3);
spline_result_smooth_noisy2 = spaps(x,noisy_y,10^2);
spline_result_smoothn_original_small_s = smoothn(y,0.01);
spline_result_smoothn_original_large_s = smoothn(y,0.5);
spline_result_smoothn_sparse_small_s = smoothn(y,0.01);
spline_result_smoothn_sparse_large_s = smoothn(y,0.5);
spline_result_smoothn_noisy_small_s = smoothn(y,0.01);
spline_result_smoothn_noisy_large_s = smoothn(y,0.5);
spline_result_smoothn_noisy_small_s_robust = smoothn(y,0.01,'robust');
spline_result_smoothn_noisy_large_s_robust = smoothn(y,0.5,'robust');


%obtain spline result using various methods (2D):
spline_result_original_2D = spapi({5,5},{x,y},mat);
spline_result_noisy_2D = spapi({5,5},{x,y},noisy_mat);
spline_result_sparse_2D = spapi({5,5},{x_sparse,y_sparse},sparse_mat);
spline_result_lsq_original_2D = spap2({5,5},{5,5},{x,y},mat);
spline_result_lsq_sparse_2D = spap2({5,5},{5,5},{x_sparse,y_sparse},sparse_mat);
spline_result_lsq_noisy_2D = spap2({5,5},{5,5},{x,y},noisy_mat);
% spline_result_lsq_sparse_better_2D = spap2(newknt(spline_result_lsq_sparse_2D),{5,5},{x_sparse,y_sparse},sparse_mat);
% spline_result_lsq_noisy_better_2D = spap2(newknt(spline_result_lsq_noisy_2D),{5,5},{x,y},noisy_mat);
spline_result_smooth_original_2D = spaps({x,y},mat,10^-3);
spline_result_smooth_sparse_2D = spaps({x_sparse,y_sparse},sparse_mat,10^-3);
spline_result_smooth_noisy1_2D = spaps({x,y},noisy_mat,10^-3);
spline_result_smooth_noisy2_2D = spaps({x,y},noisy_mat,10^-1);
spline_result_smoothn_original_small_s_2D = smoothn(mat,0.01);
spline_result_smoothn_original_large_s_2D = smoothn(mat,500);
spline_result_smoothn_sparse_small_s_2D = smoothn(sparse_mat,0.01);
spline_result_smoothn_sparse_large_s_2D = smoothn(sparse_mat,500);
spline_result_smoothn_noisy_small_s_2D = smoothn(noisy_mat,0.01);
spline_result_smoothn_noisy_large_s_2D = smoothn(noisy_mat,500);
spline_result_smoothn_noisy_small_s_robust_2D = smoothn(noisy_mat,0.01,'robust');
spline_result_smoothn_noisy_large_s_robust_2D = smoothn(noisy_mat,500,'robust');


%present results (2D):
figure(2)
fnplt(spline_result_original_2D);
view(90,90);
title('spapi original');
figure(3)
fnplt(spline_result_noisy_2D);
view(90,90);
title('spapi noisy');
figure(4)
fnplt(spline_result_sparse_2D);
view(90,90);
title('spapi sparse');
figure(5)
fnplt(spline_result_lsq_original_2D);
view(90,90);
title('spap2 original');
figure(6)
fnplt(spline_result_lsq_sparse_2D);
view(90,90);
title('spap2 sparse');
figure(7)
fnplt(spline_result_lsq_noisy_2D);
view(90,90);
title('spap2 noisy');

figure(8)
fnplt(spline_result_smooth_original_2D);
view(90,90);
title('spaps original');
figure(9)
fnplt(spline_result_smooth_sparse_2D);
view(90,90);
title('spaps sparse');
figure(10)
fnplt(spline_result_smooth_noisy1_2D);
view(90,90);
title('spaps noisy tol=10^-3');
figure(11)
fnplt(spline_result_smooth_noisy2_2D);
view(90,90);
title('spaps noisy tol=10^-1');
 
figure(4)
subplot(4,2,1)
imagesc(spline_result_smoothn_original_small_s_2D);
title('smoothn original small s');
subplot(4,2,2)
imagesc(spline_result_smoothn_original_large_s_2D);
title('smoothn original large s');
subplot(4,2,3)
imagesc(spline_result_smoothn_sparse_small_s_2D);
title('smoothn sparse small s');
subplot(4,2,4)
imagesc(spline_result_smoothn_sparse_large_s_2D);
title('smoothn sparse large s');
subplot(4,2,5)
imagesc(spline_result_smoothn_noisy_small_s_2D);
title('smoothn noisy small s');
subplot(4,2,6)
imagesc(spline_result_smoothn_noisy_large_s_2D);
title('smoothn noisy large s');
subplot(4,2,7)
imagesc(spline_result_smoothn_noisy_small_s_robust_2D);
title('smoothn noisy small s, robust');
subplot(4,2,8)
imagesc(spline_result_smoothn_noisy_large_s_robust_2D);
title('smoothn noisy large s, robust');
1;




 
% %present results (1D):
% figure(2)
% subplot(3,2,1)
% fnplt(spline_result_original);
% title('spapi original');
% ylim([0,1]);
% subplot(3,2,2)
% fnplt(spline_result_noisy);
% title('spapi noisy');
% ylim([0,1]);
% subplot(3,2,3)
% fnplt(spline_result_sparse);
% title('spapi sparse');
% ylim([0,1]);
% subplot(3,2,4)
% fnplt(spline_result_lsq_original);
% title('spap2 original');
% ylim([0,1]);
% subplot(3,2,5)
% fnplt(spline_result_lsq_sparse);
% title('spap2 sparse');
% ylim([0,1]);
% subplot(3,2,6)
% fnplt(spline_result_lsq_noisy);
% title('spap2 noisy');
% ylim([0,1]);
% 
% figure(3)
% subplot(3,2,1)
% fnplt(spline_result_lsq_sparse_better);
% title('spap2 original better newknt');
% ylim([0,1]);
% subplot(3,2,2)
% fnplt(spline_result_lsq_noisy_better);
% title('spap2 noisy better newknt');
% ylim([0,1]);
% subplot(3,2,3)
% fnplt(spline_result_smooth_original);
% title('spaps original');
% ylim([0,1]);
% subplot(3,2,4)
% fnplt(spline_result_smooth_sparse);
% title('spaps sparse');
% ylim([0,1]);
% subplot(3,2,5)
% fnplt(spline_result_smooth_noisy1);
% title('spaps noisy tol=10^-3');
% ylim([0,1]);
% subplot(3,2,6)
% fnplt(spline_result_smooth_noisy2);
% title('spaps noisy tol=10^-1');
% ylim([0,1]);
%  
% figure(4)
% subplot(4,2,1)
% plot(spline_result_smoothn_original_small_s);
% subplot(4,2,2)
% plot(spline_result_smoothn_original_large_s);
% subplot(4,2,3)
% plot(spline_result_smoothn_sparse_small_s);
% subplot(4,2,4)
% plot(spline_result_smoothn_sparse_large_s);
% subplot(4,2,5)
% plot(spline_result_smoothn_noisy_small_s);
% subplot(4,2,6)
% plot(spline_result_smoothn_noisy_large_s);
% subplot(4,2,7)
% plot(spline_result_smoothn_noisy_small_s_robust);
% subplot(4,2,8)
% plot(spline_result_smoothn_noisy_large_s_robust);


  



[smooth_mat1,s1,existflag] = smoothn(mat,0.01);
[smooth_mat2,s2,existflag] = smoothn(mat,1);
[smooth_mat3,s3,existflag] = smoothn(mat,0.01,'robust');
[smooth_mat4,s4,existflag] = smoothn(mat,1,'robust');
interpolated_mat = interp2(X,Y,mat,X_new,Y_new,'spline');
figure(2)
subplot(3,2,1)
imagesc(mat);
subplot(3,2,2)
imagesc(interpolated_mat);
subplot(3,2,3);
imagesc(smooth_mat1);
subplot(3,2,4);
imagesc(smooth_mat2);
subplot(3,2,5);
imagesc(smooth_mat3);
subplot(3,2,6);
imagesc(smooth_mat4);


%interpolate derivative:


%interpolate integral:


%add noise and spline interpolate to smooth out:







