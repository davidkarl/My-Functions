%test auto correlation on gaussians:
% a=abs(create_speckles_of_certain_size_in_pixels(50,512,1,0)).^2;
% filter_order=512;
% filter_size=512;
% filter_parameter=20;
% low_cutoff = 10;
% high_cutoff = 30; 
% high_pass=get_filter_2D_intuitive('kaiser','high',filter_size,filter_parameter,low_cutoff,high_cutoff); 
% high_pass=get_filter_2D_quick('high',filter_size,filter_order,low_cutoff,high_cutoff);
% filtered_frame = filter2(high_pass,a);
% filtered_frame = convolution_ft(a,high_pass,1);
% filtered_frame = get_mat_without_DC(a);
% matrix_mean = mean(a(:));
% subplot(2,2,1); 
% imagesc(a);
% colorbar; 
% subplot(2,2,2)
% mesh(abs(ft2(high_pass,1))); 
% subplot(2,2,3)  
% imagesc(filtered_frame);    
% colorbar;   
  
N=20;
view_size = 10;
x0=0;
y0=0;
sigma_x=10;
sigma_y=10;
rho=0;
background=15;
amplitude=5;
[gaussian_2D_centered,X,Y] = create_2D_gaussian(512,view_size,sigma_x,sigma_y,x0,y0,rho,background,amplitude);
gaussian_without_DC = get_mat_without_DC(gaussian_2D_centered);
gaussian_high_passed = filter_mat_highpass(gaussian_2D_centered,5);
gaussian_2D_shifted = create_2D_gaussian(512,512,1,1,100,100,0,10,5);
gaussin_2D_centered_derived = calculate_derivative_using_spline(gaussian_2D_centered,X(1,2)-X(1,1),4,[1,0]);
spacing = X(1,2)-X(1,1);
% cross_correlation_mat = cross_correlation_ft(gaussian_without_DC,gaussian_without_DC,1,1);
% cross_correlation_mat = cross_correlation_dan(gaussian_without_DC,gaussian_without_DC,3);
% bla=fit_polynom_surface(X(:),Y(:),gaussian_without_DC(:),6);
figure;
subplot(3,2,1) 
imagesc(gaussian_2D_centered); 
title('original');
colorbar
subplot(3,2,2)
imagesc(gaussian_without_DC);
title('without DC');
colorbar;  
subplot(3,2,3)
imagesc(log(gaussian_2D_centered));
title('log of gaussian');
colorbar;
subplot(3,2,4) 
imagesc(exp(gaussian_without_DC)); 
title('exponential of gaussian without DC');
colorbar;
subplot(3,2,5)
imagesc(get_mat_without_DC(log(gaussian_2D_centered)));
title('log of gaussian - then without DC');
colorbar; 

ft=fittype('A+B*log(abs(x))-C*(x)^2');
y = log(abs(gaussin_2D_centered_derived(256,end/2+1:end)))';
x = X(1,end/2+1:end)';
bla=fit(x,y,ft,'StartPoint',[0,0,0]);
subplot(3,2,6)
% imagesc(exp(get_mat_without_DC(log(gaussian_2D_centered))));
% title('exp(log of gaussian - then without DC)');
plot(x,y);
title('log(derivative of spline fitted original gaussian surface)');
colorbar;

1;

 