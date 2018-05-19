% function [speckle_size] = get_speckle_size_using_parabola_fit(mat_in,spacing,draw_flag)
% flag=0;
% counter=1;
% N=size(mat_in,1);

N=100;
spacing=1;
flag=0;
counter=1;
draw_flag=1;
speckle_size_in_pixels = 1;
[speckle_pattern]=create_speckles_of_certain_size_in_pixels(speckle_size_in_pixels,N,1);
mat_in=abs(speckle_pattern).^2;
up=3;  
mat_in=interpft(mat_in,200*up,1);
mat_in=interpft(mat_in,200*up,2);
auto_corr=corr2_ft(mat_in-mean(mat_in(:)),mat_in-mean(mat_in(:)),spacing);
auto_corr=abs(auto_corr);
auto_corr=auto_corr/max(max(auto_corr)); 
auto_corr_temp = auto_corr;
 
while flag==0  

N_auto_correlation=N*counter;
% auto_corr=interpft(auto_corr,N_auto_correlation,1);
% auto_corr=interpft(auto_corr,N_auto_correlation,2);

%get auto correlation, get its x cross-section, and lower its "DC level"
auto_corr = auto_corr(N_auto_correlation/2:N_auto_correlation/2+2,N_auto_correlation/2:N_auto_correlation/2+2);
%get fitting points for x and fit the polynomial:
fitting_points_x = log(auto_corr(2,1:3));
% fitting_points_x = auto_corr(2,1:3);
fitting_pixels_x = [-1,0,1];
Px = polyfit(fitting_pixels_x,fitting_points_x,2);

%get fitting points for y and fit the polynomial:
fitting_points_y = log(auto_corr(1:3,2)); 
% fitting_points_y = auto_corr(1:3,2);
fitting_pixels_y = [-1,0,1];
Py = polyfit(fitting_pixels_y,fitting_points_y',2);

sigma_x=sqrt(abs(1/(2*Px(1))));
sigma_y=sqrt(abs(1/(2*Py(1))));
sigma_total = sqrt(sigma_x^2+sigma_y^2)/sqrt(2);

if sigma_total>0 && sigma_total<15 %reasonable values
   speckle_size = (8/10.14)*4*sqrt(2)*sigma_total*(spacing/counter);
   flag=1;    
end                                                        
                                
counter=counter+1;     
end %end of while loop

% auto_corr_temp = interpft(auto_corr_temp,N_auto_correlation,1);
% auto_corr_temp = interpft(auto_corr_temp,N_auto_correlation,2);
if draw_flag==1
[cross_section_vec,cross_section,max_row,max_col] = get_cross_section_at_maximum(auto_corr_temp,spacing/(counter-1),1);
% [cross_section_vec_x,cross_section_x]=get_cross_section(auto_corr_temp,spacing/(counter-1),1);
% cross_section_x=cross_section_x-min(cross_section_x);
detailed_vec = linspace(-round(sigma_total*3),round(sigma_total*3),200)/(counter-1);
detailed_fitted_graph = exp((Px(1)*detailed_vec.^2+Px(2)*detailed_vec+Px(3))/(counter-1));
plot(detailed_vec,detailed_fitted_graph,'b',cross_section_vec_x,cross_section,'m--');
end
