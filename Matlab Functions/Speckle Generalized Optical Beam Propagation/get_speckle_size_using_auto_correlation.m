% function [speckle_size] = get_speckle_size_using_auto_correlation(mat_in,spacing,draw_flag)

N=200;
spacing=1;
flag=0;
counter=1;
draw_flag=1;
speckle_size_in_pixels = 1;
[speckle_pattern]=create_speckles_of_certain_size_in_pixels(speckle_size_in_pixels,N,1);
mat_in=abs(speckle_pattern).^2;
up=6;
mat_in=interpft(mat_in,200*up,1);
mat_in=interpft(mat_in,200*up,2);

%get auto correlation and normalize it to 1
auto_corr=corr2_ft(mat_in-mean(mat_in(:)),mat_in-mean(mat_in(:)),spacing);
% auto_corr=interpft(auto_corr,200*up,1);
% auto_corr=interpft(auto_corr,200*up,2);
auto_corr=abs(auto_corr);
auto_corr=auto_corr/max(max(auto_corr));

%recreat current grid
N=size(mat_in,1);
x=[-N/2:N/2-1]*spacing;
[X,Y]=meshgrid(x);

%get auto correlation, get its x cross-section, and lower its "DC level"
[cross_section_vec_x,cross_section_x]=get_cross_section_at_maximum(auto_corr,spacing,1);
% cross_section_x=cross_section_x-min(cross_section_x);

%first, plot current cross section (later add the fitted graph):
if draw_flag==1
plot(cross_section_vec_x,cross_section_x,'m--');
end

%TRY AND FIT GAUSSIAN GRAPH:
% f=fittype('A*exp(-x^2/sigma_squarred)');
f=fittype('A*exp(-(x-b)^2/sigma_squarred)');

fitting_axis = find(cross_section_x>0.1); %most of the cross-corr is noise, so i look to fit the central lobe

short_vec=cross_section_vec_x(fitting_axis);
short_section=cross_section_x(fitting_axis);

speckle_size_suggested = (length(short_vec)/2)*spacing;

[fit1,goodness_of_fit] = fit(short_vec',short_section',f,'StartPoint',[max(cross_section_x),0,speckle_size_suggested^2]);
% fitted_graph=(fit1.A)*exp(-cross_section_vec_x.^2/(fit1.sigma_squarred));
fitted_graph=(fit1.A)*exp(-(cross_section_vec_x-fit1.b).^2/(fit1.sigma_squarred));
if draw_flag==1
hold on;
plot(cross_section_vec_x,fitted_graph,'b');
end         
                                                           
speckle_size=4*sqrt(fit1.sigma_squarred)/up; %double the FWHM
1                      
close gcf;        
   

