function Z = statistics_get_bivariate_histogram(input_mat,flag_plot_option,bin_width)
% CSHIST2D  Bivariate histogram.
%
%   Z = CSHIST2D(DATA,FLAG,H)
%
%   Constructs a bivariate histogram using the observed DATA. Each row
%   of DATA corresponds to a bivariate observation. 
%   H is the smoothing parameter. This is optional. The default is the 
%   Normal Reference Rule bin width.
%   FLAG indicates what type of plot to provide: 1 = SURF,2 = Square bins
%   
%   X and Y provide the coordinates of the bins. Z is the height of the
%   histogram.
%
%   See also CSHISTDEN, CSFREQPOLY, CSASH

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

%   Revision 1/02 - Fixed bug for axes - they were set at (-3,3) for the
%   standard normal. Removed the X and Y axes limits. 

%   Revision 1/02  - A user wrote in with a problem: the surface was not plotting. The
%   data had a covariance matrix where one variance was 1400 and the other
%   was 600. This resulted in 1 bin for each direction, using the default h. When the default
%   bin width calculation yields too few bins, the user needs to input 
%   the window widths, h. Put in some code to catch this.

[number_of_samples,number_of_dimensions] = size(input_mat);
if number_of_dimensions ~= 2
    error('Must be bivariate data.')
end

if nargin == 2
    %then get the default bin width:
    input_mat_covariance = cov(input_mat);
    bin_width(1) = 3.5*input_mat_covariance(1,1)*number_of_samples^(-1/4);
    bin_width(2) = 3.5*input_mat_covariance(2,2)*number_of_samples^(-1/4);
else
    if length(bin_width)~=2
        error('Must have two bin widths in h.')
    end
end

%Need bin origins:
bin0 = [floor(min(input_mat(:,1))) , floor(min(input_mat(:,2)))]; 
%find the number of bins:
number_of_bins_x = ceil((max(input_mat(:,1))-bin0(1))/bin_width(1));
number_of_bins_y = ceil((max(input_mat(:,2))-bin0(2))/bin_width(2));

%check the number of bins. If too small, then the user should enter the h values:
if number_of_bins_x < 5 || number_of_bins_y < 5
    disp(['Number of bins in the X_1 direction is ' int2str(number_of_bins_x)])
    disp(['Number of bins in the X_2 direction is ' int2str(number_of_bins_y)])
    error('You must enter the window width h to yield 5 or more bins in each direction.')
end

%find the mesh:
t1 = bin0(1):bin_width(1):(number_of_bins_x*bin_width(1)+bin0(1));
t2 = bin0(2):bin_width(2):(number_of_bins_y*bin_width(2)+bin0(2));
[X,Y] = meshgrid(t1,t2);

%Find bin frequencies:
[number_of_mesh_rows,number_of_mesh_columns] = size(X);
vu = zeros(number_of_mesh_rows-1,number_of_mesh_columns-1);
for i = 1:(number_of_mesh_rows-1)
   for j = 1:(number_of_mesh_columns-1)
      %check which sample is inside the 2D bin square and sum them up:
      xv = [X(i,j) X(i,j+1) X(i+1,j+1) X(i+1,j)];
      yv = [Y(i,j) Y(i,j+1) Y(i+1,j+1) Y(i+1,j)];
      in = inpolygon(input_mat(:,1),input_mat(:,2),xv,yv);
      vu(i,j) = sum(in(:));
   end
end
Z = vu/(number_of_samples*bin_width(1)*bin_width(2));

    
if flag_plot_option == 1
    surf(Z)
    grid off
    axis tight
    set(gca,'YTickLabel',' ','XTickLabel',' ')
    set(gca,'YTick',0,'XTick',0)
elseif flag_plot_option == 2
    % The Z matrix is obtained in Example 5.14
    bar3(Z,1)
    % Use some Handle Graphics.
    set(gca,'YTick',0,'XTick',0)
    grid off
    axis tight
end


