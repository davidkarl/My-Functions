function [fp,varargout] = statistics_get_frequency_polygon_density_estimate(input_data_points,window_width)
% CSFREQPOLY    Univariate or bivariate frequency polygon density estimate.
% 
%   [FP,X] = CSFREQPOLY(DATA,H)
%   Used in the 1-D case. This returns the values of X where FP is evaluated.
%
%   [FP,X,Y] = CSFREQPOLY(DATA,H)
%   Used in the 2-D case. This returns the values of X and Y where FP is
%   evaluated.
%
%   See also CSHISTDEN, CSHIST2D, CSFREQPOLY, CSKERN1D


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

[number_of_samples,number_of_dimensions] = size(input_data_points);
% just in case it is a row vector ...
if number_of_samples ==1 || number_of_dimensions ==1
    number_of_dimensions = 1;
end

% Get the stuff for the output arguments.
nout = max(nargout,1)-1;

if number_of_dimensions == 1
    % do the univariate case.
    number_of_samples = length(input_data_points); % just in case it is a row vector.
    if nargin==1
        % get the h from the normal reference rule
        window_width = 2.15*std(input_data_points)*number_of_samples^(-1/5);
    end
    t0 = min(input_data_points)-1;
    tm = max(input_data_points)+1;
    bins_vec = t0:window_width:tm;
    histogram_values = histc(input_data_points,bins_vec);
    histogram_values(end) = [];
    fhat = histogram_values/(number_of_samples*window_width);
    
    %For freq polygon, get the bin centers, with empty bin center on each end:
    bins_centers_vec = (t0-window_width/2):window_width:(tm+window_width/2);
    binh = [0 fhat 0];
    
    %Use linear interpolation between bin centers get the interpolated values at x:
    xinterp = linspace(min(bins_centers_vec),max(bins_centers_vec));
    fp = interp1(bins_centers_vec, binh, xinterp);
    
    %to plot this, use bar with the bin centers
    tm = max(bins_vec);
    bc = (t0+window_width/2):window_width:(tm-window_width/2);
    bar(bc,fhat,1,'w')
    hold on
    plot(xinterp,fp)
    hold off
    varargout{1} = xinterp;
    
elseif number_of_dimensions==2
    [number_of_samples,number_of_dimensions] = size(input_data_points);
    x = input_data_points;
    if nargin == 1
        % then get the default bin width
        covm = cov(x);
        window_width(1) = 2*covm(1,1)*number_of_samples^(-1/6);
        window_width(2) = 2*covm(2,2)*number_of_samples^(-1/6);
    else
        if length(window_width)~=2
            error('Must have two bin widths in h.')
        end
    end
    bin0 = min(x)-1;
    % do the bivariate case.
    % Find the number of bins.
    nb1 = ceil((max(x(:,1))-bin0(1))/window_width(1));
    nb2 = ceil((max(x(:,2))-bin0(2))/window_width(2));
    % Find the mesh or bin edges.
    t1 = bin0(1):window_width(1):(nb1*window_width(1)+bin0(1));
    t2 = bin0(2):window_width(2):(nb2*window_width(2)+bin0(2));
    [X,Y] = meshgrid(t1,t2);
    
    % Find bin frequencies. 
    [nr,nc]=size(X);
    vu = zeros(nr-1,nc-1);
    for i=1:(nr-1)
        for j=1:(nc-1)
            xv = [X(i,j) X(i,j+1) X(i+1,j+1) X(i+1,j)];
            yv = [Y(i,j) Y(i,j+1) Y(i+1,j+1) Y(i+1,j)];
            in = inpolygon(x(:,1),x(:,2),xv,yv);
            vu(i,j) = sum(in(:));
        end
    end
    fhat = vu/(number_of_samples*window_width(1)*window_width(2));
    % Now get the bin centers for the frequency polygon.
    % We add bins at the edges with zero height.
    t1=(bin0(1)-window_width(1)/2):window_width(1):(max(t1)+window_width(1)/2);
    t2=(bin0(2)-window_width(2)/2):window_width(2):(max(t2)+window_width(2)/2);
    [bcx,bcy]=meshgrid(t1,t2);
    [nr,nc]=size(fhat);
    binh = zeros(nr+2,nc+2);  % add zero bin heights
    binh(2:(1+nr),2:(1+nc))=fhat;
    % Get points where we want to interpolate to get
    % the frequency polygon.
    [xint,yint]=meshgrid(linspace(min(t1),max(t1),30),...
        linspace(min(t2),max(t2),30));
    fp = interp2(bcx,bcy,binh,xint,yint,'linear');
    varargout{1} = xint;
    varargout{2} = yint;
    surf(xint,yint,fp)
    axis tight
else
    % can't do here.
    error('Must be univariate or bivariate data.')
end