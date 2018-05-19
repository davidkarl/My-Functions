function [x0,yhat,S] = smooth_using_smoothing_splines(x_input_vec,y_input_vec,alpha_smoothing_parameter)

% CSSPLINESMTH   Smoothing Splines
%
%   [x0,yhat,S] = cssplinesmth(x,y,alpha) returns the value of the 
%   smoothing spline using the data pairs x and y. The smoothing parameter
%   is represented by alpha. 
%
%   The output variable contains the estimates yhat(x0), where
%   x0 corresponds to the unique values of x.
%
%   The output variable S is the smoother matrix for the smoothing spline.
%   This is useful in using cross-validation.
%
%   The smoothing parameter is given by alpha. Large values of alpha
%   produce smoother curves. Small values yield curves that are more
%   wiggly. 
%
%   EXAMPLE:
%   
%   load vineyard
%   [x,ind] = sort(row);
%   y = totlugcount(ind);
%   [x0,yhat,s] = cssplinesmth(x,y,5);
%   plot(x,y,'.',x0,yhat)
%
%   See also CSLOCPOLY, CSLOCLIN, CSNARDWATS, CSLOESSR, CSLOESS, CSBINSMTH,
%   CSRMEANSMTH

%   Computational Statistics Toolbox with MATLAB, 2nd Edition
%   May 2007

if alpha_smoothing_parameter <= 0
    error('Alpha must be greater than 0.')
end

%Sort the x values:
[x_input_vec_sorted,ind] = sort(x_input_vec);
y_values_of_sorted_x_vec = y_input_vec(ind);

%Make sure they are column vectors:
x_input_vec = x_input_vec_sorted(:); 
y_input_vec = y_values_of_sorted_x_vec(:);
number_of_samples = length(x_input_vec);

%Create weight vector - starts out as ones:
w = ones(number_of_samples,1);

%The next portion of code gets rid of tied observations.
%The y values for tied observations are replaced with their average value:
h = diff(x_input_vec);
ind0 = find(h==0);
if ~isempty(ind0)
    xt = x_input_vec;
    yt = y_input_vec;
    wt = w;
    i = 1;
    while ~isempty(ind0)
        indt = find(x_input_vec(ind0(end)) == x_input_vec);
        ym(i) = mean(y_input_vec(indt));
        xm(i) = x_input_vec(indt(1));
        wm(i) = length(indt);    
        i = i+1;
        xt(indt) = [];
        yt(indt) = [];
        wt(indt) = [];
        [c,ia,ib] = intersect(indt,ind0);
        ind0(ib) = [];
    end
    xu = [xt(:); xm(:)];
    yu = [yt(:); ym(:)];
    wu = [wt(:); wm(:)];
    [xus,inds] = sort(xu);
    yus = yu(inds);
    wus = wu(inds);
    x_input_vec = xus;
    y_input_vec = yus;
    w = wus;
end
number_of_samples = length(x_input_vec);

% Find h_i = x_i+1 - x_i
h = diff(x_input_vec);
% Find 1/h_i;
hinv = 1./h;
W = diag(w);

%Use my own way of doing it with sparse matrix notation. 
%Keep the Q matrix as n by n orginally, so the subscripts match
%the G&S book. Then, I will get rid of the first and last column.
qDs = -hinv(1:number_of_samples-2) - hinv(2:number_of_samples-1);
I = [1:number_of_samples-2, 2:number_of_samples-1, 3:number_of_samples];
J = [2:number_of_samples-1,2:number_of_samples-1,2:number_of_samples-1];
S = [hinv(1:number_of_samples-2), qDs, hinv(2:number_of_samples-1)];

%Create a sparse matrix:
Q = sparse(I,J,S,number_of_samples,number_of_samples);

%Delete the first and last columns:
Q(:,number_of_samples) = []; 
Q(:,1) = [];

%Now find the R matrix:
I = 2:number_of_samples-2;  
J = I + 1;
tmp = sparse(I,J,h(I),number_of_samples,number_of_samples);
t = (h(1:number_of_samples-2) + h(2:number_of_samples-1))/3;
R = tmp' + tmp + sparse(2:number_of_samples-1,2:number_of_samples-1,t,number_of_samples,number_of_samples);

%Get rid of the rows/cols that are not needed:
R(number_of_samples,:) = []; 
R(1,:) = [];
R(:,number_of_samples) = []; 
R(:,1) = [];

%Get the smoothing spline:
S1 = Q'*y_input_vec;
S2 = R + alpha_smoothing_parameter*Q'*inv(W)*Q;

%Solve for gamma:
gam = S2\S1;

%Find f^hat:
yhat = y_input_vec - alpha_smoothing_parameter*inv(W)*Q*gam;
S = inv(W + alpha_smoothing_parameter*Q*inv(R)*Q')*W;

% S = inv(eye(n) + alpha*Q*inv(R)*Q');
x0 = x_input_vec; 







