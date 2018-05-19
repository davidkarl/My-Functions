function output = statistics_fit_spline(x_points_vec,y_points_vec,x_knots,fit_order)
%SPFIT Fit a spline to noisy data.
%   PP = SPFIT(X,Y,XB) fits a piecewise cubic spline with break
%   points XB to the noisy data (X,Y). Use PPVAL to evaluate PP.
%
%   Example:
%       x = 2*pi*(0:100)/100;
%       y = sin(x) + 0.1*randn(size(x));
%       xb = 0:6;
%       pp = spfit(x,y,xb);
%       z = ppval(pp,x);
%       plot(x,y,'.',x,z);
%
%   PP = SPFIT(X,Y,XB,N) is a generalization to piecewise polynomial 
%   functions of order N (degree N-1), with continuous derivatives
%   up to order N-2. Default is a cubic spline with N = 4.
%
%       N	Function
%       -----------------
%       1	Step function
%       2   Piecewise linear and continuous
%       3   Piecewise quadratic with continuous first derivative
%       4   Piecewice cubic with continuous second derivative (default)
%       5   Etc.

%   Author: jonas.lundgren@saabgroup.com, 2007.

      x_points_vec = 2*pi*(0:100)/100;
      y_points_vec = sin(x_points_vec) + 0.1*randn(size(x_points_vec));
      x_knots = 0:6;
      fit_order = 4;
%       pp = spfit(x_points_vec,y_points_vec,x_knots);
%       z = ppval(pp,x_points_vec);
%       plot(x_points_vec,y_points_vec,'.',x_points_vec,z);

% if nargin < 1, help spfit, return, end
% if nargin < 2, y_points_vec = 1; end
% if nargin < 3, x_knots = 0; end
% if nargin < 4, fit_order = 4; end

%Check data vectors:
x_points_vec = x_points_vec(:);
y_points_vec = y_points_vec(:);
number_of_x_points = length(x_points_vec);
if length(y_points_vec) ~= number_of_x_points
    if length(y_points_vec) == 1
        y_points_vec = y_points_vec*ones(size(x_points_vec));
    else
        error('Data vectors x and y must have the same length!')
    end
end

%Sort and check the break points:
x_knots = sort(x_knots(:));
x_knots = x_knots([diff(x_knots)>0; true]); %keep only unique knots
number_of_knots_minus_1 = length(x_knots) - 1;

if number_of_knots_minus_1 < 1
    number_of_knots_minus_1 = 1;
    x_knots = [min(x_points_vec); max(x_points_vec)];
    if x_knots(1) == x_knots(2)
        x_knots(2) = x_knots(1) + 1;
    end
end

%Adjust limits:
xlim = x_knots;
xlim(1) = -Inf;
xlim(end) = Inf;

%Generate power- and coefficient-matrices for smoothness conditions:
as = [ ones(1,fit_order); ones(fit_order-1,1)*(fit_order-1:-1:0) - (0:fit_order-2)'*ones(1,fit_order) ];
as = max(as,0);
cs = cumprod(as(1:fit_order-1,:));
ps = as(2:fit_order,:);
B0 = cs.*0.^ps;

%Smoothness conditions:
B = zeros((fit_order-1)*(number_of_knots_minus_1-1),fit_order*number_of_knots_minus_1);
h = diff(x_knots);
for k = 1:number_of_knots_minus_1-1
    Bk = cs.*h(k).^ps;
    B((fit_order-1)*(k-1)+1:(fit_order-1)*k, fit_order*(k-1)+1:fit_order*(k+1)) = [Bk, -B0];
end

%QR-factorization:
nn = min(size(B));
[Q,R] = qr(B');
Q2 = Q(:,nn+1:end);

%Weak conditions (least square sense):
A = zeros(number_of_x_points,number_of_knots_minus_1+fit_order-1);
a = zeros(number_of_x_points,1);
mm = 0;
for k = 1:number_of_knots_minus_1
    I = (x_points_vec <= xlim(k+1)) & (x_points_vec > xlim(k));
    xdata = x_points_vec(I) - x_knots(k);
    ydata = y_points_vec(I);
    d = length(xdata);
    Ak = (xdata*ones(1,fit_order)).^(ones(d,1)*(fit_order-1:-1:0));
    A(mm+1:mm+d,:) = Ak*Q2(fit_order*(k-1)+1:fit_order*k,:);
    a(mm+1:mm+d) = ydata;
    mm = mm + d;
end
 
%Solve:
c = Q2*(A\a);

%Make piecewise polynomial:
coefs = reshape(c,fit_order,number_of_knots_minus_1).';
output = mkpp(x_knots,coefs); %make piece-wise polynomial from knots and coefficients
% plot(x_points_vec,y_points_vec);
% hold on;
% plot(x_points_vec,ppval(output,x_points_vec),'g');
return

