%{
I'll start with some linear regression basics. While polyfit does
a lot, a basic understanding of the process is useful.

Lets assume that you have some data in the form y = f(x) + noise.
We'll make some up and plot it.
%}

%Sort a random data stream x, and construct a function y which is linear in x:
x = sort(rand(20,1));
y = 2 + 3*x + randn(size(x));
plot(x,y,'o')
title 'A linear relationship with added noise'

%Construct the design matrix for a linear equation fit:
M = [ones(length(x),1),x];

%Find the solution of the equation y=M*x using the "regular" inverse matrix LSQ solution:
coef_ridge = inv(M'*M)*M'*y;

%Reconstruct the y estimate using the design matrix(!):
yhat = M*coef_ridge; 
plot(x,y,'o',x,yhat,'-')
title 'Linear regression model'

%Find the solution using the MORE STABLE '\' solution which uses matrix decomposition:
coef2 = M\y;

%Find the solution using the 'pinv' solution which uses SVD:
% Pinv is also an option. It too is numerically stable, but it
% will yield subtly different results when your matrix is singular
% or nearly so. Is pinv better? There are arguments for both \ and
% pinv. The difference really lies in what happens on singular or
% nearly singular matrixes. See the sidebar below.
% Pinv will not work on sparse problems, and since pinv relies on
% the singular value decomposition, it may be slower for large
% problems.
coef3 = pinv(M)*y;

%Find the solution using a sparse iterative solution using 'lsqr':
% Large-scale problems where M is sparse may sometimes benefit
% from a sparse iterative solution. An iterative solver is overkill
% on this small problem, but ...
coef4 = lsqr(M,y,1.e-13,10);

%Find the solution using the 'lscov':
% There is another option, lscov. lscov is designed to handle problems
% where the data covariance matrix is known. It can also solve a
% weighted regression problem (see section 2.)
coef5 = lscov(M,y);

%Find the solution using QR decomposition, which is closely related to the '\' solution:
%here the design matrix M is decomposed into M=Q*R such that coef = inv(R)*Q'*y:
% Why show this solution at all? Because later on, when we discuss
% confidence intervals on the parameters, this will prove useful.
[Q,R] = qr(M,0);
coef6 = R\(Q'*y);


%ROBUST estimation in the presence of outliers:
%{
In this case, some sort of trimming or iterative re-weighting scheme
may be appropriate. Iterative re-weighting simply means to compute
a regression model, then generates weights which are somehow inversely
related to the residual magnitude. Very often this relationship will
be highly nonlinear, perhaps the 5% of the points with the largest
residuals will be assigned a zero weight, the rest of the points
their normal weight. Then redo the regression model as a weighted
regression.
%}
number_of_points = 50;
number_of_outliers = 5;
x = rand(number_of_points,1);
y = 2 + 3*x + randn(size(x))/10;
y(1:number_of_outliers) = y(1:number_of_outliers) + exp(rand(number_of_outliers,1)*3);
M = [ones(number_of_points,1),x];
initial_solution = M\y;
for i=1:3
  residuals = M*initial_solution - y;
  weights = exp(-3*abs(residuals)/max(abs(residuals)))'; %arbitrary weighting function
  new_solution = lscov(M,y,weights);
  initial_solution = new_solution;
end



%RIDGE REGRESSION:
%{
Ridge regression produces a biased estimator of the parameters
compared to a simple linear regression, but biased estimators
are not always a bad thing.
%}
number_of_data_points = 10;
x = randn(number_of_data_points,2);
y = sum(x,2) + randn(number_of_data_points,1);
% the coefficients of the regression model should be [1 1]',
% since we formed y by summing the columns of x, then adding
% noise to the result.
coef_ridge = x\y;
%     coef =
%              0.752858197185749
%              0.817955777516129
% if we generated many sets of data, we would see that with few data points in each set and a 
% large noise component, our estimates of the parameters are fairly volatile.

%{
Traditional ridge regression can be viewed & accomplished
in several ways. The pure linear regression is designed to
minimize the quadratic form (A*coef - y)'*(A*coef - y).
That is, it minimizes the sum of squares of the residuals
(A*coef - y). A ridge regression simply adds a term to that
objective: lambda*sum(coef.^2). 

Clearly when lambda is large, the coefficient vector (coef) will be biased towards zero.
For small lambda, the ridge estimates will be largely unchanged from the simple linear regression.

There are two simple ways to accomplish this ridge regression.
The first is to use the normal equations. Recall that the
normal equations (to solve the system A*coef = y) are simply:
        coef = inv(A'*A)*A'*y
The associated ridge estimates for a given value of lambda
are (for p parameters to estimate, in our example, p = 2.)

 coef_ridge = inv(A'*A + lambda^2*eye(p,p))*A'*y

We can even think of the simple linear regression as a ridge
estimate with lambda = 0.

As before, we never want to actually use the normal equations.
We can use \ or any other solver to accomplish this by
augmenting the matrix A.

         coef = [A;lambda*eye(p,p)] \ [y;zeros(p,1)]

One way to interpret this expression is as if we had added
extra data points, each of which implies that one of the
parameters in the regression be zero. We can then think of
lambda as a weight for each of those new "data" points.
%}
p = 2;
lambda = 1;
coef_ridge = zeros(10,4);
for i = 1:10
  y = sum(x,2) + randn(number_of_data_points,1);
  coef_ridge(i,1:2) = (x\y)';
  coef_ridge(i,3:4) = ([x;lambda*eye(p,p)]\[y;zeros(p,1)])';
end
% The first two columns of coef are the standard regression
% estimates. Columns 3 and 4 are ridge estimates. Note that
% the ridge estimates are always biased towards zero. Had
% we used a larger value of lambda, the bias would be more
% pronounced.




% Simple ridge estimators rarely seem useful to me. The type
% of bias (always towards zero) that they produce is often
% not that helpful.
% There are other variations on the ridge estimator that can
% be quite helpful however. To show this, we will build a
% simple spline model - a piecewise constant model. Start
% out, as always, with some data.
number_of_data_points = 50;
x = sort(rand(number_of_data_points,1));
y = sin(pi*x) + randn(size(x))/20;
% Choose some knots:
knots = 0:.01:1;
% p is the number of parameters we will estimate
p = length(knots);
% Build the regression problem - the least squares spline.
% Define the spline in terms of its value in a given knot
% interval. There are 50 data points and 101 knots. We need
% to know which knot interval every point falls in. Histc
% tells us that. (Old versions of matlab which do not have
% histc can find bindex from the file exchange.)
[junk,bind] = histc(x,knots);
% Build the matrix as a sparse one! It is sparse, so use
% that fact to our advantage.
A = sparse((1:number_of_data_points)',bind,1,number_of_data_points,p);
% solve for the least squares spline. Remember that we
% had 101 knots, so 101 parameters to estimate.
spl_coef = A\y;

plot(x,y,'go',[knots(1:(end-1));knots(2:end)], ...
  repmat(spl_coef(1:(end-1))',2,1),'r-')
axis([0 1 -.2 1.2])
title 'Unregularized zero order "spline" model'
xlabel 'x'
ylabel 'y'


%USE RIDGE REGRESSION TO BIAS EACH PARAMETER TO BE CLOSE TO ITS NEIGHBORS
%BY REQUIRING THAT EACH SEQUENTIAL DIFFERENCE WILL GO TO ZERO, BUT WITH
%SMALL LAMBDA WEIGHT:
%{
% When we looked at the coefficients for this spline, we should
% have seen many coefficients that were zero. More than 1/2 of
% them were zero in fact. Where there was no data, there the
% spline was estimated as zero. There was also a warning of
% "rank deficiency". Since there were only 50 data points, but
% 101 knots (therefore 101 parameters to estimate) this was
% expected.
% We can improve this dramatically with the use of a ridge
% regression. Here I'll set up the bias so that each parameter
% is biased to be close to its neighbors in the spline.
%}
B = spdiags(ones(p-1,1)*[-1 1],[0 1],p-1,p);

% Solve. Try it with a reasonably small lambda.
lambda = 1e-3;
spl_coef_ridge = ([A;lambda*B]\[y;zeros(p-1,1)]);
% This time, no zero coefficients, and no rank problems.

plot(x,y,'go',[knots(1:(end-1));knots(2:end)], ...
  repmat(spl_coef_ridge(1:(end-1))',2,1),'r-')
title 'Zero order spline model with minimal regularization'
xlabel 'x'
ylabel 'y'

% This least squares spline, with only a tiny amount of a bias,
% looks quit reasonable. The trick is that the regularization
% term is only significant for those knots where there is no data
% at all to estimate the spline. The information for those spline
% coefficients is provided entirely by the regularizer.

% We can make lambda larger. Try varying lambda in this block
% of code on your own. You will see that lambda = 1 yields just
% a bit more smoothing. Lambda = 10 or 100 begins to seriously
% bias the result not towards zero, but to the overall mean of
% the data!

% There are other ways to do this regularization. We might also
% choose to bias the integral of the second derivative of the
% curve (for higher order splines.)





%HOW TO CHOOSE LAMBDA FOR RIDGE REGRESSION: CROSS-VALIDATION
%{
There is one more question to think about when doing any
regularized regression: What is the correct value for lambda?
Too small or too large, either is not good. In the middle is
just right. But where is Goldilocks when you need her?

An answer can sometimes be found in cross validation.
This entails repeated fits, dropping out each data point in
turn from the fit, then predicting the dropped point from the
model. The ridge parameter (lambda) which results in the lowest
overall prediction error sum of squares is the choice to use.

A nice discussion of cross validation in its many forms can
be found here:
http://www.quantlet.com/mdstat/scripts/csa/html/node123.html
I'll show an example of ordinary cross validation (OCV) in
action for the same spline fit. First, we'll plot the prediction
error sums of squares (PRESS) as a function of lambda.
%}
number_of_different_lambdas = 21;
lambda = logspace(-1,2,number_of_different_lambdas);
press = zeros(1,number_of_different_lambdas);
% loop over lambda values for the plot
for i = 1:number_of_different_lambdas
  k = 1:number_of_data_points;
  % loop over data points, dropping each out in turn
  for j = 1:number_of_data_points
    % k_j is the list of data points, less the j'th point
    k_j = setdiff(k,j);

    % fit the reduced problem without the one data point chosen to be outed:
    spl_coef = ([A(k_j,:);lambda(i)*B]\[y(k_j);zeros(p-1,1)]);

    % prediction at the point dropped out
    pred_j = A(j,:)*spl_coef;
    % accumulate press for this lambda
    press(i) = press(i) + (pred_j - y(j)).^2;
  end
end
% plot, using a log axis for x
semilogx(lambda,press,'-o')
title 'The "optimal" lambda minimizes PRESS'
xlabel 'Lambda'
ylabel 'PRESS'
% Note: there is a minimum in this function near lambda == 1,
% although it is only a slight dip. We could now use fminbnd
% to minimize PRESS(lambda).

% Now we can be lazy here, and guess from the plot that PRESS was
% minimized roughly around lambda == 2.
lambda_rougly_minimum_PRESS = 2;
spl_coef_r = ([A;lambda_rougly_minimum_PRESS*B]\[y;zeros(p-1,1)]);
% This time, no zero coefficients, and no rank problems.

plot(x,y,'go',[knots(1:(end-1));knots(2:end)], ...
  repmat(spl_coef_r(1:(end-1))',2,1),'r-')
title 'Zero order spline model with lambda == 2'
xlabel 'x'
ylabel 'y'



%PARTITIONED LEAST SQUARES- PARTITION THE OPTIMIZATION INTO INTRINSICALLY
%LINEAR AND INTRINSICALLY NONLINEAR VARIABLES:
t = sort(rand(100,1)*2-1);
y = 2 - 1*exp(-1.5*t) + randn(size(t))/10;
fun = @(coef,t) coef(1) + coef(2)*exp(coef(3)*t);
start = [1 2 -3];
options = optimset('disp','iter');
coef0 = lsqcurvefit(fun,start,t,y,[],[],options);
yhat = fun(coef0,t);

% Suppose however, I told you the value of the intrinsically
% nonlinear parameter that minimized the overall sums of squares
% of the residuals. If c is known, then all it takes is a simple
% linear regression to compute the values of a and b, GIVEN the
% "known" value of c. In essence, we would compute a and b
% conditionally on the value of c. We will let an optimizer
% choose the value of c for us, but our objective function will
% compute a and b. You should recognize that since the linear
% regression computes a sum of squares minimizer itself, that
% when the optimization has converged for the intrinsically
% nonlinear parameter that we will also have found the overall
% solution.
%
% The function pleas is a wrapper around the lsqnonlin function,
% it takes a list of functions, one for each intrinsically
% linear parameter in the model. This model is of the form
%
%   y = a*1 + b*exp(c*t)
%
% So there are two linear parameters (a,b), and one nonlinear
% parameter (c). This means there will be a pair of functions
% in funlist, the scalar constant 1, and exp(c*t)
funlist = {1, @(coef,t) exp(coef*t)};
NLPstart = -3;
options = optimset('disp','iter');
[INLP,ILP] = pleas(funlist,NLPstart,t,y,options);
fun = @(coef,t) coef(1) + coef(2)*exp(coef(3)*t);
yhat = fun([ILP;INLP],t);




%ERRORS IN THE INDEPENDENT VARIABLE - USING SVD(!!!!)
%{
Knowing your data is important in any modeling process. Where
does the error arise? The traditional regression model assumes
that the independent variable is known with no error, and that
any measurement variability lies on the dependent variable.
We might call this scenario "errors in y". Our model is of the
form

  y = f(x,theta) + noise

where theta is the parameter (or parameters) to be estimated.

In other cases, the independent variable is the one with noise
in it.

  y = g(x + noise,theta)

If this functional relationship is simply invertible, then the
best solution is to do so, writing the model in the form

  x = h(y,theta) + noise

Now one applies standard regression techniques to the inverse
form, estimating the parameter(s) theta.

Why is this important? Lets do a simple thought experiment
first. Suppose we have a simple linear model,

  y = a + b*x

With no error in the data at all, our regression estimates will
be perfect. Lets suppose there is some noise on the data points
x(i). Look at the highest point in x. The extreme points will
have the highest leverage on your estimate of the slope. If the
noise on this is positive, moving x higher, then this point will
have MORE leverage. It will also tend to bias the slope estimate
towards zero. High values of x with noise which decreases their
value will see their leverage decreased. A decrease in these
values would tend to bias the slope away from zero. But remember
that its the points with high leverage that affect the slope
the most. The same effects happen in reverse at the bottom end
of our data.

The net effect is that errors in x will tend to result in slope
estimates that are biased towards zero. Can we back this up with
an experiment?
%}
x = linspace(-1,1,201)';
y = 1 + x;
coef0 = [ones(size(x)) , x]\y;

% The errors in variables problem is also known as Total Least
% Squares. Here we wish to minimize the squared deviations of
% each point from the regression line. We'll now assume that
% both x and y have variability that we need to deal with.
x = linspace(-1,1,101)';
y = 1 + 2*x;
% add in some noise, the variance is the same for each variable.
x = x + randn(size(x))/10;
y = y + randn(size(x))/10;
% if we use the basic \ estimator, then the same errors in x
% problem as before rears its ugly head. The slope is biased
% towards zero.
coef0 = [ones(size(x)) , x]\y
coef0 =
       1.0164
       1.9653
% The trick is to use principal components. In this case we can
% do so with a singular value decomposition.
M = [x-mean(x),y-mean(y)];
[u,s,v] = svd(M,0);
% The model comes from the (right) singular vectors.
v1 = v(:,1);
disp(['(x - ',num2str(mean(x)),')*',num2str(v1(2)), ...
  ' - (y - ',num2str(mean(y)),')*',num2str(v1(1)),' = 0'])

% Only a little algebra will be needed to convince you that
% this model is indeed approximately y = 1+2*x.
(x - -0.0036897)*0.89623 - (y - 1.0092)*0.44359 = 0






%MINIMIZING THE SUM OF ABSOLUTE DEVIATIONS:
%{
Minimizing the sums of squares of errors is appropriate when
the noise in your model is normally distributed. Its not
uncommon to expect a normal error structure. But sometimes
we choose instead to minimize the sum of absolute errors.

How do we do this? Its a linear programming trick this time.
For each data point, we add a pair of unknowns called slack
variables. Thus

  y(i) = a + b*x(i) + u(i) - v(i)

Here the scalars a and b, and the vectors u and v are all unknowns.
We will constrain both u(i) and v(i) to be non-negative. Solve
the linear programming system with equality constraints as
above, and the objective will be to minimize sum(u) + sum(v).

The total number of unknowns will be 2+2*n, where n is the
number of data points in our "regression" problem.
%}
x = sort(rand(100,1));
y = 1+2*x + rand(size(x))-.5;
close
plot(x,y,'o')
title 'Linear data with noise'
xlabel 'x'
ylabel 'y'

                 
% formulate the linear programming problem.
n = length(x);
% our objective sums both u and v, ignores the regression
% coefficients themselves.
f = [0 0 ones(1,2*n)]';
% a and b are unconstrained, u and v vectors must be positive.
LB = [-inf -inf , zeros(1,2*n)];
% no upper bounds at all.
UB = [];
% Build the regression problem as EQUALITY constraints, when
% the slack variables are included in the problem.
Aeq = [ones(n,1), x, eye(n,n), -eye(n,n)];
beq = y;
% estimation using linprog
params = linprog(f,[],[],Aeq,beq,LB,UB);
% we can now drop the slack variables
coef = params(1:2);
% and plot the fit
plot(x,y,'o',x,coef(1) + coef(2)*x,'-')
title 'Linprog solves the sum of absolute deviations problem (1 norm)'
xlabel 'x'
ylabel 'y'



          
%MINIMIZING MAXIMUM ABSOLUTE DEVIATIONS:          
%{
We can take a similar approach to this problem as we did for the
sum of absolute deviations, although here we only need a pair of
slack variables to formulate this as a linear programming problem.

The slack variables will correspond to the maximally positive
deviation and the maximally negative deviation. (As long as a
constant term is present in the model, only one slack variable
is truly needed. I'll develop this for the general case.)

Suppose we want to solve the linear "least squares" problem

  M*coef = y

in a mini-max sense. We really don't care what the other errors
do as long as the maximum absolute error is minimized. So we
simply formulate the linear programming problem (for positive
scalars u and v)

  min (u+v)

  M*coef - y <= u
  M*coef - y >= -v

If the coefficient vector (coef) has length p, then there are
2+p parameters to estimate in total.
%}
% As usual, lets make up some data.
x = sort(rand(100,1));
y = pi - 3*x + rand(size(x))-.5;
close
plot(x,y,'o')
title 'Linear data with noise'
xlabel 'x'
ylabel 'y'          
          
 % Build the regression matrix for a model y = a+b*x + noise
n = length(x);
M = [ones(n,1),x];

% Our objective here is to minimize u+v
f = [0 0 1 1]';

% The slack variables have non-negativity constraints
LB = [-inf -inf 0 0];
UB = [];

% Augment the design matrix to include the slack variables,
% the result will be a set of general INEQUALITY constraints.
A = [[M,-ones(n,1),zeros(n,1)];[-M,zeros(n,1),-ones(n,1)]];
b = [y;-y];

% estimation using linprog
params = linprog(f,A,b,[],[],LB,UB);

% strip off the slack variables
coef = params(1:2)
Optimization terminated.
coef =
       3.1515
      -3.0474
% The maximum positive residual
params(3)
ans =
      0.49498
% And the most negative residual
params(4)
ans =
      0.48579
% plot the result
plot(x,y,'o',x,coef(1) + coef(2)*x,'-')
title 'Linprog solves the infinity norm (minimax) problem'
xlabel 'x'
ylabel 'y'





          
          
          
          
%{
Suppose we wanted to solve many simple nonlinear optimization
problems, all of which were related. To pick one such example,
I'll arbitrarily decide to invert a zeroth order Bessel
function at a large set of points. I'll choose to know only
that the root lies in the interval [0,4].

%}
% Solve for x(i), given that y(i) = besselj(0,x(i)).
n = 1000;
y = rand(n,1);
fun = @(x,y_i) besselj(0,x) - y_i;

% first, in a loop
tic
x = zeros(n,1);
for i=1:n
  x(i) = fzero(fun,[0 4],optimset('disp','off'),y(i));
end
toc
% as a test, compare the min and max residuals
yhat = besselj(0,x);
disp(['Min & max residuals: ',num2str([min(y-yhat),max(y-yhat)])])

% tic and toc reported that this took roughly 1.9 seconds to run on
% my computer, so effctively 0.002 seconds per sub-problem.
Elapsed time is 1.931070 seconds.
Min & max residuals: -2.7756e-16   2.498e-16
% Can we do better? Suppose we considered this as a multivariable
% optimization problem, with hundreds of unknowns. We could batch
% many small problems into one large one, solving all our problems
% simultaneously. With the optimization toolbox, this is possible.
% At least it is if we use the LargeScale solver in conjunction
% with the JacobPattern option.

% I'll use lsqnonlin because I chose to bound my solutions in the
% interval [0,4]. Fsolve does not accept bound constraints.

% define a batched objective function
batchfun = @(x,y) besselj(0,x) - y;

options = optimset('lsqnonlin');
options.Display = 'off';
options.Largescale = 'on';
options.TolX = 1.e-12;
options.TolFun = 1.e-12;

% I'll just put 50 problems at a time into each batch
batchsize = 50;
start = ones(batchsize,1);
LB = zeros(batchsize,1);
UB = repmat(4,batchsize,1);
xb = zeros(size(y));
tic
% note that this requires n to be an integer multiple of batchsize
% as I have written the loop, but that is easily modified if not.
j = 1:batchsize;
for i = 1:(n/batchsize)
  xb(j) = lsqnonlin(@(x) batchfun(x,y(j)),start,LB,UB,options);
  j = j + batchsize;
end
toc

% This took 2.2 seconds on my computer, roughly the same amount
% of time per problem as did the loop, so no gain was achieved.
Elapsed time is 1.984180 seconds.
% Why was the call to lsqnonlin so slow? Because I did not tell
% lsqnonlin to expect that the Jacobian matrix would be sparse.
% How sparse is it? Recall that each problem is really independent
% of every other problem, even though they are all related by
% a common function we are inverting. So the Jacobian matrix
% here will be a diagonal matrix. We tell matlab the sparsity
% pattern to expect with JacobPattern.

% Lets add that information and redo the computations. Only, this
% time, I'll do all of the points at once, in one single batch.
tic
batchsize = 1000;
start = ones(batchsize,1);
LB = zeros(batchsize,1);
UB = repmat(4,batchsize,1);
xb = zeros(size(y));

j = 1:batchsize;
options.JacobPattern = speye(batchsize,batchsize);
xb = lsqnonlin(@(x) batchfun(x,y),start,LB,UB,options);
toc
Elapsed time is 0.071789 seconds.
% The batched solution took only 0.092 seconds on my computer for
% all 1000 subproblems. Compare this to the 1.9 seconds it took when
% I put a loop around fzero.
%
% How did the solutions compare? They are reasonably close.
std(x - xb)

% Why is there such a significant difference in time? Some of the gain
% comes from general economies of scale. Another part of it is
% due to the efficient computation of the finite difference
% approximation for the Jacobian matrix.
ans =
   6.2945e-11
% We can test the question easily enough by formulating a problem
% with simple derivatives. I'll kill two birds with one stone by
% showing an example of a batched least squares problem for
% lsqcurvefit to solve. This example will be a simple, single
% exponential model.

% One important point: when batching subproblems together, be careful
% of the possibility of divergence of a few of the subproblems,
% since the entire system won't be done until all have converged.
% Some data sets may have poor starting values, allowing divergence
% otherwise. The use of bound constraints is a very good aid to
% avoid this bit of nastiness.

n = 100;  % n subproblems
m = 20;  % each with m data points

% some random coefficients
coef = rand(2,n);

% negative exponentials
t = rand(m,n);
y = repmat(coef(1,:),m,1).*exp(-t.*repmat(coef(2,:),m,1));
y = y+randn(size(y))/10;

% first, solve it in a loop
tic
expfitfun1 = @(coef,tdata) coef(1)*exp(-coef(2)*tdata);
options = optimset('disp','off','tolfun',1.e-12);
LB = [-inf,0];
UB = [];
start = [1 1]';
estcoef = zeros(2,n);
for i=1:n
  estcoef(:,i) = lsqcurvefit(expfitfun1,start,t(:,i), ...
    y(:,i),LB,UB,options);
end
toc
Elapsed time is 2.331607 seconds.
% next, leave in the loop, but provide the gradient.
% fun2 computes the lsqcurvefit objective and the gradient.
tic
expfitfun2 = @(coef,tdata) deal(coef(1)*exp(-coef(2)*tdata), ...
  [exp(-coef(2)*tdata), -coef(1)*tdata.*exp(-coef(2)*tdata)]);

options = optimset('disp','off','jacobian','on');
LB = [-inf,0];
UB = [];
start = [1 1]';
estcoef2 = zeros(2,n);
for i=1:n
  estcoef2(:,i) = lsqcurvefit(expfitfun2,start,t(:,i),y(:,i),LB,UB,options);
end
toc
% I saw a 33% gain in speed on my computer for this loop.
Elapsed time is 1.379798 seconds.
% A partitioned least squares solution might have sped it up too.
% Call expfitfun3 to do the partitioned least squares. lsqnonlin
% seems most appropriate for this fit.
tic
options = optimset('disp','off','tolfun',1.e-12);
LB = 0;
UB = [];
start = 1;
estcoef3 = zeros(2,n);
for i=1:n
  nlcoef = lsqnonlin('expfitfun3',start,LB,UB,options,t(:,i),y(:,i));
  % Note the trick here. I've defined expfitfun3 to return
  % a pair of arguments, but lsqnonlin was only expecting one
  % return argument. The second argument is the linear coefficient.
  % Make one last call to expfitfun3, with the final nonlinear
  % parameter to get our final estimate for the linear parameter.
  [junk,lincoef]=expfitfun3(nlcoef,t(:,i),y(:,i));
  estcoef3(:,i)=[lincoef,nlcoef];
end
toc
Elapsed time is 1.410227 seconds.
% In an attempt to beat this problem to death, there is one more
% variation to be applied. Use a batched, partitioned solution.
tic
% Build a block diagonal sparse matrix for the Jacobian pattern
jp = repmat({sparse(ones(m,1))},1,n);
options = optimset('disp','off','TolFun',1.e-13, ...
  'JacobPattern',blkdiag(jp{:}));

start = ones(n,1);
LB = zeros(n,1);
UB = [];
nlcoef = lsqnonlin('expfitfun4',start,LB,UB,options,t,y);
% one last call to provide the final linear coefficients
[junk,lincoef]=expfitfun4(nlcoef,t,y);
estcoef4 = [lincoef';nlcoef'];
toc
Elapsed time is 0.192884 seconds.          
          
          
          
          
          
          
%SIMULATED ANNEALING:
% Define an annealing schedule as a function of time, where time
% is proportional to iteration count. The annealing schedule is
% just a function that describes how the "temperature" of our
% process will be lowered with "time". Thus it may be a function
% of iteration number, as
k = 1e-2;
sched1 = @(iter) exp(-k*iter);

% or we could choose to define the new temperature at time t+1
% as a function of the old temperature...
k = 0.9905;
sched2 = @(Temp) k*Temp;

% Either choice allows us to define a rate of temperature decrease
% I'll let an interested reader decide how the two are related.

% Choose a starting value for the minimization problem. I've
% intentionally chosen a point in a poor place relative to the
% global solution.
X = 0;

% Loop for MaxIter number of steps, remembering all the steps we
% have taken along the way, and plot the results later.
MaxIter = 10000;
X_T = zeros(MaxIter,2);
fold = f(X);
X_T(1,:) = [X,fold];
std0 = 3;
for iter = 2:MaxIter
  % current temperature (using the exponential form)
  T = sched1(iter);

  % Temperature dependent perturbation
  Xpert = std0*randn(1)*sqrt(T);
  Xnew = X + Xpert;

  fnew = f(Xnew);

  % Do we accept it? Always move to the new location if it
  % results in an improvement in the objective function.
  if fnew < fold
    % automatically accepted step
    X = Xnew;
    fold = fnew;
    X_T(iter,:) = [X,fnew];
  elseif rand(1) <= (T^2)
    % Also accept the step with probability based on the
    % temperature of the system at the current time. This
    % helps us to tunnel out of local minima.
    X = Xnew;
    fold = fnew;
    acceptflag = false;
    X_T(iter,:) = [X,fnew];
  else
    X_T(iter,:) = [X,fold];
  end
  % if we dropped through the last two tests, then we will
  % go on to the next iteration.
end
[fmin,minind] = min(X_T(:,2));

% all is done, so plot the results
x1 = min(-50,min(X(:,1)));
x2 = max(50,max(X(:,1)));
ezplot(f,[x1,x2])
hold on
plot(X_T(:,1),X_T(:,2),'r-')
plot(X_T(minind,1),fmin,'go')
hold off
          
          
          
          
          

%SOLVING A CONSTRAINED LINEAR PROBLEM:
% The real question in this section is not how to solve a linearly
% constrained problem, but how to solve it programmatically, and
% how to solve it for possibly multiple constraints.

% Start with a completely random least squares problem.
n = 20; % number of data points
p = 7;  % number of parameters to estimate
A = rand(n,p);
% Even generate random coefficients for our ground truth.
coef0 = 1 + randn(p,1);
y = A*coef0 + randn(n,1);

% Finally, choose some totally random constraints
m = 3;  % The number of constraints in the model
C = randn(m,p);
D = C*coef0;
         



% How does one solve the constrained problem? There are at least
% two ways to do so (if we choose not to resort to lsqlin.) For
% those devotees of pinv and the singular value distribution,
% one such approach would involve a splitting of the solution to
% A*x = y into two components: x = x_u + x_c. Here x_c must lie
% in the row space of the matrix C, while x_u lies in its null
% space. The only flaw with this approach is it will fail for
% sparse constraint matrices, since it would rely on the singular
% value decomposition.
%
% I'll discuss an approach that is based on the qr factorization
% of our constraint matrix C. It is also nicely numerically stable,
% and it offers the potential for use on large sparse constraint
% matrices.

[Q,R,E]= qr(C,0);

% First, we will ignore the case where C is rank deficient (high
% quality numerical code would not ignore that case, and the QR
% allows us to identify and deal with that event. It is merely a
% distraction in this discussion however.)
%
% We transform the constraint system C*x = D by left multiplying
% by the inverse of Q, i.e., its transpose. Thus, with the pivoting
% applied to x, the constraints become
%
%  R*x(E) = Q'*D
%
% In effect, we wanted to compute the Q-less QR factorization,
% with pivoting.
%
% Why did we need pivoting? As I suggested above, numerical
% instabilities may result otherwise.
%
% We will reduce the constraints further by splitting it into
% two fragments. Assuming that C had fewer rows than columns,
% then R can be broken into two pieces:
%
%  R = [R_c, R_u]

R_c = R(:,1:m);
R_u = R(:,(m+1):end);

% Here R_c is an mxm, upper triangular matrix, with non-zero
% diagonals. The non-zero diagonals are ensured by the use of
% pivoting. In effect, column pivoting provides the means by
% which we choose those variables to eliminate from the regression
% model.
%
% The pivoting operation has effectively split x into two pieces
% x_c and x_u. The variables x_c will correspond to the first m
% pivots identified in the vector E.
%
% This split can be mirrored by breaking the matrices into pieces
%
%  R_c*x_c + R_u*X_u = Q'*D
%
% We will use this version of our constraint system to eliminate
% the variables x_c from the least squares problem. Break A into
% pieces also, mirroring the qr pivoting:

A_c = A(:,E(1:m));
A_u = A(:,E((m+1):end));


% So the least squares problem, split in terms of the variable
% as we have reordered them is:
%
%  A_c*x_c + A_u*x_u = y
%
% We can now eliminate the appropriate variables from the linear
% least squares.
%
%  A_c*inv(R_c)*(Q'*D - R_u*x_u) + A_u*x_u = y
%
% Expand and combine terms. Remember, we will not use inv()
% in the actual code, but instead use \. The \ operator, when
% applied to an upper triangular matrix, is very efficient
% compared to inv().
%
%  (A_u - A_c*R_c\R_u) * x_u = y - A-c*R_c\(Q'*D)

x_u = (A_u - A_c*(R_c\R_u)) \ (y - A_c*(R_c\(Q'*D)));

% Finally, we recover x_c from the constraint equations
x_c = R_c\(Q'*D - R_u*x_u);


%PUTTING IT ALL TOGETHER:
n = 20; % number of data points
p = 7;  % number of parameters to estimate
A = rand(n,p);
% Even generate random coefficients for our ground truth.
coef0 = 1 + randn(p,1);
y = A*coef0 + randn(n,1);
m = 3;  % The number of constraints in the model
C = randn(m,p);
D = C*coef0;
[Q,R,E]= qr(C,0);
A_c = A(:,E(1:m));
A_u = A(:,E((m+1):end));
x_u = (A_u - A_c*(R_c\R_u)) \ (y - A_c*(R_c\(Q'*D)));
x_c = R_c\(Q'*D - R_u*x_u);









%{
There are various things that people think of when the phrase
"confidence limits" arises.

- We can ask for confidence limits on the regression parameters
themselves. This is useful to decide if a model term is statistically
significant, if not, we may choose to drop it from a model.

- We can ask for confidence limits on the model predictions (yhat)

These goals are related of course. They are obtained from the
parameter covariance matrix from the regression. (The reference to
look at here is again Draper and Smith.)

If our regression problem is to solve for x, such that A*x = y,
then we can compute the covariance matrix of the parameter vector
x by the simple

  V_x = inv(A'*A)*s2

where s2 is the error variance. Typically the error variance is
unknown, so we would use a measure of it from the residuals.

  s2 = sum((y-yhat).^2)/(n-p);

Here n is the number of data points, and p the number of parameters
to be estimated. This presumes little or no lack of fit in the model.
Of course, significant lack of fit would invalidate any confidence
limits of this form.

Often only the diagonal of the covariance matrix is used. This
provides simple variance estimates for each parameter, assuming
independence between the parameters. Large (in absolute value)
off-diagonal terms in the covariance matrix will indicate highly
correlated parameters.

In the event that only the diagonal of the covariance matrix is
required, we can compute it without an explicit inverse of A'*A,
and in way that is both computationally efficient and as well
conditioned as possible. Thus if one solves for the solution to
A*x=y using a qr factorization as

  x = R\(Q*y)

then recognize that

  inv(A'*A) = inv(R'*R) = inv(R)*inv(R') = inv(R)*inv(R)'

If we have already computed R, this will be more stable
numerically. If A is sparse, the savings will be more dramatic.
There is one more step to take however. Since we really want
only the diagonal of this matrix, we can get it as:

  diag(inv(A'*A)) = sum(inv(R).^2,2)
%}
% Compare the two approaches on a random matrix.
A=rand(10,3);

diag(inv(A'*A))

[Q,R]=qr(A,0);
sum(inv(R).^2,2)

% Note that both gave the same result.











          
          








