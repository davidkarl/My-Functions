function [pies,mus,vars] = statistics_EM_for_finite_mixtures_model_based(...
                                                Z_input_data_mat,...
                                                mixture_means_mat,...
                                                mixture_variances_mat,...
                                                mixture_pies_vec,...
                                                model_type_number)

% MODEL-BASED FINITE MIXTURES USING THE EM ALGORITHM
%
% This is written for Mclust. This uses relative differences  
% between log-likelihoods to check for convergence. If the  
% estimated covariance matrices become close to singular, the function
% returns empty arrays for the model.
%
% [WTS,MUS,VARS] = MBCFINMIX(DATA,MUIN,VARIN,WTSIN,MODEL)
%
%
% INPUTS: DATA is a matrix of observations, one on each row.
%
%         The following are the initial values to start the EM algorithm:
%           MUIN is an array of means, each column corresponding to a mean.
%           VARIN is a vector of variances in the univariate case. In the
%               multivariate case, it is a 3-D array of covariance matrix, one
%               page per component density.
%           WTSIN is a vector of weights.
%
%         MODEL is one of the following models:
%           SPHERICAL FAMILY (DIAGONAL COVARIANCE, SAME VARIANCES FOR VARIABLES/DIMENSIONS):
%           1. COV are of form lambda*I       (clusters have equal covariances)
%           2: COV are of form lambda_k*I     (clusers have unequal covariances)
%
%           DIAGONAL FAMILY (DIAGONAL COVARIANCE, DIFFERENT VARIANCES FOR VARIABLES/DIMENSIONS):
%           3. COV are of form lambda*B       (clusters have equal covariances)
%           4. COV are of form lambda*B_k     (clusters have same volume, unequal shape)
%           5. COV are of form lambda_k*B_k   (clusters have unequal volume, unequal shape)
%           where B = diag(b_1,...,b_d); B is a diagonal matrix with different values and
%           det(B) = 1.
%           
%           GENERAL FAMILY (FULL COVARIANCE, OFF-DIAGONAL ELEMENTS ARE NON-ZERO)
%           6. COV are of form lambda*D*A*D'          (clusters have equal covariance)
%           7. COV are of form lambda*D_k*A*(D_k)'    (clusters have different orientation)
%           8. COV are of form lambda*D_k*A_k*(D_k)'  (clusters have different orientation and shape)
%           9. COV are of form SIGMA_k_hat            (unconstrained, all aspects vary)
%           where lambda represents the volume, D governs the orientation, and A is a diagonal matrix
%           that describes the shape.
%
%   See the paper by Celeux and Govaert for more information on the meaning and form of the
%   covariance matrices for each term or cluster. This is also discussed in the documentation.

%   Model-based Clustering Toolbox, January 2003
%   Revised May 2003 to include other models. Note that we are re-writing
%   the model numbers and categories to match the paper by Celeux and Goveart,
%   as well as our write-up.

[n,p] =  size(Z_input_data_mat);
c = length(mixture_pies_vec);

if n==1 | p==1
    % then it is univariate data.
    error('This is for multivariate data only.')
end

max_it = 100;
  
% reset the parameters to the right names.
mu = mixture_means_mat;
var_mat = mixture_variances_mat;
mix_cof = mixture_pies_vec;
num_it=1;

% get the initial value of the likelihood function.
oldlike = likelihood(Z_input_data_mat,mu,var_mat,mix_cof);

newlike = realmax;

% to store the condition numbers.
condnum = zeros(1,c);

while num_it <= max_it & ~isequal(oldlike,newlike)
    
    oldlike = newlike;
	
	
	% Check for singularity in covariances.
	for i = 1:c
		condnum(i) = rcond(var_mat(:,:,i));
	end

	if any(condnum <= 10^-10)
		pies = [];
		mus = [];
		vars = [];
		return
	end
		
   
    posterior = postupm(Z_input_data_mat,mu,var_mat,mix_cof);
    mix_coff = piupm(posterior);
    muf = muupm(Z_input_data_mat,posterior,mix_coff);
    varf = varupm(Z_input_data_mat,posterior,mix_coff,muf,model_type_number);
	
	% reset parameters
    newlike = likelihood(Z_input_data_mat,muf,varf,mix_coff);
    num_it = num_it+1;
    mix_cof = mix_coff;
    mu = muf;
    var_mat = varf;
    
end  % while loop

% assign the output variables.
pies = mix_coff;
mus = muf;
vars = varf;

%%%%%%%%%%%%%%  FUNCTION TO EVALUATE LIKELIHOOD %%%%%%%%%%%%%%
function like = likelihood(data,mu,var_mat,mix_cof)  
% This will return the likelihood
[n,d]=size(data);
[d,c]=size(mu);
tmplike = 0;
for i=1:c	
    % Find the value of the mixture at each data point and for each term.
    tmplike = tmplike + mix_cof(i)*evalnorm(data,mu(:,i)',var_mat(:,:,i));
end
% The likelihood is the product.
like = sum(log(tmplike));


%%%%%%%%%%%%%%% FUNCTION TO UPDATE VARIANCES. %%%%%%%%%%%%%%
function varm = varupm(data,posterior,mix_cof,mu,model)
[nn,c]=size(posterior);
[n,d]=size(data);
switch model
case 1
    % lambda*I
    % Spherical family.
    % diagonal equal covariance matrices
    % first find the full one.
    W_k = zeros(d,d,c);
    for i=1:c
        cen_data=data-ones(n,1)*mu(:,i)';
        mat=cen_data'*diag(posterior(:,i))*cen_data;
        W_k(:,:,i)=mat;
    end
    % common covariance is the sum of these individual ones.
    W = sum(W_k,3);
    lambda = trace(W)/(n*d);
    varmt = lambda*eye(d);
    varm = zeros(d,d,c);
    for i = 1:c
        varm(:,:,i) = varmt;
    end
case 2
    % lambda_k*I
    % Spherical family.
    % diagonal, unequal covariance matrices
    % first find the full one.
    varm = zeros(d,d,c);
    nk = mix_cof*n; % one for each term. See equation 13-15, C&G
    for i=1:c
        cen_data=data-ones(n,1)*mu(:,i)';
        Wk=cen_data'*diag(posterior(:,i))*cen_data;
        lambda = trace(Wk)/(d*nk(i));
        varm(:,:,i) = lambda*eye(d);
    end
case 3
    % lambda*B
    % Added April, 2003
    % Diagonal family. Equal covariances. Fixed volume and shape.
    % first find the full one.
    W_k = zeros(d,d,c);
    for i=1:c
        cen_data=data-ones(n,1)*mu(:,i)';
        mat=cen_data'*diag(posterior(:,i))*cen_data;
        W_k(:,:,i)=mat;
    end
    % common covariance is the sum of these individual ones.
    W = sum(W_k,3);
    dw = diag(W);
    detW = det(diag(dw))^(1/d);
    B = diag(dw)/detW;    % the matrix B is the diagonal of these
    lambda = detW/n;
    % put same covariance in each term
    varm = zeros(d,d,c);
    mt = lambda*B;
    for i = 1:c
        varm(:,:,i) = mt;
    end
case 4
    % lambda*B_k
    % Added April, 2003
    % Diagonal family. Unequal shapes, same volume.
    B_k = zeros(d,d,c);
    tmp = 0;    % to calculate the lambda
    for i=1:c
        cen_data=data-ones(n,1)*mu(:,i)';
        Wk=cen_data'*diag(posterior(:,i))*cen_data;
        dWk = diag(Wk);
        detW = det(diag(dWk))^(1/d);
        tmp = tmp + detW;
        B_k(:,:,i) = diag(dWk)/detW;
    end
    % Now get the new matrices based on each individual term.
    varm = zeros(d,d,c);
    lambda = tmp/n;
    for i = 1:c
        varm(:,:,i) = lambda*B_k(:,:,i);
    end    
case 5
    % lambda_k*B_k
    % Added April 2003
    % Diagonal family. Unequal shapes, unequal volume.
    varm = zeros(d,d,c);
    nk = mix_cof*n; % one for each term. See equation 13-15, C&G
    for i=1:c
        cen_data=data-ones(n,1)*mu(:,i)';
        Wk=cen_data'*diag(posterior(:,i))*cen_data;
        dWk = diag(Wk);
        detW = det(diag(dWk))^(1/d);
        lambdak = detW/nk(i);
        Bk = diag(dWk)/detW;
        varm(:,:,i) = Bk*lambdak;
    end
case 6
    % lambda*D*A*D'
    % Full covariance matrix, equal across terms
    % Same volume, shape, orientation
    W_k = zeros(d,d,c);
    for i=1:c
        cen_data=data-ones(n,1)*mu(:,i)';
        mat=cen_data'*diag(posterior(:,i))*cen_data;
        W_k(:,:,i)=mat;
    end
    % common covariance is the sum of these individual ones.
    varmt = sum(W_k,3);
    % put same covariance in each term
    varm = zeros(d,d,c);
    for i = 1:c
        varm(:,:,i) = varmt/n;
    end
case 7
    % lambda*D_k*A*(D_k)'
    % Added April 2003
    % Full covariance matrix.
    % same volume and shape, different orientation
    varm = zeros(d,d,c);
    omegak = zeros(d,d,c);
    dk = zeros(d,d,c);
    for i=1:c
        cen_data=data-ones(n,1)*mu(:,i)';
        Wk=cen_data'*diag(posterior(:,i))*cen_data;
        [dk(:,:,i), omegak(:,:,i)] = eig(Wk);
        % reorder so the eigenvalues are in decreasing order
        [es,index] = sort(diag(omegak(:,:,i)));
        index = flipud(index(:));  % sorts descending
        dk(:,:,i) = dk(:,index,i);
        omt = diag(omegak(:,:,i));
        omts = omt(index);
        omegak(:,:,i) = diag(omts);
    end
    A = sum(omegak,3);
    detA = det(A)^(1/d);
    A = A/detA;
    lambda = detA/n;
    for i = 1:c
        varm(:,:,i) = lambda*dk(:,:,i)*A*dk(:,:,i)';
    end
case 8
    % lambda*D_k*A_k*(D_k)'
    % Added April 2003
    % Full covariance matrix.
    % same volume, different shape and orientation.
    C_k = zeros(d,d,c);
    tmp = 0;
    for i=1:c
        cen_data=data-ones(n,1)*mu(:,i)';
        Wk=cen_data'*diag(posterior(:,i))*cen_data;
        detWk = det(Wk)^(1/d);
        C_k(:,:,i) = Wk/detWk;
        tmp = tmp + detWk;
    end
    varm = zeros(d,d,c);
    lambda = tmp/n;
    for i = 1:c
        varm(:,:,i) = lambda*C_k(:,:,i);
    end

case 9
    % lambda_k*D_k*A_k*(D_k)'
    % this is the unconstrained version
    % variable shape, volume, and orientation
    W_k = zeros(d,d,c);
    for i=1:c
        cen_data=data-ones(n,1)*mu(:,i)';
        mat=cen_data'*diag(posterior(:,i))*cen_data;
        W_k(:,:,i)=mat./(mix_cof(i)*n);
    end
    varm = W_k;
otherwise
    error(['You entered ' int2str(model) ' for the model. Values must be 1, ..., 9'])
    return
end


%%%%%%%%%%%%FUNCTION - UPDATE POSTERIORS %%%%%%%%%%%%%%%%%%%%%%
function posterior=postupm(data,mu,var_mat,mix_cof)
% Note that this returns a matrix where the ijth element corresponds to
% the ith data point, jth term.
[n,d]=size(data);
[d,c]=size(mu);
totprob = zeros(n,1);	% need one per data point, denon of eq 2.19, pg 37
posterior = zeros(n,c);
for i=1:c	%loop to find total prob in denominator (hand, pg 37)
  posterior(:,i)=mix_cof(i)*evalnorm(data,mu(:,i)',var_mat(:,:,i));
  totprob=totprob+posterior(:,i);
end
den=totprob*ones(1,c);  
posterior=posterior./den;

%%%%%%%%%%%%%%%  FUNCTION TO UPDATE MIXING COEFFICIENTS %%%%%%%%%%%%%%%
function mix_cof=piupm(posterior)
[n,c]=size(posterior);
mix_cof=sum(posterior)/n;

%%%%%%%%%%%%%  FUNCTION TO UPDATE MEANS %%%%%%%%%%%%%%%%%%%%%%%%%%  
function mu=muupm(data,posterior,mix_cof)
[n,c]=size(posterior);
[n,d]=size(data);
mu=data'*posterior;
MIX=ones(d,1)*mix_cof;
mu=mu./MIX;
mu=mu/n;

%%%%%%%%%%%%%%%  FUNCTION EVALNORM %%%%%%%%%%%%%
function prob = evalnorm(x,mu,cov_mat);
[n,d]=size(x);
prob = zeros(n,1);
a=(2*pi)^(d/2)*sqrt(det(cov_mat));
covi = inv(cov_mat);
for i = 1:n
	xc = x(i,:)-mu;
	arg=xc*covi*xc';
	prob(i)=exp((-.5)*arg);
end
prob=prob/a;

