function [bics,modelout,model,Z,clabs] = statistics_EM_for_finite_mixtures_model_based_search_best_model(...
                                                                Z_input_data_mat,...
                                                                max_number_of_clusters_to_check_for)

% Model-based clustering - entire process
%
%   [BICS,BESTMODEL,ALLMODELS,Z,CLABS] = MBCLUST(DATA,MAXCLUS);
%
% This does the entire MB Clustering given a set of data.
% It only does the 9 basic models, unequal-unknown priors. It
% returns the BESTMODEL based on the highest BIC. 
%
% See the HELP on MBCFINMIX for information on the model types.
%
% The output variable BICS contains the values of the BIC for
% each model (row) and number of clusters (col). The output variable 
% CLABS contains the class labels for the input data according to the 
% optimal clustering given by BIC.
%
% The output variable Z contains the cluster structure from the 
% agglomerative model-based clustering. The matrix Z can be used 
% in the DENDROGRAM function or the RECTPLOT plotting function.
% 
% The output variable ALLMODELS is a structure containing all of the models.
% ALLMODElS(I) indicates the I-th model type (1-9) and CLUS(J) indicates
% the model for J clusters.
%
% The input variable MAXCLUS denotes the maximum number of clusters to
% check for.
%

%   Model-based Clustering Toolbox, January 2003
%   Revised June 2004 to add all 9 models to this.

% model(i) indicates the i-th model type.
% clus(i) indicates that there are i clusters in the model.

warning off

[n,d] = size(Z_input_data_mat);
bics = zeros(9,max_number_of_clusters_to_check_for);     % Each row is a BIC for a model. Each col is a BIC for # clusters.

% Initialize the structure using all data points.
% This is the information for the one term/cluster model.
for i = 1:9
	model(i).clus(1).pies = 1;
	model(i).clus(1).mus = mean(Z_input_data_mat)';
	model(i).clus(1).vars = varupm(Z_input_data_mat,ones(n,1),1,mean(Z_input_data_mat)',i);
    bics(i,1) = bic(Z_input_data_mat,1,model(i).clus(1).mus,model(i).clus(1).vars,i);
end

if nargin == 3 
	disp('Getting the Adaptive Mixtures initial partition.')
	% Find an initial partition using AMDE.
	[pies,mus,vars,nterms] = amde(Z_input_data_mat,100);
	disp('Getting the agglomerative model based clustering structure')
	% Do the agglomerative model based clustering using mclust.
	Z = amdemclust(Z_input_data_mat,pies,mus,vars);
else
	disp('Getting the agglomerative model based clustering structure')
	Z = agmbclust(Z_input_data_mat);
end
	
% Based on the initialization of AMBC, get the models.

for m = 2:max_number_of_clusters_to_check_for		% Loop over the different number of clusters.
	
	% m represents the number of clusters/terms in the model.
	
	% Find the cluster labels for the number of clusters.
	labs = cluster(Z,m);
	
    % Loop over the 4 different types of models.
    for i = 1:9 
        musin = zeros(d,m);     % each column is a term.     
        piesin = zeros(1,m); 
        % Find all of the points belonging to each cluster. 
        for j = 1:m
            ind = find(labs==j);
            musin(:,j) = mean(Z_input_data_mat(ind,:))';
            piesin(j) = length(ind)/n;
            varsin(:,:,j) = varupm(Z_input_data_mat(ind,:),ones(length(ind),1),1,musin(:,j),i);
        end % j loop
        % get the finite mixture only if the previous one did not diverge
        tmp = length(model(i).clus);
        if ~isempty(model(i).clus(tmp).mus)     % then get the model
            disp(['Getting the finite mixture estimate for model ' int2str(i) ', ' int2str(m) ' clusters.'])
            [model(i).clus(m).pies,model(i).clus(m).mus,model(i).clus(m).vars] = mbcfinmix(Z_input_data_mat,musin,varsin,piesin,i);
            if ~isempty(model(i).clus(m).mus)
                bics(i,m) = bic(Z_input_data_mat,model(i).clus(m).pies,model(i).clus(m).mus,model(i).clus(m).vars,i);
            else
                bics(i,m) = 0/0;   % set it equal to a nan
            end
        else
            bics(i,m) = 0/0;
		end
        
    end  % i model type loop
        
end   % for m loop

% Once we have the BIC for each model, then get the 
% Then get the class labels according to the highest BIC.
[maxbic,maxi] = max(bics(:));
[mi,mj] = ind2sub(size(bics),maxi);

disp(['Maximum BIC is ' num2str(maxbic) '. Model number ' int2str(mi) '. Number of clusters is ' int2str(mj)])

% get the best model.
pies = model(mi).clus(mj).pies;
mus = model(mi).clus(mj).mus;
vars = model(mi).clus(mj).vars;

clabs = zeros(1,n);
for i = 1:n     
    posterior = postm(Z_input_data_mat(i,:)',pies,mus,vars);
    [v, clabs(i)] = max(posterior);     % classify it with the highest posterior prob.
end

modelout.pies = pies;
modelout.mus = mus;
modelout.vars = vars;

warning on
%%%%%%%%%%%%%%%%%%%%%%%%%5
%	INITIALIZE VARS
%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    error(['You entered ' int2str(model) ' for the model. Values must be 1, 2, 3 or 4'])
    return
end

%%%%%%%%%%%%%%%%%%%%  FUNCTION - BIC %%%%%%%%%%%%%%%%%%%%%%%%

function val = bic(data,pies,mus,vars,model)

% function val = bic(data,pies,mus,vars,model)
% This function returns the BIC criterion for 
% evaluating a finite mixture model obtained from
% the EM algorithm. This is an approximation to
% twice the Bayes factor.

% Reference: How many clusters? Which clustering method? 
% Answers via model-based cluster analysis. 
% Reference is also Celeux and Govaert, 1995, from their
% table.

[n,d] = size(data);
c = length(pies);	% number of terms.
alpha = c*d + c-1;  % number of parameters in means and pies.
beta = d*(d+1)/2;   % number of parameters in cov matrix - full.

% Now find the number of independent parameters in the model.
switch model
case 1
	m = d*c + c;
case 2
	m = 2*c + d*c-1;
case 3
    m = alpha + d;
case 4
    m = alpha + c*d - c +1;
case 5
    m = alpha + d*c;
case 6
	m = c-1 + d*c + d*(d+1)/2;
case 7
    m = alpha + c*beta - (c -1)*d;
case 8
    m = alpha + c*beta - (c-1);
case 9
	m = c-1 + d*c + c*d*(d+1)/2;
otherwise
	error('Model not recognized')
	return
end

loglike = likelihood(data,mus,vars,pies);

val = 2*loglike - m*log(n);
	
%%%%%%%%%%%%%%  FUNCTION TO EVALUATE LIKELIHOOD %%%%%%%%%%%%%%
function like = likelihood(data,mu,var_mat,mix_cof)  
% This will return the likelihood - actually the log likelihood
[n,d]=size(data);
[d,c]=size(mu);
tmplike = 0;
for i=1:c	
    % Find the value of the mixture at each data point and for each term.
    tmplike = tmplike + mix_cof(i)*evalnorm(data,mu(:,i)',var_mat(:,:,i));
end
% The likelihood is the product.
like = sum(log(tmplike));


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

%%%%%%%%%%%%FUNCTION - POSTM %%%%%%%%%%%%%%%%%%%%%%
function posterior = postm(x,pies,mus,vars)
nterms = length(pies);
totprob=0;
posterior=zeros(1,nterms);
for i=1:nterms	%loop to find total prob in denominator (hand, pg 37)
  posterior(i)=pies(i)*evalnorm(x',mus(:,i)',vars(:,:,i));
  totprob=totprob+posterior(i);
end
posterior=posterior/totprob;
