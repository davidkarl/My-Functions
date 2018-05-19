%TEST PR TOOLS!!!:

% % EXAMPLE
% x = gendatb(100);
% w1 = x*ldc;                  % standard classifier
% C = [0 2; 1 0];              % new cost matrix
% w2 = w1*classc*costm(C);  % classifier using this cost-matrix
% confmat(x*w1)
% confmat(x*w2)

% % EXAMPLE
% a_dataset = gendatb;                     % generate banana classes
% [w_map,J_support_vector_indices] = a_dataset*libsvc(proxm('p',3));  % compute svm with 3rd order polynomial
% a_dataset*w_map*testc                        % show error on train set
% scatterd(a_dataset)                      % show scatterplot
% plotc(w_map)                         % plot classifier
% hold on; 
% scatterd(a_dataset(J_support_vector_indices,:),'o')             % show support objcts


% % EXAMPLE
% a = gendatb([100 100]);    % training set of 200 objects
% r = gendatb([10 10]);      % representation set of 20 objects
% w = fdsc(a,r);             % compute classifier, SVM based! - when rewriting the name know this
% a*w*testc
% scatterd(a);               % scatterplot of trainingset
% hold on; scatterd(r,'ko'); % add representation set to scatterplot
% plotc(w);                  % plot classifier


% % EXAMPLES
% B = A*filtm('histc',[1:256]);


% % EXAMPLE
% A = delft_images; %cant find delft_images
% B = A([120 121 131 230])*doublem*col2gray; %choose 4 objects from a, convert to double and to greyscale
%                                            %so size is 1/3 the original
%                                            %size
% E = B*filtim('fft2')*filtim('abs')*filtim('fftshift');
% figure; 
% show(E); 
% figure;
% show((1+E)*filtim('log'));


% % EXAMPLES
% % Numeric labels, 1..10, 10 classes, 100 labels per class (didn't specify
% % classes so they're just ordinarily specified as 1,2,3,....10).
% labels1 = genlab(100*ones(1,10));
% % Character labels, 3 classes, 10 labels per class.
% labels2 = genlab([10 10 10], ['A';'B';'C']); 
% % Name labels, 2 classes, 50 labels per class.
% % The commands below are equivalent.
% labels3 = genlab([50 50], {'Apple'; 'Pear'});
% labels3 = genlab([50 50], ['Apple'; 'Pear ']);



% % EXAMPLES
% % %IN THE CLASS PARAMETERS PARAMETER:
% % (1).a column vector [1;2;3...] specifies different classes parameters, 
% % (2).a row vector [1,2,3...] specifies a single channel's multi-dimensional parameters
% % %IN THE NUMBER OF OBJECTS PARAMETER:
% % (1). a column vector specifies the number of objects from each class
% % 1. Generation of 100 points in 2D with mean [1 1] and default covariance
% %    matrix:
% a = gendatgauss(100,[1,1]);
% %
% % 2. Generation of 50 points for each of two 1-dimensional distributions with
% %    mean -1 and 1 and with variances 1 and 2:
% a = gendatgauss([50 50],[-1;1],cat(3,1,2));
% %
% %   Note that the two 1-dimensional class means should be given as a column
% %   vector [1;-1], as [1 -1] defines a single 2-dimensional mean. Note that
% %   the 1-dimensional covariance matrices degenerate to scalar variances,
% %   but have still to be combined into a collection of square matrices using
% %   the CAT(3,....) function.
% % 3. Generation of 300 points for 3 classes with means [0 0], [0 1] and
% %    [1 1] and covariance matrices [2 1; 1 4], EYE(2) and EYE(2):
% a = gendatgauss(300,[0 0; 0 1; 1 1]*3,cat(3,[2 1; 1 4],eye(2),eye(2)));


% % EXAMPLE
% a = gendatb;
% w = a*gaussm(2); %mixture of gaussians
% scatterd(a)
% plotm(w) %plots equiv-value lines as gaussm() returns gaussian mixture densities effectively
%          %so we use plotm as apposed to plotc!!!


% % EXAMPLE
% a = gendats([25 25],20,5);    % 50 points in 20 dimensional feature space
% d = sqrt(distm(a));           % Euclidean distances
% dendg = hclust(d,'complete'); % dendrogram
% plotdg(dendg)
% lab = hclust(d,'complete',2); % labels
% confmat(lab,getlabels(a));    % confusion matrix
   

% % EXAMPLE
% % This example assumes that the Kimia images are available as datafile
% % and that the DipImage image processing package is available.
% prwaitbar on
% a = kimia_images;
% a = kimia;
% x = im_moments(a,'hu');
% x = setname(x,'Hu moments');
% y = im_measure(a,a,{'size','perimeter','ccbendingenergy'});
% y = setname(y,'Shape features');
% [R,T,L] = im_dbr(a,{x,y});  % do your own search , no im_dbr.fig!!!@!#
% delfigs
% figure(1); show(a(R,:)); % show ranking
% figure(2); show(a(T,:)); % show targets
% figure(3); show(a(L,:)); % show outliers
% showfigs


% % EXAMPLE
% a = delft_idb; 
% a = seldat(a,9);        
% delfigs
% mask = a*im_gray*im_threshold;         
% figure, 
% show(mask)
% seed = mask*im_berosion;               
% figure, 
% show(seed)
% cleaned = im_bpropagation(seed,mask);
% figure, 
% show(cleaned)
% showfigs

 

% % EXAMPLE
% prdatafiles;            % make sure prdatafiles is in the path
% x = kimia;       % load kimia images
% x = x*im_box(0);        % remove all empty rows and columns
% x = x*im_box(0,1);      % add rows/columns to make images square
% x = x*im_resize([32 32]); % resample them 
% x = x*im_box(1,0);      % add rows/columns and keep image square
% % now all images are 34x34 and no object touches the border.
% show(gendat(x,4))       % show a few. gendat randomly picks a number of objects! 


 
% % EXAMPLE
% delfigs
% a = gendatb;
% scatterd(a)
% disp('Draw a polygon in the scatterplot')
% h = impoly;  % use the mouse to draw a polygon in the scatterplot
% [jin,jout] = inpoly(a,h);
% hold on; 
% scatterd(a(jin,:),'ko'); % encircle selected objects
% figure;  
% scatterd(a(jout,:));     % show objects outside polygon
% showfigs



% % EXAMPLE
% a = gendatb;
% w = (scalem*kernelm([],'random',5)*fisherc); 
% scatterd(a)
% plotc(a*w) %why do i get different results each row? isn't the only thing different the color
% plotc(a*w,'r')
% plotc(a*w,'b')



% % EXAMPLE
% a = gendatb([100 100]);    % training set of 200 objects
% r = gendatb([7 7]);      % representation set of 20 objects
% v = proxm(r,'p',3);        % compute kernel
% v2 = proxm(a,'p',3);
% bla = a*v; %gives a 200X20 matrix (prdataset) including the distances of a objects from the 20 r objects
% bla2 = a*v2;
% w = kernelc(a,v,fisherc)   % compute classifier
% w2 = kernelc(a,v2,fisherc)
% scatterd(a);               % scatterplot of trainingset
% hold on; 
% scatterd(r,'ko'); % add representation set to scatterplot
% plotc(w);                  % plot classifier
% hold on;
% plotc(w2,'r'); %gives the same result as w2



% % EXAMPLE
% a = gendatb;                     % generate banana classes
% [w,J] = a*libsvc(proxm('p',3));  % compute svm with 3rd order polynomial
% a*w*testc                        % show error on train set
% scatterd(a)                      % show scatterplot
% plotc(w)                         % plot classifier
% hold on; 
% scatterd(a(J,:),'o')             % show support objcts



% % EXAMPLES
% a = gendatd;  % generate Gaussian distributed data in two classes
% w = ldc(a);   % compute a linear classifier between the classes - LINEAR CLASSIFIER IS A STRAIGHT LINE!!!
% scatterd(a);  % make a scatterplot
% plotc(w)      % plot the classifier



% %   Example: train the 3-NN classifier on the generated data.
% W = knnc([],3);         % untrained classifier
% V = gendatd([50 50])*W; % trained classifier
% %EXAMPLE:
% A = gendatd([50 50],10);	% generate random 10D datasets
% B = gendatd([50 50],10);
% W = klm([],0.9);          % untrained mapping, Karhunen-Loeve projection
% V = A*W;                  % trained mapping V
% D = B*V;                  % the result of the projection of B onto V
% %   Example: normalize the distances of all objects in A such that their
% %   city block distances to the origin are one.
% %
% A = gendatb([50 50]);
% W = normm;
% D = A*W;
% %EXAMPLE:
% A = gendatd([50 50],10); % generate random 10D datasets
% B = gendatd([50 50],10);
% V = klm([],0.9);         % untrained Karhunen-Loeve (KL) projection
% W = ldc;                 % untrained linear classifier LDC
% U = V*W;                 % untrained combiner
% T = A*U;                 % trained combiner
% D = B*T;                 % apply the combiner (first KL projection,
                             %       then LDC) to B



% % EXAMPLES
% argmin = mapm('min')*out2; 
% % A*argmin returns for every column the row indices of the minimum
% % A'*argmin returns for every row the column indices of its minimum
% nexpm = mapm('uminus')*mapm('exp');
% % A*nexpm returns exp(-A), useful incombination with other commands


% % EXAMPLE
% mink1 = proxm('m',1)*mapex
% mink1 = mapex(proxm,'m',1)    % the same
% mink1 = mapex('proxm','m',1)  % the same
% % Herewith D = A*mink1 computes a dissimilarity matrix based on the
% % Minkowsky_1 metric between all objects in A.


% % EXAMPLES:
% a = gendatb;
% opt.optim = 'scg'; %scaled conjugate gradients (scg)
% opt.init  = 'cs';  %classical scaling
% D  = sqrt(distm(a)); % Compute the Euclidean distance dataset of A
% w1 = mds(D,2,opt);   % An MDS map onto 2D initialized by Classical Scaling,
%                      % optimised by a Scaled Conjugate Gradients algorithm
% n  = size(D,1);
% y  = rand(n,2);
% w2 = mds(D,y,opt);   % An MDS map onto 2D initialized by random vectors
% z = rand(n,n);       % Set around 40% of the random distances to NaN, i.e. 
% z = (z+z')/2;        % not used in the MDS mapping
% z = find(z <= 0.6);
% D(z) = NaN;
% D(1:n+1:n^2) = 0;    % Set the diagonal to zero ????????
% opt.optim = 'pn';
% opt.init  = 'randnp'; 
% opt.etol  = 1e-8;    % Should be high, as only some distances are used
% w3 = mds(D,2,opt);   % An MDS map onto 2D initialized by a random projection



% % As NMSC is a linear classifier, a non-linear combiner might give an
% % improvement in multi-dimensional problems, e.g. by 
% a = gendatb;
% w = a*(nmc*qdc([],[],1e-6)); %nmc = nearest mean classifier, qdc = quadratic classifier
%                              %what is the 1e-6 figure????
% % EXAMPLES
% [u,g] = meancov(gendatb(25));
% w = nbayesc(u,g); %w has a "det" parameter, what is it??????
% % EXAMPLES
% % 	Train set A and test set T:
% t = a;
% bla = nmc(a); %nmc seems to give an offset+rotation, seems weird. this basically trains the nmc with a
% b = t*nmc(a); %this gives class confidences, which are based on assumed spherical gaussian distributions
%               %of the same size?!?!?!
% e = prroc(t,50); %wrong statement?.....
% plote(e); % Plots a single curve
% e = prroc(t,a*{nmc,udc,qdc});  %plots error2 vs. error1!!!!
% plote(e); % Plots 3 curves



% % EXAMPLES
% a = gendatb([50,50]);
% b = gendatb([20,20]);
% W = a*proxm('m',1);              % define L1 distances
% W = a*proxm('m',1); D = b*W;     % L1 distances between B and A
% W = proxm('r',2)*mapex; D = a*W; % Distances between A and itself


% % EXAMPLES
% a = gendatb;
% figure; 
% scatterd(a); 
% plotc(a*rfishercc)


% % EXAMPLE:
% a = gendatm; 
% figure; scatterd(a); 
% plotc(a*rfishercc(2),'col')


% % EXAMPLE:
% prload sonar
% % a = setprior(sonar,0); % make priors equal
% a = setprior(a,0); %make priors equal
% w1 = setname(rfishercc(10),'RFisher-10');
% w2 = setname(rfishercc(20),'RFisher-20');
% w3 = setname(rfishercc(40),'RFisher-40');
% randreset(1); % for reproducability ?????????????????
% e = cleval(a,{w1,w2,w3},[5,10,20,40,80],10);
% plote(e);


% % EXAMPLES
%  a = gendatb;              % create trainingset
%  w = ldc(a);               % create supervised classifier
%  wr = rejectm(a*w,0.05);   % reject 5% of the data
%  scatterd(a); 
%  plotc(w*wr); % show, notice how we write w*wr to reject 5% of the classified data to create a new map


% % EXAMPLE
% a = gendatb;
% w = ldc(a);
% v = rejectc(a,w,0.2);
% scatterd(a);
% plotc(w);
% plotc(v,'r')


% % EXAMPLE
% A = gendatd([30 30],50);
% W = ldc(A,0,NaN); % set first reg par to 0 and optimise second.
% getopt_pars       % retrieve optimal paameter set



% % EXAMPLE
% % compute a dissimilarity based classifier for a representation set of
% % 10 objects using a Minkowski-1 distance.
% a = gendatb;
% u = selproto(10)*proxm('m',1)*fisherc;
% u2 = selproto(10)*fisherc;
% w = a*u;
% w2 = a*u2;
% scatterd(a)
% plotc(w)
% plotc(w2,'r') %WE CAN SEE THAT W, THE DISSIMILARITY BASED CLASSIFIER IS MUCH(!) BETTER



% % EXAMPLES
% % Generate 8 class, 2-D dataset and select: the second feature, objects
% % 1 from class 1, 0 from class 2 and 1:3 from class 6
% a = gendatm([3,3,3,3,3,3,3,3]);
% a = seldat(A,[1 2 6],2,{1;[];1:3});
% % or
% a = seldat(A,[],2,{1;[];[];[];[];1:3});



% % EXAMPLES
% a = gendatm; 
% b = selclass(a,[2 3 4]); % selects 3 classes
% a = gendatm; 
% b = a*selclass;         % returns every class in a cell(!!!!!!!!!!)
% a = gendatb; 
% a = addlabels(a,genlab(25*ones(4,1)),'4class'); % add second label list
% b = a*selclass('4class'); % returns 4 cells, preserves label list.



% % EXAMPLE
% prdatasets;            % make sure prdatasets is in the path
% a = satellite;         % 36D dataset, 6 classes, 6435 objects
% [x,y] = gendat(a,0.5); % split in train and test set
% w = x*sammonm;         % compute mapping
% figure; scattern(x*w); % show trainset mapped to 2D: somewhat overtrained
% figure; scattern((x+randn(size(x))*1e-5)*w); % some noise helps
% figure; scattern(y*w); % show test set mapped to 2D. scattern is a simple 2D scatter plot!!!!!!




% % EXAMPLE
% prdatasets;            % make sure prdatasets is in the path
% a = satellite;         % 36D dataset, 6 classes, 6435 objects
% a = gendat(a,0.3);     % take a subset to make it faster
% [x,y] = gendat(a,0.5); % split in train and test set
% w = x*tsnem;           % compute mapping
% %   W = TSNEM(A,K,N,P,MAX)
% %   W = A*TSNEM([],K,N,P,MAX)
% %   W = A*TSNEM(K,N,P,MAX)
% %   D = B*W
% %
% % INPUT
% %   A    Dataset or matrix of doubles, used for training the mapping
% %   B    Dataset, same dimensionality as A, to be mapped
% %   K    Target dimension of mapping (default 2)
% %   N    Initial dimension (default 30)
% %   P    Perplexity (default 30)
% %   MAX  Maximum number of iterations, default 1000
% %
% % OUTPUT
% %   W    Trained mapping
% %   D    2D dataset
% %
% % DESCRIPTION
% % This is PRTools inteface to the t-SNE Simple Matlab routine for high
% % dimensional data visualisation. The output is a non-linear projection of
% % the original vector space to a K-dimensional target space. The procedure
% % starts with a preprocessing to N dimensions by PCA. The perplexity
% % determines the number of neighbors taken into account, see references.
% %
% figure; scattern(x*w); % show train set mapped to 2D: looks overtrained
% figure; scattern((x+randn(size(x))*1e-5)*w); % some noise helps, LOOK HOW LITTLE NOISE IT TAKES
%                                              %...UNDERSTAND!!!!
% figure; scattern(y*w); % show test set mapped to 2D
% figure; scattern(y*pca(x,2)); % compare with pca

 


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%Kimia_simple:
% % Section 1 Check availablity of toolboxes
% % <http://37steps.com/software/prtools/ PRTools> and <http://37steps.com/software/prdatafiles/ PRDataFiles> should be in the path
% if exist('ldc',         'file') ~= 2, 
%     error('PRTools not in path'); 
% end
% if exist('kimia_images','file') ~= 2, 
%     error('PRDataFiles not in path'); 
% end
% delfigs; % delete all figures to avoid clutter
% % Section 2 load datafile
% % Show all images, class sizes and class names
% A = kimia_images; % accept downloading again, neglect warnings,
% figure; 
% show(A,12);
% classnames(A)
% % Section 3 Select two classes
% % 
% B = selclass(A,char('camel','elephant'));
% figure; show(B,6);
% % Section 4 Compute features
% % Compute for every object its 
% % - area (sum of pixels inside) 
% % - perimeter (sum of pixels on the contour)
% feat1 = im_stat([],'sum');
% feat2 = filtim('bwperim')*im_stat([],'sum');
% C = B*[feat1 feat2]*datasetm;
% C = setfeatlab(C,{'Size','Perimenter'});
% disp(+C);
% % Section 5 Scatterplots
% % Show the original scatterplot,
% % after normalization on variance,
% % and with a classifier
% figure; scatterd(C,'legend'); axis equal; fontsize(14);
% % Some tricky statements to get non-matching NN labels ?????????????
% [~,nn] = min(distm(C)+1e100*eye(size(C,1))); %???????????
% hold on; scatterd(C(labcmp(getlab(C),getlab(C(nn,:))),:),'ko');
% fontsize(14);
% D = C*mapex(scalem,'variance');
% figure; scatterd(D,'legend'); axis equal; fontsize(14);
% % Some tricky statements to get non-matching NN labels
% [~,nn] = min(distm(D)+1e100*eye(size(D,1)));
% hold on; scatterd(D(labcmp(getlab(D),getlab(D(nn,:))),:),'ko');
% fontsize(14);
% figure; scatterd(D,'legend'); axis equal; fontsize(14);
% plotc(knnc(D,1),'col'); %awsome way to display classifier with colors!!!
%                         %notice how each point on the classifier line is
%                         %halfway between two points from the two classes as
%                         %this is a 1-NN (nearest neighbor) classifier!!!
% fontsize(14);
% % Section 6 First classification
% % Compute the nearest neighbor errors for the two scatterplots
% fprintf('LOO NN error of original features %4.2f\n',testk(C,1));
% fprintf('LOO NN error of rescaled features %4.2f\n',testk(D,1));
% % 	E = TESTK(A,K,T)
% %
% % INPUT
% % 	A 	Training dataset
% % 	K 	Number of nearest neighbors (default 1)
% % 	T 	Test dataset (default [], i.e. find leave-one-out estimate on A)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % classifiers
% %         
% prwaitbar off                % waitbar not needed here
% delfigs                      % delete existing figures
% % randreset(n);                % takes care of reproducability ??????
% %        Define a classifier
% u = knnc([],3);              % the untrained 3-NN classifier
% %      Define datasets for training and testing
% a = gendatb([20 20],2);      % define dataset
% a = setlablist(a,[' A ';' B ']); % define class names
% [t,s] = gendat(a,0.5);       % split it 50-50 in train set and test set
% t = setname(t,'Train Set');  % name the train set 
% s = setname(s,'Test Set');   % name the test set
% %        Train the classifier
% w = t*u;                     % train the classifier
% %      Show the trained classifier on the training set
% figure;
% scatterd(t);                 % show training set
% axis equal
% plotc(w);                    % plot classifier
% V = axis;
% dt = t*w;                    % apply classifier to the training set
% et = dt*testc;               % compute its classification error
% fprintf('The apparent error: %4.2f \n',et); % print it
% labt = getlabels(t);         % true labels of training set
% labtc= dt*labeld;            % estimated labels of classified training set
% disp([labt labtc]);          % show them. They correspond to the estimated error
% %
% % Compute the apparent error and show the estimated and true labels
% % in classifying the training set. They corespond to the apparent error 
% % and the classifier in the scatter plot
% %        Show the trained classifier on the test set
% figure;
% scatterd(s);                 % show test set
% axis(V);
% plotc(w);                    % plot classifier
% ds = s*w;                    % apply classifier on the test set
% es = ds*testc;               % compute its classification error
% fprintf('The test error: %4.2f \n',es); % print it
% labs = getlabels(t);         % true labels of test set
% labsc= ds*labeld;            % estimated labels of classified test set
% disp([labs labsc]);          % show them. They correspond to the estimated error
% % Compute the test error and show the estimated and true labels
% % in classifying the test set. They corespond to the test error
% % and the classifier in the scatter plot        
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %lcurves
% % Learning curves for Bayes-Normal, Nearest Mean and Nearest Neighbour 
% % on the Iris dataset. Averages over 100 repetitions.
% %        Show datasets and best classifier
% delfigs
% % randreset(1)
% % a = iris;
% prload iris;
% a = setprior(a,0);
% scattern(a*pcam(a,2));
% title('Projection of the Iris dataset')
% fontsize(14);
% %     Show learning curve of qdc with apparent error
% figure
% prwarning off;
% e = cleval(a,qdc,[6 8 10 14  20 30 40],100);
% plote(e,'nolegend')
% legend('Test error','Apparent error')
% title('Learning curve Bayes-normal (QDA) on Iris data')
% ylabel('Averaged error (100 exp.)');
% fontsize(14);
% axis([2.0000 40.0000 0 0.1200])
% %
% % The two curves approximate each other to the performance of the best
% % possible QDA model.
% %      Show learning curves of nmc and k-nn
% e2 = cleval(a,nmc,[2 3 4 5 6 8 10 14  20 30 40],100);
% e3 = cleval(a,knnc,[2 3 4 5 6 8 10 14  20 30 40],100);
% figure;
% plote({e2,e3,e},'nolegend','noapperror')
% title('Learning curves for Iris data') 
% ylabel('Averaged error (100 exp.)');
% legend('Nearest Mean','k-Nearest Neighbor','Bayes Normal')
% fontsize(14);
% axis([2.0000 40.0000 0 0.1200])
%
% For small training sets more simple classifiers are better. More
% complicated classifiers (more parameters to be estimated) are expected to
% perform better for larger training sets. The k-NN classifier improves
% slowly, but is expected to beat Bayes Normal at some point as it
% asymptotically approximates the Bayes classifier. 
%
% The learning curve for the Nearest Mean classifier shows surprisingly a
% minimum, a phenomenon 
% <http://link.springer.com/chapter/10.1007/978-3-642-34166-3_34 discussed
% by Marco Loog et al.>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % crossval_ex 
% % Experiment to compare some crossvalidation procedures on the basis of
% % their ability to rank the expected perfomance of some classifiers
% % for various sizes of the training set. The experiment is based on the
% % satelliite dataset.
% %
% % The following crossvalidation procedures are compared:
% %
% % * kohavi10      - single 10-fold crossvalidation, see [1]
% % * kohavi10x10   - 10 times 10-fold crossvalidation, see [1]
% % * dietterich5x2 -  5 times 2-fold crossvalidation, see [2],[3]
% % * dps8          - single 8-fold density preserving split, see [4]
% % Initialise
% a       = satellite;
% a       = setprior(a,getprior(a,0));  % give classes equal priors
% alfs    = [0.01 0.02 0.05 0.1 0.2];   % part of design set used for training
% clsfs   = {nmc,knnc([],1),ldc,naivebc,parzenc,udc,treec}; % classifiers
% procs   = {'kohavi10','kohavi10x10','dietterich5x2','dps8'};
% repeats = 100;                        % number of generated training sets
% nprocs  = numel(procs);               % number of crossval prpcedures
% nclsfs  = numel(clsfs);               % number of classifiers
% nalfs   = numel(alfs);                % number of training set sizes
% eclsfs  = zeros(repeats,nalfs,nclsfs);% true classifier performances
% R       = zeros(repeats,nprocs,nalfs);% final Borda count differences
% % Run all crossvalidations
% % * various training set sizes (e.g. 5)
% % * various train / test splits (e.g. 100)
% % * various crossvalidation procedures (4)
% % * all classifiers (e.g. 7)
% % This may take several hours
% q = sprintf('Running %i crossvals: ',repeats*nprocs*nalfs);
% [NUM,STR,COUNT] = prwaitbarinit(q,repeats*nprocs*nalfs);
% for n = 1:nalfs                     % run over training set sizes
%   alf = alfs(n);
%   for r=1:repeats                   % repeat repeats times
%     randreset(r);                   % take care of reproducability
%     [s,t] = gendat(a,alf);          % full training set s, large test set t
%     eclsfs(r,n,:) = cell2mat(testc(t,s*clsfs)); % train, test all classifiers
%     [dummy,T] = sort(squeeze(eclsfs(r,n,:))'); 
%     T(T) = [1:nclsfs];              % true Borda count classifiers
%     for j=1:nprocs                  % run over crossval procedures
%       COUNT = prwaitbarnext(NUM,STR,COUNT);
%       proc = procs{j};
%       switch lower(proc)
%         case 'kohavi10'             % single 10-fold crossval, all clsfs
%           exval = prcrossval(s,clsfs,10,1); 
%         case 'kohavi10x10'          % 10 times 10-fold crossval, all clsfs
%           exval = prcrossval(s,clsfs,10,10);
%         case 'dietterich5x2'        % 5 times 2-fold crossval, all clsfs
%           exval = prcrossval(s,clsfs,2,5);
%         case 'dps8'                 % single 8-fold density preserving split
%           exval = prcrossval(s,clsfs,8,'dps');
%         otherwise
%           error('Unknown crossvalidation procedure')
%       end
%       [dummy,S] = sort(exval); 
%       S(S) = [1:nclsfs];            % estimated Borda counts
%       R(r,j,n) = sum(abs(T-S))/2;   % Borda count differences
%     end
%   end
% end
% % Learning curves
% figure;
% plot(alfs,squeeze(mean(eclsfs)));
% legend(getname(clsfs));
% xlabel('Fraction of design set used for training')
% ylabel(['Averaged test error (' num2str(repeats) ' exp.)'])
% title('Learning curves')
% fontsize(14)
% linewidth(1.5)
% % Table with results
% fprintf('    procedure  |'); 
% fprintf('    %4.2f     |',alfs); 
% fprintf('\n  ');
% fprintf('%s',repmat('-------------|',1,nalfs+1));
% fprintf('\n');
% for j=1:nprocs
%   fprintf(' %13s |',procs{j}); 
%   for n=1:nalfs
%     fprintf(' %4.2f (%4.2f) |',mean(R(:,j,n)),std(R(:,j,n))/sqrt(repeats));
%   end
%   fprintf('\n');
% end
% %
% % This table shows how well the four crossvalidation procedures are able to
% % predict the ranking of the performances of 7 classifiers trained by the
% % total training set. The columns refer to different sizes of the training
% % set. They are fractions of the total design set of 6435 objects.
% %
% % The numbers in the table are the averaged sums of the Borda count
% % differences between the estimated ranking (by the corresponding
% % procdure) and the true ranking, divided by two. A value of 0 stands for 
% % equal rankings and 12 for fully opposite rankings. A single swap between 
% % two classifiers results in a value of 1. In brackets the standard deviations
% % in the estimated avarages.
% %
% % The 10 x 10-fold crossvalidation needs about 10 times more computing time
% % than the three other procedures. Except for the smallest training sizes,
% % it performs always better than the other three. The dietterich5x2 procedure
% % starts as the best one but is for larger training sets the worst. Here it
% % suffers from the fact that in the crossvalidation it uses training sets 
% % that have just half the size of the final set for which it has to predict
% % the classifier performances. Some learning curves are not yet saturated.
% % This explains also why the final performances based on using 20% of the
% % design set are worse then at 10%: At 20% there is more confusion between
% % the classifier rankings.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % fcurves
% % Feature curves for Bayes-Normal, on the Satellite dataset.
% % Average over 25 repetitions
% %        Initialization
% delfigs 
% a = satellite;
% a = setprior(a,getprior(a));
% w = featself(a,'maha-s');
% trainsize = [20 30 50 100 500];
% iter = 25;
% % Create feature curves for optimized feature ranking
% x = a*w;
% randreset;
% e1 = cell(1,numel(trainsize));
% % prwaitbar calls are just used for reporting progress and may be skipped
% [n,s,count] = prwaitbarinit('Processing %i curves:',2*numel(trainsize));
% for j=1:numel(trainsize)
%   e1{j} = clevalf(x,remclass(2)*qdc([],[],1e-6),[1:15],trainsize(j),iter,[],testd);
%   count = prwaitbarnext(n,s,count);
% end
% plote(e1,'nolegend')
% legend('train size: 20','train size: 30','train size: 50','train size: 100','train size: 500')
% title('Feature curve for Satellite dataset, optimized feature order')
% fontsize(14)
% set(gcf,'position',[ 680 558 808 420]);
% % Create feature curves for randomized feature ranking
% figure
% randreset;
% x = a(:,randperm(size(a,2))); 
% e2 = cell(1,numel(trainsize));
% for j=1:numel(trainsize)
%   randreset;
%   e2{j} = clevalf(x,remclass(2)*qdc([],[],1e-6),[1:15],trainsize(j),iter,[],testd);
%   count = prwaitbarnext(n,s,count);
% end
% plote(e2,'nolegend');
% legend('train size: 20','train size: 30','train size: 50','train size: 100','train size: 500')
% title('Feature curve for Satellite dataset, random feature order')
% fontsize(14)
% set(gcf,'position',[ 680 558 808 420]);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % featsel_ex1
% % Examples of various feature selection procedures, organised per procedure
% % Feature curves are shown for 6 feature rankings computed by 3 procedures:
% % 
% % * Individual selection
% % * Forward selection
% % * Backward selecton
% %
% % and 2 criteria:
% %
% % * The Mahalanobis distance
% % * The leave-one-out nearest neighbor performance
% %
% % These criteria are computed for the entire training set. 
% % Each of the 6 plots shows the performance of 3 classifiers based on a
% % 50-50 random split of the dataset for training and testing.
% %
% % * 1-NN rule
% % * The Fisher classifier
% % * The linear support vector machine.
% 
% %        Show dataset
% % The Breast Wisconsin dataset is based on 9 features and 683 objects in
% % two classes of 444 and 239 objects.
% delfigs
% prload breast;
% % a = breast;
% a = setprior(a,0);
% scattern(a*pcam(a,2));
% title(['PCA projection of the ' getname(a) ' dataset'])
% fontsize(14);
% %        Define classifiers
% w1 = setname(knnc([],1),'1-NN');
% w2 = setname(fisherc,'Fisher');
% w3 = setname(libsvc,'LibSVC-1');
% % number of random train-test splits used in the feature curve routine.
% nreps = 25;
% %        Original feature curves
% % These curves are based on the feature ranking of the original dataset
% randreset; e = clevalf(a,{w1,w2,w3},[],0.5,nreps);
% figure; plote(e);
% title('Feature curves for original feature ranking')
% xlabel('Feature number'); set(gca,'xscale','linear');
% set(gca,'xticklabel',1:size(a,2)); set(gca,'xtick',1:size(a,2));
% fontsize(14);
% %       Individual selection by the Mahalanobis criterion
% v = featseli(a,'maha-s',size(a,2));
% randreset; e = clevalf(a*v,{w1,w2,w3},[],0.5,nreps);
% figure;  plote(e); 
% title('Individual selection by the Mahalanobis criterion')
% xlabel('Selected feature'); set(gca,'xscale','linear');
% set(gca,'xticklabel',+v); set(gca,'xtick',1:size(a,2));
% fontsize(14);
% %       Forward selection by the Mahalanobis criterion
% v = featself(a,'maha-s',size(a,2));
% randreset; e = clevalf(a*v,{w1,w2,w3},[],0.5,nreps);
% figure;  plote(e); 
% title('Forward selection by the Mahalanobis criterion')
% xlabel('Selected feature'); set(gca,'xscale','linear');
% set(gca,'xticklabel',+v); set(gca,'xtick',1:size(a,2));
% fontsize(14);
% %       Backward selection by the Mahalanobis criterion
% [v,r] = featselb(a,'maha-s',1);
% % the next statment is needed to retrieve the feature ranking
% v = featsel(size(a,2),[+v abs(r(2:end,3))']);
% randreset; e = clevalf(a*v,{w1,w2,w3},[],0.5,nreps);
% figure;  plote(e); 
% title('Backward selection by the Mahalanobis criterion')
% xlabel('Selected feature'); set(gca,'xscale','linear');
% set(gca,'xticklabel',+v); set(gca,'xtick',1:size(a,2));
% fontsize(14);
% %       Individual selection by the NN criterion
% v = featseli(a,'NN',size(a,2));
% randreset; e = clevalf(a*v,{w1,w2,w3},[],0.5,nreps);
% figure;  plote(e); 
% title('Individual selection by the NN criterion')
% xlabel('Selected feature'); set(gca,'xscale','linear');
% set(gca,'xticklabel',+v); set(gca,'xtick',1:size(a,2));
% fontsize(14);
% %       Forward selection by the NN criterion
% v = featself(a,'NN',size(a,2));
% randreset; e = clevalf(a*v,{w1,w2,w3},[],0.5,nreps);
% figure;  plote(e); 
% title('Forward selection by the NN criterion')
% xlabel('Selected feature'); set(gca,'xscale','linear');
% set(gca,'xticklabel',+v); set(gca,'xtick',1:size(a,2));
% fontsize(14);
% %       Backward selection by the NN criterion
% [v,r] = featselb(a,'NN',1);
% % the next statment is needed to retrieve the feature ranking
% v = featsel(size(a,2),[+v abs(r(2:end,3))']);
% randreset; e = clevalf(a*v,{w1,w2,w3},[],0.5,nreps);
% figure;  plote(e); 
% title('Backward selection by the NN criterion')
% xlabel('Selected feature'); set(gca,'xscale','linear');
% set(gca,'xticklabel',+v); set(gca,'xtick',1:size(a,2));
% fontsize(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% featsel_ex2
% % Examples of various feature selection procedures, organised per
% % classifier
% % Feature curves are shown for 6 feature rankings computed by 3 procedures:
% % 
% % * Individual selection
% % * Forward selection
% % * Backward selecton
% %
% % and 2 criteria:
% %
% % * The Mahalanobis distance
% % * The leave-one-out nearest neighbor performance
% %
% % These criteria are computed for the entire training set. 
% % Each of the 6 plots shows the performance of one of three classifiers
% % for 3 ranking procedures in a comparison with the original ranking.
% % Classifier performances are based on a 50-50 random split of the dataset
% % for training and testing. The three classifiers are:
% %
% % * 1-NN rule
% % * The Fisher classifier
% % * The linear support vector machine.
% %
% %        Show dataset
% % The Breast Wisconsin dataset is based on 9 features and 683 objects in
% % two classes of 444 and 239 objects.
% delfigs
% a = breast;
% a = setprior(a,0);
% scattern(a*pcam(a,2));
% title(['PCA projection of the ' getname(a) ' dataset'])
% fontsize(14);
% %        Define classifiers
% w1 = setname(knnc([],1),'1-NN');
% w2 = setname(fisherc,'Fisher');
% w3 = setname(libsvc,'LibSVC-1');
% w = {w1,w2,w3};
% nreps = 25;
% %        Define feature selectors using Mahalanobis distance
% % define unit mapping
% v0 = setname(prmapping,'Original ranking');
% % individual selection
% v1 = setname(featseli(a,'maha-s',size(a,2)),'Individual Selection');
% % forward selection 
% v2 = setname(featself(a,'maha-s',size(a,2)),'Forward Selection');
% % backward selection
% [v3,r] = featselb(a,'maha-s',1);
% v3 = setname(featsel(size(a,2),[+v3 abs(r(2:end,3))']),'Backward Selection');
% v = {v0,v1,v2,v3};
% %       Compute feature curves per classifier ranked for Mahalanobis distance
% for j=1:numel(w)
%   e = cell(1,numel(v));
%   for i=1:numel(v)
%     randreset; e{i} = clevalf(a*v{i},w{j},[],0.5,nreps);
%     e{i}.names = getname(v{i});
%   end
%   figure; plote(e)
%   title(['Feature curves for ' getname(w{j}) ', based on Mahalanobis distance']);
%   set(gca,'xticklabel',1:size(a,2)); set(gca,'xtick',1:size(a,2));
%   fontsize(14);
% end
% %        Define feature selectors using NN performance
% % define unit mapping
% v0 = setname(prmapping,'Original ranking');
% % individual selection
% v1 = setname(featseli(a,'NN',size(a,2)),'Individual Selection');
% % forward selection 
% v2 = setname(featself(a,'NN',size(a,2)),'Forward Selection');
% % backward selection
% [v3,r] = featselb(a,'NN',1);
% v3 = setname(featsel(size(a,2),[+v3 abs(r(2:end,3))']),'Backward Selection');
% v = {v0,v1,v2,v3};
% %       Compute feature curves per classifier ranked for NN performance
% for j=1:numel(w)
%   e = cell(1,numel(v));
%   for i=1:numel(v)
%     randreset; e{i} = clevalf(a*v{i},w{j},[],0.5,nreps);
%     e{i}.names = getname(v{i});
%   end
%   figure; plote(e)
%   title(['Feature curves for ' getname(w{j}) ', based on NN performance']);
%   set(gca,'xticklabel',1:size(a,2)); set(gca,'xtick',1:size(a,2));
%   fontsize(14);
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % apperror
% % Examples of the behavior of the apparent error for increasing training
% % set size, dimensionality and complexity.
% %        Show dataset
% delfigs
% randreset(1)
% prload iris
% a = setprior(a,0);
% scattern(a*pcam(a,2));
% title('PCA projection of the Iris dataset')
% %       Show learning curve of qdc with apparent error
% figure
% prwarning off;
% e = cleval(a,qdc,[6 8 10 14  20 30 40],100);
% plote(e,'nolegend')
% legend('Test error','Apparent error')
% title('Learning curve Bayes-normal (QDA) on Iris data')
% ylabel('Averaged error (100 exp.)');
% fontsize(14)
% axis([2.0000 40.0000 0 0.1200])
% %      Show learning curves of nmc and 1-nn
% e2 = cleval(a,nmc,[2 3 4 5 6 8 10 14  20 30 40],100);
% e3 = cleval(a,knnc([],1),[2 3 4 5 6 8 10 14  20 30 40],100);
% figure;
% plote({e2,e3,e},'nolegend','noapperror')
% title('Learning curves for Iris data')
% ylabel('Averaged error (100 exp.)');
% legend('Nearest Mean','Nearest Neighbor','Bayes Normal')
% hold on;
% plot([2 3 4 5 6 8 10 14 20 30 40],e2.apperror,'k--')
% plot([2 3 4 5 6 8 10 14 20 30 40],e3.apperror,'r--');
% plot([6 8 10 14 20 30 40],e.apperror,'b--');
% linewidth(1.5)
% fontsize(14)
% axis([2.0000 40.0000 0 0.1200])
% %
% % Note that the apparent errors (dashed lines) are expected to increase with
% % the size of the training set (more difficult for the classifier to
% % classify all training objects correctly), but will decrease with the
% % classifier complexity (more easy to classify the given training objects
% % correctly).
% %      Show feature curves of satellite dataset
% figure;
% a = satellite;
% a = setprior(a,getprior(a));
% w = featself(a,'maha-s');
% trainsize = [20 50 500];
% iter = 25;
% 
% x = a*w;
% e4 = cell(1,numel(trainsize));
% % prwaitbar calls are just used for reporting progress and may be skipped
% [n,s,count] = prwaitbarinit('Processing %i curves:',2*numel(trainsize));
% for j=1:numel(trainsize)
%   e4{j} = clevalf(x,remclass(2)*qdc([],[],1e-6),[1:15],trainsize(j),iter,[],testd);
%   count = prwaitbarnext(n,s,count);
% end
% h = plote(e4,'nolegend');
% legend(h,'train size: 20','train size: 50','train size: 500')
% title('Feature curve for Satellite dataset, optimized feature order')
% fontsize(14)
% set(gcf,'position',[ 680 558 808 420]);
% %
% % The apparent errors (dashed lines) decrease with growing classifier
% % complexity and thereby also with growing dimensionality. For small
% % training sets they decrease faster, as classifiers are then faster
% % overtrained.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% bayes_classifier
% % Example of the Bayes classifier, learning curves and the Bayes error
% % Initialisation 
% delfigs
% gridsize(300); % needed to show class borders accurately
% randreset;     % take care of reproducability
% % Define dataset based on 4 normal distributions
% % Covariance matrices
% G = {[1 0; 0 0.25],[0.02 0; 0 3],[1 0.9; 0.9 1],[3 0; 0 3]};
% % Class means
% U = [1 1; 2 1; 2.5 2.5; 2 2];
% % Class priors
% P = [0.2 0.1 0.2 0.5];
% %
% % Note that the the first three classes have small prior probabilities,
% % and the probability of the background class is much larger.
% %
% % Show maximum densities (not the sum!!) (standard Matlab)
% % This part of the code may be skipped. It is only needed for generating
% % the 3D density plot.
% % Domain of interest
% x1 = -2:.1:6; x2 = -1.5:.1:5.5;
% % Make a grid
% [X1,X2] = meshgrid(x1,x2);
% % Compute maximum density times prior probs for the 4 classes
% F = zeros(numel(x2)*numel(x1),1);
% for n=1:numel(G)
%   F = max(F,P(n)*mvnpdf([X1(:) X2(:)],U(n,:),G{n}));
% end
% % Create plot
% F = reshape(F,length(x2),length(x1));
% surf(x1,x2,F);
% caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
% axis([min(x1) max(x1) min(x2) max(x2) 0 max(F(:))])
% xlabel('x1'); ylabel('x2'); zlabel('Maximum Probability Density');
% title('3D view of the weighted densities.')
% fontsize(14);
% %
% % The 3D plot shows for every point of the grid the maxima of the four 
% % density functions, each multiplied by their class priors.
% % Compute Bayes classifier by PRTools
% U = prdataset(U,[1:numel(G)]'); % create labeled dataset for class means
% U = setprior(U,P);              % add class priors
% G = cat(3,G{:});                % put covariance matrices in 3D array
% w = nbayesc(U,G);               % Bayes classifier
% % Show class domains according to Bayes classifier
% figure; 
% axis equal
% axis([min(x1) max(x1) min(x2) max(x2)]);
% plotc(w,'col');
% box on;
% xlabel('Feature 1');
% ylabel('Feature 2');
% title('Domains assigned to the four classes by the Bayes classifier')
% fontsize(14);
% % Show classifiers in prior weighted density plot 
% figure; 
% axis equal
% axis([min(x1) max(x1) min(x2) max(x2)]);
% plotm(w); % plot weighted density map
% plotc(w); % show classifier boundaries
% box on
% title('Weighted density plots and the Bayes classifier')
% fontsize(14);
% %
% % Note that the classifier passes exactly the points of equal densities of
% % the classes.
% % Computation of the Bayes error
% %
% % If the distributions are known the Bayes error $\epsilon^*$ can be 
% % computed by integrating the areas of overlap. Here this is done be a
% % Monte Carlo procedure: a very large dataset of a $n = 1000000$ objects
% % per class, is generated from the true class distributions and tested on
% % the Bayes classifier. The standard deviation of this estimate is
% % $\sqrt(\epsilon^* \times (1-\epsilon^*)/n)$
% n = 1000000;
% a = gendatgauss(n*ones(1,4),U,G);
% f = a*w*testc;
% s = sqrt(f*(1-f)/n);
% fprintf('The estimated Bayes error is %6.4f with standard deviation %6.4f\n',f,s);
% % The learning curve
% %
% % Here it is shown how the expected classification error approximates the
% % Bayes error for large training set sizes. It is eeseential that a
% % classifier is used that can model the true class distributions. In this
% % case this is fulfilled as the true distributions are normal and the
% % classifier, qdc, can model them.
% figure;
% a = gendatgauss(10000*ones(1,4),U,G); % generate 10000 objects per class
% e = cleval(a,qdc,[3 5 10 20 50 100 200 500],100);
% plote(e);
% hold on;
% feval(e.plot,[min(e.xvalues) max(e.xvalues)],[f f],'b-.');
% legend off
% legend('Test error','Apparent error','Bayes error')
% fontsize(14);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% adaboost_ex 
% % Adaboost 2D examples based on perceptrons and decision stumps
% % Generate simple 2D examples
% randreset; % for reproducability
% a = prdataset([gencirc(100);gencirc(100)*0.5],genlab([100 100]));
% a = setprior(a,0);
% delfigs
% % Run adaboost, 200 iterations, linear peceptron as weak classifier
% figure; w = adaboostc(a,perlc([],1),200,[],1);
% title('200 base classifiers: single epoch linear perceptron');
% fontsize(14);
% figure; scatterd(a); plotc(w)
% title('Result adaboost, 200 weak classifiers combined');
% fontsize(14);
% % Run adaboost, 200 iterations, decision stump as weak classifier
% figure; w = adaboostc(a,stumpc,200,[],3);
% title('200 base classifiers: single epoch decision stump');
% fontsize(14);
% figure; scatterd(a); plotc(w)
% title('Result adaboost, 200 weak classifiers combined');
% fontsize(14);
% %showfigs
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % adaboost_comp 
% % Adaboost is compared with other base classifier generators and combiners.
% % Initialisation
% randreset(1000);
% t = gendatb([10000 10000]);
% N = [1 2 3 5 7 10 15 20 30 50 70 100 150 200 300];
% nrep = 10;
% e1 = zeros(4,numel(N),nrep);
% e2 = zeros(4,numel(N),nrep);
% delfigs
% % Adaboost base classifiers: computation and combination
% % N base classifiers based on single epoch linear percoptron are trained
% % for N = 1 ... 300. They are combined in various ways: 
% % * standard Adaboost weights
% % * decision tree
% % * Fisher based on the binary outcomes of the base classifiers
% % * Fisher based on the confidence outcomes of the base classifiers
% % Classification errors are computed for 10 repititions. 
% for i=1:nrep
%   randreset(i);
%   a = gendatb([100 100]);
%   w = adaboostc(a,perlc([],1),300);
%   v = w.data{1}.data;
%   u = w.data{2};
%   for j=1:numel(N)
%     n = N(j);
%     w1 = wvotec(stacked(v(1:n)),u(1:n));
%     w2 = a*(stacked(v(1:n))*mapm('ge',0.5)*dtc);
%     w3 = a*(stacked(v(1:n))*mapm('ge',0.5)*fisherc);
%     w4 = a*(stacked(v(1:n))*fisherc);
%     e1(:,j,i) = cell2mat(testc(t,{w1 w2 w3 w4}))';
%   end
% end
% % Adaboost classifier example
% % The first 20 base classifiers are shown and the resulting Adaboost
% % classifier based on 300 base classifiers.
% figure;
% scatterd(a);
% plotc(w,'r',4);
% plotc(v(1:20),'k--',1); legend off
% title('The problem, the first 20 base classifiers, the final Adaboost')
% fontsize(14)
% % Learning curves for increasing numbers of Adaboost generated base classifiers
% figure;
% plot(N,mean(e1,3)')
% legend('Adaboost','Dec Tree','Binary Fisher','Fisher')
% title('Adaboost compared with other combiners')
% xlabel('Number of base classifiers')
% ylabel(['Average classification error (' num2str(nrep) ' exp.)'])
% fontsize(15);
% linewidth(2);
% % Random base claasifiers
% % Instead of the Adaboost incrementally computed base classifiers, now a
% % set of N (1 ... 300) base classifiers is generated by the 1-NN rule based
% % on a randomly chosen single objects per class. They are combined in
% % various ways: 
% % * weighted voting, similar to Adaboost, but based on the performance of
% % the entire training set.
% % * decision tree
% % * Fisher based on the binary outcomes of the base classifiers
% % * Fisher based on the confidence outcomes of the base classifier
% for i=1:nrep
%   randreset(i);
%   a = gendatb([100 100]);
%   v = a*repmat({gendat([],[1 1])*knnc([],1)},1,300);
%   w = wvotec(a,stacked(v));
%   u =w.data{2};
%   for j=1:numel(N)
%     n = N(j);
%     w1 = wvotec(stacked(v(1:n)),u(1:n));
%     w2 = a*(stacked(v(1:n))*mapm('ge',0.5)*dtc);
%     w3 = a*(stacked(v(1:n))*mapm('ge',0.5)*fisherc);
%     w4 = a*(stacked(v(1:n))*fisherc);
%     e2(:,j,i) = cell2mat(testc(t,{w1 w2 w3 w4}))';
%   end
% end
% % Random Fisher combiner example
% % The first 20 random base classifiers are shown and the resulting Fisher
% % combiner based on just these 20 base classifiers.
% figure;
% scatterd(a);
% plotc(a*(stacked(v(1:20))*fisherc),'r',4)
% plotc(v(1:20),'k--',1); legend off
% title('The problem, the first 20 base classifiers combined by Fisher')
% fontsize(14)
% % Learning curves for increasing numbers of randomly generated base classifiers
% figure;
% plot(N,mean(e2,3)')
% legend('Weighted Voting','Dec Tree','Binary Fisher','Fisher')
% title('Trainable combiners compared on random base classifiers')
% xlabel('Number of base classifiers')
% ylabel(['Average classification error (' num2str(nrep) ' exp.)'])
% fontsize(15);
% linewidth(2);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% multi-class classification
% % multi-class classifiers, one-against-rest solutions, trained combiners
% %        Preparation
% delfigs;       % delete existing figures
% gridsize(300); % make accurate scatterplots
% randreset;     % set fized seed for random generator for reporducability
% % generate 2d 8-class training set
% train = gendatm(250)*cmapm([1 2],'scale'); 
% % generate test set
% test  = gendatm(1000)*cmapm([1 2],'scale');
% % rename PRTools qdc, include regularisation
% qda   = setname(qdc([],[],1e-6),'QDA')*classc;
% %        Multi-class classifiers
% %
% % Multi-class classifiers are usually based on class models, e.g. class
% % wise esimate probability density functions (PRTools examples are qdc,
% % ldc, parzendc and mogc) or optimise a multi-class decision function, e.g.
% % neural networks. The support vector classifier offered by LibSVM can also
% % be considered as an example.
% figure;
% scatterd(train);    % show the training set
% axis equal;
% w = train*qda;      % train by QDA
% e = test*w*testc;   % find classification error
% plotc(w);           % show decision boundaries
% title([getname(w) ' (' num2str(e,'%5.3f)')])
% %
% % This plot shows the non-linear decision boundaries on top of the training
% % set. They are based on the normal distributions estimated for each of the
% % classes. Note that for some regions on the left it is not clear to which
% % class they are assigned. The classification error estimated by the test
% % set is shown in the title.
% figure;
% scatterd(train);    % show the training set
% axis equal;
% plotc(w,'col');     % show class domains
% title([getname(w) ' (' num2str(e,'%5.3f)')])
% %
% % Here every class domain is indicated by a color. By an artifact of the
% % procedure the two classes on the top right are given the same color.
% figure;
% scatterd(train);    % show the training set
% axis equal;
% w = train*libsvc;   % train by LIBSVM
% e = test*w*testc;   % find classification error
% plotc(w,'col');     % show class boundaries
% title([getname(w) ' (' num2str(e,'%5.3f)')])
% %
% % Although the basis SVM is a linear classifier, the multi-class solution
% % for LIBSVM shows slight non-linear class borders. It suggests that in its
% % design a non-linear kernel is used. The multi-class LIBSVM yields very
% % often good results and is surprisingly fast in training.
% %
% %        One-against-rest classifiers
% %
% % Many classifiers like the Fisher's Linear Discriminant and the
% % traditional support vector machine (fisherc and svc in PRTools) offer
% % primiarily a 2-class discriminant. In a multi-class setting they offer
% % don't perform well, as the rest-class may dominate the result.
% figure;
% scatterd(train);    % show the training set
% axis equal;
% w = train*fisherc;  % train by FISHERC
% e = test*w*testc;   % find classification error
% plotc(w,'col');     % show class domains
% title([getname(w) ' (' num2str(e,'%5.3f)')])
% %
% % The multi-class solution found for the one-against-rest implementation of
% % Fisher shows clearly its linear nature. The test result shown in the
% % title is disappointing. Note that some class domains are degenerated in
% % the centre of the plot.
% figure;
% scatterd(train);    % show the training set
% axis equal;
% w = train*(fisherc*fisherc);  % train by FISHERC*FISHERC                      
% e = test*w*testc;   % find classification error
% plotc(w,'col');     % show class domains
% title([getname(w) ' (' num2str(e,'%5.3f)')])
% %
% % Combining the linear Fisher classifiers by, azgain, Fisher, produces 
% % a non-linear result.
% figure;
% scatterd(train);    % show the training set
% axis equal;
% w = train*svc;      % train by SVC
% e = test*w*testc;   % find classification error
% plotc(w,'col');     % show class domains
% title([getname(w) ' (' num2str(e,'%5.3f)')])
% %
% % Also the multi-class version of the linear SVM shows bad results. Some
% % class domains are not available at all.
% %         Post-processing One-against-rest classifiers by a trained combiner
% %
% % The result of a 8-class classifier is a matrix of 8 columns showing the
% % class memberships of every object to the 8 classes. This matrix can be
% % used as an input for an 8-dimensional multi-class classifier in a second
% % attempt to improve the 8-class problem. Here it is shown how this may
% % improve the disappointing results of the multi-class implementations of
% % the Fisher and SVM classifiers shown above.
% figure;
% scatterd(train);          % show the training set
% axis equal;
% w = train*(fisherc*qda);  % train by Fisher, combine by QDA
% e = test*w*testc;         % find classification error
% plotc(w,'col');           % show class domains
% title([getname(w) ' (' num2str(e,'%5.3f)')])
% %
% % Using the quadratic postprocessing by QDA, which may also be understood
% % as a trained combiner, the results for Fisher improve considerably: an
% % error decrease form 0.479 to 0.118. Note that this is also better than
% % the result of QDA alone (0.139).
% figure;
% scatterd(train);          % show the training set
% axis equal;
% w = train*(svc*qda);      % train by SVC, combine by QDA
% e = test*w*testc;         % find classification error
% plotc(w,'col');           % show class domains
% title([getname(w) ' (' num2str(e,'%5.3f)')])
% %
% % Also the linear SVC classifier improves significantly by the QDA
% % postprocessing, but it obtains exactly the same result as for QDA alone
% % (0.139).
% figure;
% scatterd(train);          % show the training set
% axis equal;
% w = train*(libsvc*qda);   % train by LIBSVC, combine by QDA
% e = test*w*testc;         % find classification error
% plotc(w,'col');           % show class domains
% title([getname(w) ' (' num2str(e,'%5.3f)')])
% %
% % LIBSVM profits from the postprocessing as well: 0.273 to 0.120. Here we
% % have two different multi-class classifiers for which the sequential
% % combination is better (0.120) than each of them separately (0.139 and
% % 0.273) 
% %         Post-processing multi-class classifiers by a trained combiner
% %
% % Finally it is illustrated that applying QDA twice may cause overtraining
% % as the postprocessing deteriorates the result.
% figure;
% scatterd(train);          % show the training set
% axis equal;
% w = train*(qda*qda);      % train by QDA, combine by QDA
% e = test*w*testc;         % find classification error
% plotc(w,'col');           % show class domains
% title([getname(w) ' (' num2str(e,'%5.3f)')])
% %
% % Applying QDA twice deteriorates the preformance: 0.139 to 0.180. The plot
% % show also clear indications of ovetraining as at some places the domain
% % of a remote class pops up.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% pca_vs_classf
% % Feature curves showing the replacement of PCA by a classifier
% %
% % Three classifiers are considered: 
% %
% % * Fisher
% % * 95% PCA followed by Fisher
% % * Fisher followed by Fisher. 
% %
% % The experiment is based on an 8-class dataset with feature sizes up to 
% % 200. Feature curves, based on 10 repeats of randomly chosen training
% % sets, are computed up to 200 features. Three sets of curves are created:
% % for 10, 100 and 1000 training objects per class.
% %        Initialization
% % Define the classifiers and the parameters of the experiment
% delfigs
% prwaitbar off
% repeats = 10;
% randreset;  % take of reproducability
% classf1 = setname(fisherc,'Fisher');
% classf2 = setname(pcam(0.95)*fisherc,'PCA95 + Fisher');
% classf3 = setname(fisherc*fisherc,'Fisher + Fisher');
% classfiers = {classf1,classf2,classf3};
% number_of_features = 200;
% number_of_objects_per_class = 1000;
% number_of_classes = 8;
% trainingset= genmdat('gendatm',number_of_features,number_of_objects_per_class*ones(1,number_of_classes));
% testset    = genmdat('gendatm',200,1000*ones(1,8));
% featsizes  = [2 3 5 7 10 14 20 30 50 70 100 140 200];
% % small sample size
% % A training set of 10 objects per class is used. This is smaller than
% % almost all feature sizes that are considered.
% figure;
% trainsize = 0.01;
% tset = gendat(trainingset,trainsize);
% featcurves = clevalf(tset,classfiers,featsizes,[],repeats,testset);
% plote(featcurves,'noapperror');
% title('Small trainingset: 10 objects per class');
% legend Location southeast
% %
% % Feature reduction by PCA is for small sample sizes globally good.
% % Supervised training of two routines like in Fisher preceded by Fisher
% % is only useful for small feature sizes.
% % intermediate sample size
% % A training set of 100 objects per class is used. The total size (800) is
% % thereby larger than the highest feature size considered (200).
% figure;
% trainsize = 0.1;
% tset = gendat(trainingset,trainsize);
% featcurves = clevalf(tset,classfiers,featsizes,[],repeats,testset);
% plote(featcurves,'noapperror');
% title('Intermediate trainingset: 100 objects per class');
% legend Location southeast
% %
% % Feature reduction by PCA is for intermediate sample sizes globally good,
% % but is for small feature sizes outperformed by a preprocessing by Fisher.
% % In these cases the training set is sufficiently large to train a second
% % routine on the outputs of a first one.
% % large sample size
% % A training set of 1000 objects per class is used. The total size (8000) is
% % thereby much larger than the highest feature size considered (200).
% figure;
% featcurves = clevalf(trainingset,classfiers,featsizes,[],repeats,testset);
% plote(featcurves,'noapperror');
% title('Large trainingset: 1000 objects per class');
% legend Location southeast
% %
% % For large training sets, feature reduction by PCA is useful for larger
% % feature sizes. However, for the considered feature sizes it is everywhere 
% % outperformed by a preprocessing by Fisher. Apparently, the training set
% % is sufficiently large to train a second routine on the outputs of a first
% % one.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% classifiers
% % Introduction of defining, training and evaluating classifiers
% %
% %         
% prwaitbar off                % waitbar not needed here
% delfigs                      % delete existing figures
% % randreset(n);                % takes care of reproducability
% %        Define a classifier
% u = knnc([],3);              % the untrained 3-NN classifier
% %        Define datasets for training and testing
% a = gendatb([20 20],2);      % define dataset
% a = setlablist(a,[' A ';' B ']); % define class names
% [t,s] = gendat(a,0.5);       % split it 50-50 in train set and test set
% t = setname(t,'Train Set');  % name the train set 
% s = setname(s,'Test Set');   % name the test set
% %       Train the classifier
% w = t*u;                     % train the classifier
% %        Show the trained classifier on the training set
% figure;
% scatterd(t);                 % show training set
% axis equal
% plotc(w);                    % plot classifier
% V = axis;
% dt = t*w;                    % apply classifier to the training set
% et = dt*testc;               % compute its classification error
% fprintf('The apparent error: %4.2f \n',et); % print it
% labt = getlabels(t);         % true labels of training set
% labtc= dt*labeld;            % estimated labels of classified training set
% disp([labt labtc]);          % show them. They correspond to the estimated error
% %
% % Compute the apparent error and show the estimated and true labels
% % in classifying the training set. They corespond to the apparent error 
% % and the classifier in the scatter plot
% %        Show the trained classifier on the test set
% figure;
% scatterd(s);                 % show test set
% axis(V);
% plotc(w);                    % plot classifier
% ds = s*w;                    % apply classifier on the test set
% es = ds*testc;               % compute its classification error
% fprintf('The test error: %4.2f \n',es); % print it
% labs = getlabels(t);         % true labels of test set
% labsc= ds*labeld;            % estimated labels of classified test set
% disp([labs labsc]);          % show them. They correspond to the estimated error
% %
% % Compute the test error and show the estimated and true labels
% % in classifying the test set. They corespond to the test error
% % and the classifier in the scatter plot        
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% sspca 
% % Semi-supervised classification by PCA, Sonar example, 250 repitions
% % Get data
% prload sonar;
% a = setprior(a,getprior(a)); % set priors equal to class frequencies
% c = getsize(a,3);            % get number of classes
% delfigs                      % delete figures
% % Initialization
% itermax = 250;               % run over itermax repititions
% classf  = setname(qdc,'QDA');% define classifier and its name
% T = [2 3 4 5:2:11 15:5:40];  % training set sizes for learnig curves
% S = [4 5 7];                 % desired PCA dimensions
% n = itermax*numel(T)*numel(S);
% q = sprintf('running over %i experiments: ',n);
% prwaitbar(n,q);
% es = zeros(numel(T),numel(S),itermax); % space for supervised results
% eu = zeros(numel(T),numel(S),itermax); % space for semisupervised results
% % Process
% r = 0;
% for iter = 1:itermax
%   for j=1:numel(T)                        % run over training set sizes
%     randreset(100*iter+j);                % take care of reproducability
%     [X,Y] = gendat(a,T(j)*ones(1,c));     % trainset and testset
%     for i=1:numel(S)                      % run over all dims
%       r = r+1; prwaitbar(n,r,[q num2str(r)]); % report progress
%       es(j,i,iter) = Y*(X*(pcam(+X,S(i))*classf))*testc; % supervised
%       eu(j,i,iter) = Y*(X*(pcam(+a,S(i))*classf))*testc; % semisupervised
%     end 
%   end
% end
% prwaitbar(0)
% % Present results
% semilogx(T,mean(es,3));
% hold on; semilogx(T,mean(eu,3),'--');
% linewidth(1.5);
% legendpar = cell(1,numel(S));
% for j=1:numel(S)
%   legendpar{j} = ['super dim = ' num2str(S(j))];
% end
% for j=1:numel(S)
%   legendpar{j+numel(S)} = [' semi dim = ' num2str(S(j))];
% end
% legend(legendpar{:},'Location','SouthWest');
% title([getname(classf) ', Learning curves ' getname(a) ' dataset'])
% xlabel('# training objects per class')
% ylabel(['Averaged class. error (' num2str(itermax) ' exp.)']);
% fontsize(14)
% % Adapt to this problem
% set(gcf,'Position',[23    57   847   468]);
% axis([2 40 0.2 0.51])
% set(gca,'xtick',[2 4 10 20 40])
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % adaboost_ex 
% % Adaboost 2D examples based on perceptrons and decision stumps
% %
% % Generate simple 2D examples
% randreset; % for reproducability
% a = prdataset([gencirc(100);gencirc(100)*0.5],genlab([100 100]));
% a = setprior(a,0);
% delfigs
% % Run adaboost, 200 iterations, linear peceptron as weak classifier
% figure; w = adaboostc(a,perlc([],1),200,[],1);
% title('200 base classifiers: single epoch linear perceptron');
% fontsize(14);
% figure; scatterd(a); plotc(w)
% title('Result adaboost, 200 weak classifiers combined');
% fontsize(14);
% % Run adaboost, 200 iterations, decision stump as weak classifier
% figure; w = adaboostc(a,stumpc,200,[],3);
% title('200 base classifiers: single epoch decision stump');
% fontsize(14);
% figure; scatterd(a); plotc(w)
% title('Result adaboost, 200 weak classifiers combined');
% fontsize(14);
% %showfigs
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% %(1).
% prload nutsbolts;
% w = gaussm(z,1);             % Estimate a mixture of Gaussians
% figure(1); 
% scatterd (z); 
% hold on; 
% plotm(w,6,[0.1 0.5 1.0]);    % Plot in 3D
% figure(2); 
% scatterd (z); 
% hold on; 
% for c = 1:4
%   w = gaussm(seldat(z,c),1); % Estimate a Gaussian per class
%   plotm(w,2,[0.1 0.5 1.0]);  % Plot in 2D
% end;
  


% %(2). 
% prload nutsbolts;
% cost = [ -0.20  0.07  0.07   0.07 ; ...
%           0.07 -0.15  0.07   0.07 ; ...
%           0.07  0.07 -0.05   0.07 ; ...
%           0.03  0.03  0.03   0.03 ];
% w1 = qdc(z);      % Estimate a single Gaussian per class
% w2 = w1*classc*costm([],cost'); % Take cost into account 
% scatterd(z); 
% plotc(w1);        % Plot without using cost
% plotc(w2);        % Plot using cost



% %(3).
% mus = [0.2 0.3; 0.35 0.75; 0.65 0.55; 0.8 0.25];
% C = [0.018 0.007; 0.007 0.011];
% z = gauss(200,mus,C); 
% w = ldc(z);          % Normal densities, identical covariances
% figure(1); scatterd(z); hold on; plotm(w);
% figure(2); scatterd(z); hold on; plotc(w);



% %(4).
% mus = [0.2 0.3; 0.35 0.75; 0.65 0.55; 0.8 0.25];
% C = 0.01*eye(2); 
% z = gauss(200,mus,C); 
% % Normal densities, uncorrelated noise with equal variances
% w = nmsc(z);         
% figure (1); scatterd (z); hold on; plotm (w);
% figure (2); scatterd (z); hold on; plotc (w);



% %(5).
% prload nutsbolts;
% cost = [ -0.20  0.07  0.07  0.07 ; ...
%           0.07 -0.15  0.07  0.07 ; ...
%           0.07  0.07 -0.05  0.07 ; ...
%           0.03  0.03  0.03  0.03 ; ...
%          -0.16 -0.11  0.01  0.07 ];
% clabels = str2mat(getlablist(z),'reject'); 
% w1 = qdc(z);      % Estimate a single Gaussian per class
% scatterd(z); 
%                   % Change output according to cost
% w2 = w1*classc*costm([],cost',clabels); 
% plotc(w1);        % Plot without using cost
% plotc(w2);        % Plot using cost



% %(6).
% z = gendats(100,1,2);             % Generate a 1D dataset
% w = qdc(z);                       % Train a classifier
% r = roc(z*w);                     % Compute the ROC curve
% plote(r);                         % Plot it



% %(7).
% prload scatter;                    % Load dataset (zset,xset)
% z  = 0.005:0.005:1.5;            % Interesting range of z
% x_ml = z;                        % Maximum likelihood
% mu_x = mean(xset); 
% mu_z = mean(zset);
% K = ((xset-mu_x)'*(zset-mu_z))*inv((zset-mu_z)'*(zset-mu_z));
% a = mu_x - K*mu_z; 
% x_ulmse = K*z + a;               % Unbiased linear MMSE
% figure; 
% clf; 
% plot(zset,xset,'.'); 
% hold on; 
% plot(z,x_ml,'k-'); 
% plot(z,x_ulmse,'k--'); 


% %(8).
% prload levelsensor;                      % prload dataset (t,z)
% figure; 
% clf; 
% plot(t,z,'k.'); 
% hold on;  % Plot it
% y = 0:0.2:30; 
% M = [ 1 2 10 ]; 
% plotstring = {'k--','k-','k:'};
% for m = 1:3
%     p = polyfit(t,z,M(m)-1);           % Fit polynomial
%     z_hat = polyval(p,y);              % Calculate plot points
%     plot(y,z_hat,plotstring{m});       %   and plot them
% end;
% axis([0 30 -0.3 0.2]);



% %(9).
% % Create a standard MATLAB dataset from data and labels.
% % Method (5.1):
% dat = [ 0.1 0.9 ; 0.3 0.95 ; 0.2 0.7 ];
% lab = { 'class 1', 'class 2', 'class 3' };
% z = prdataset(dat,lab);
% % Method (5.3):
% [nlab,lablist] = getnlab(z);  % Extract the numeric labels
% [m,k,c] = getsize(z);         % Extract number of classes
% for i = 1:c; 
%   T{i} = seldat(z,i);
% end;



% %(10).
% prload nutsbolts;      % prload the mechanical parts dataset 
% w_l = ldc(z,0,0.7);  % Train a linear classifier on z
% w_q = qdc(z,0,0.5);  % Train a quadratic classifier on z
% figure; 
% scatterd(z); % Show scatter diagram of z
% plotc(w_l);          % Plot the first classifier
% plotc(w_q,':');      % Plot the second classifier
% [0.4 0.2]*w_l*labeld % Classify a new object with z=[0.4 0.2]



% %(11).
% n = 50; 
% a = 2; 
% b = 1.5;
% x = (-2:0.01:10)'; 
% y = gampdf(x,a,b);     % Generate function. 
% z = prdataset(gamrnd(a,b,n,1),genlab(n));   % Generate dataset.
% w = parzenm(z,1);                         % Parzen, sigma = 1.
% figure; 
% scatterd(z); 
% axis([-2 10 0 0.3]);
% plotm(w,1); 
% hold on; 
% plot(x,y,':');
% w = parzenm(z,0.2);                       % Parzen, sigma = 0.2.
% figure; 
% scatterd(z); 
% axis([-2 10 0 0.3]);
% plotm(w,1); 
% hold on; 
% plot(x,y,':');



% %(12).
% prload nutsbolts;         % prload the dataset
% [w,k] = knnc(z);        % Train a k-NNR
% disp(k);                % Show the optimal k found (using knnc actually searches best k?!?!?!)
% figure; 
% scatterd(z)     % Plot the dataset
% plotc(w);               % Plot the decision boundaries
% w = knnc(z,1);          % Train a 1-NNR. why does it seem knnc(z) is worse then knnc(z,1)??!?!?!?!
% figure; 
% scatterd(z);    % Plot the dataset
% plotc(w);               % Plot the decision boundaries



% %(13).
% prload nutsbolts;                % prload the dataset z
% J = edicon(z*proxm(z),3,5,[]); % Edit z
% w = knnc(z(J,:),1);            % Train a 1-NNR
% figure; 
% scatterd(z(J,:)); plotc(w);
% J = edicon(z*proxm(z),3,5,10); % Edit and condense z
% w = knnc(z(J,:),1);            % Train a 1-NNR
% figure; 
% scatterd(z(J,:)); 
% plotc(w); 



% %(14).
% prload nutsbolts;                % prload the dataset
% w = perlc(z,1000,0.01);        % Train a linear perceptron
% figure; 
% scatterd(z);
% plotc(w);   
% w = fisherc(z);                % Train a LS error classifier
% figure; 
% scatterd(z);
% plotc(w);   
 


% %(15).
% prload nutsbolts;                % prload the dataset
% w = svc(z,'p',2,100);          % Train a quadratic kernel svc
% figure;
% scatterd(z); 
% plotc(w);
% w = svc(z,'r',0.1,100);        % Train a Gaussian kernel svc
% figure;
% scatterd(z);
% plotc(w); 



% %(16).
% prload nutsbolts;                      % prload the dataset
% [w,R] = bpxnc(z,5,500);              % Train a small network
% figure; 
% scatterd(z); 
% plotc(w);       % Plot the classifier
% figure; 
% plotyy(R(:,1),R(:,2),R(:,1),R(:,4));
%                                      % Plot the learn curves 
% [w,R] = bpxnc(z,[100 100],1000);     % Train a larger network
% figure;
% scatterd(z);
% plotc(w);       % Plot the classifier
% figure; 
% plotyy(R(:,1),R(:,2),R(:,1),R(:,4));
%                                      % Plot the learn curves 


% %(17).
% % Create a labeled dataset with 8 features, of which only 2
% % are useful, and apply various feature selection methods. 
% z = gendatd(200,8,3,3);
% w = featselm(z,'maha-s','forward',2);   % Forward selection
% figure; 
% clf; 
% scatterd(z*w); 
% title(['forward: ' num2str(+w)]);
% w = featselm(z,'maha-s','backward',2);  % Backward selection
% figure; 
% clf; 
% scatterd(z*w); 
% title(['backward: ' num2str(+w)]);
% w = featselm(z,'maha-s','b&b',2);       % B&B selection
% figure; 
% clf; 
% scatterd(z*w); 
% title(['b&b: ' num2str(+w)]);



% %(18).
% z = gendatl([200 200],0.2);	% Generate a dataset
% J = bhatm(z,0);				% Calculate criterion values. LOOK AT THIS AGAIN AND UNDERSTAND!!!
% figure; 
% clf; 
% plot(J,'r.-');	%   and plot them
% w = bhatm(z,1);				% Extract one feature
% figure; 
% clf;
% scatterd(z);   % Plot original data
% figure;
% clf;
% scatterd(z*w);	% Plot mapped data



% %(19).
% prload license_plates.mat	    % prload dataset
% figure; 
% clf;
% show(z);       % Display it
% J = fisherm(z,0);		    % Calculate criterion values
% figure; 
% clf;
% plot(J,'r.-'); %   and plot them
% w = fisherm(z,24,0.9);		% Calculate the feature extractor 
% figure;
% clf; 
% show(w);	    % Show the mappings as images



% %(20).
% im = double(imread('car.tif'));  % prload image
% figure; 
% clf; 
% imshow(im,[0 255]); % Display image
% x = im2col(im,[8 8],'distinct'); % Extract 8x8 windows
% z = prdataset(x');                 % Create dataset
% z.featsize = [8 8];              % Indicate window size
% % Plot fraction of cumulative eigenvalues
% v = pca(z,0); figure; clf; plot(v); %using 0 in pca(z,0) i get a vec of fractions of cumulative eigenvalues
% % Find 8D PCA mapping and show basis vectors 
% w = pca(z,8); figure; clf; show(w) %show(w) shows basis vectors?
% % Reconstruct image and display it
% z_hat = z*w*w';   %this is only in prtools language, in algebra it's w'*w*z              
% im_hat = col2im (+z_hat',[8 8],size(im),'distinct'); %what does the +z_hat' do?!!?1
%                                                  %apparently col2im can reproduce the image? find out!!!
% figure; clf; imshow (im_hat,[0 255]);



% %(21).
% prload worldcities;                  % prload dataset D 
% options.q = 2;
% w = mds(D,2,options);              % Map to 2D with q = 2
% figure; 
% clf; 
% scatterd(D*w,'both'); % Plot projections



% %(22).
% z = gendatb(3);      % Create some train data
% y = gendats(5);      %   and some test data
% w = proxm(z,'d',2);  % Squared Euclidean distance to z
% D = y*w;             % 5x3 distance matrix
% D = distm(y,z);      % The same 5x3 distance matrix
% w = proxm(z,'o');    % Cosine distance to z
% D = y*w;             % New 5x3 distance matrix



% %(23).
% z = gendats(5);                  % Generate some data
% figure; clf; scatterd(z);        %   and plot it
% dendr = hclust(distm(z),'s');    % Single link clustering
% figure;
% clf; 
% plotdg(dendr);      % Plot the dendrogram 



% %(24).
% prload nutsbolts_unlabeled;  % prload the data set z
% lab = kmeans(z,4);         % Perform k-means clustering
% y = prdataset(z,lab);        % Label by cluster assignment
% figure; 
% clf; 
% scatterd(y);  %   and plot it



% %(25).
% prload nutsbolts_unlabeled;    % prload the data set z
% z = setlabtype(z,'soft');    % Set probabilistic labels
% [lab,w] = emclust(z,qdc,4);  % Cluster using EM
% figure; 
% clf; 
% scatterd(z);
% plotm(w,[],0.2:0.2:1);       % Plot results



% %(26).
% prload nutsbolts_unlabeled;    % prload the data set z
% z = setlabtype(z,'soft');    % Set probabilistic labels
%                              % Cluster 1D PCAs using EM
% [lab,w] = emclust(z,qdc([],[],[],1),4); %?????????
% figure; 
% clf; 
% scatterd(z);
% plotm(w,[],0.2:0.2:1);       % Plot results



% %(27).
% z = rand(100,2);           % Generate the data set z
% w = som(z,15);             % Train a 1D SOM and show it
% figure; clf; scatterd(z); plotsom(w); 



% %(28).
% z = rand(100,2);           % Generate the data set z
% w = gtm(z,15);             % Train a 1D GTM and show it
% figure; clf; scatterd(z); plotgtm(w); 



% %(29).
% F = [0.66 0.12; 0.32 0.74];   % Define the system
% H = [1/3 1/4];
% Fs = double(single(F));       % Round-off to 32 bits 
% Hs = double(single(H));       %   floating point precision 
% B = [1; 0]; 
% D = 0;                  
% sys = ss(Fs,B,Hs,D,-1);       % Create state-space model 
% M = obsv(F,H);                % Get observability matrix 
% G = gram(sys,'o');            %   and Gramian
% eigG = eig(G);                % Calculate eigenvalues
% svdM = svd(M);                %   and singular values
% min(eigG)/max(eigG)           % Show the ratios
% min(svdM)/max(svdM)



% %(30).
% lambda = diag([0.9 1.5]);   % Define a system with 
% V = [1/3 -1; 1/4 1/2];      %   eigenvalues 0.9 and 0.5
% F = V*lambda*inv(V);
% H = inv(V);
% H(2,:) = [];    % Define a measurement matrix 
%                             %   that only observes one state
% Cv=eye(1); 
% Cw=eye(2);       % Define covariance matrices
% % Discrete steady state Kalman filter:
% [M,P,Z,E] = dlqe(F,eye(2),H,Cw,Cv);
% Cx_inf = dlyap(F,Cw)        % Solution of discrete 
% disp('Kalman gain matrix');                         disp(M);
% disp('Eigenval. of Kalman filter');                 disp(E);
% disp('Error covariance');                           disp(Z);
% disp('Prediction covariance');                      disp(P);
% disp('Eigenval. of prediction covariance');         disp(eig(P));
% disp('Solution of discrete Lyapunov equation');     disp(Cx_inf);
% disp('Eigenval. of sol. of discrete Lyapunov eq.'); disp(eig(Cx_inf));



% %(31).
% prload linsys                         % prload a system: 
% [B,p] = cholinc(sparse(Cx0),'0');   % Initialize squared
% B = full(B); B(p:M,:)=0;            %   prior uncertainty
% x_est = x0;                         %   and prior mean
% [SqCw,p] = cholinc(sparse(Cw),'0'); % Squared Cw
% SqCw = full(SqCw);
% while (1)                           % Endless loop    
%     z = acquire_measurement_vector;
%     % 1. Sequential update:
%     for n = 1:N                     % For all measurements...
%         m = B*H(n,:)';              %   get row vector from H
%         norm_m = m'*m;
%         S = norm_m + Cv(n,n);       %   innovation variance
%         K = B'*m/S;                 %   Kalman gain vector
%         inno = z(n) - H(n,:)*x_est;
%         x_est = x_est + K*inno;     %   update estimate
%         beta = (1 + sqrt(Cv(n,n)/S))/norm_m;
%         B = (eye(M)-beta*m*m')*B;   %   covariance update
%     end
%     % 2. Prediction:
%     u = get_control_vector;
%     x_est = F*x_est + L*u;          % Predict the state
%     A = [SqCw; B*F'];               % Create block matrix
%     [q,B] = qr(A);                  % QR factorization
%     B = B(1:M,1:M);                 % Delete irrelevant part
% end




% %(32).
% % prload the housing dataset, and set the baseline performance
% prload housing;
% a                           % Show what dataset we have
% % Define an untrained linear classifier
% w = ldc;      
% % Perform 5-fold cross-validation 
% err_ldc_baseline = prcrossval(a,w,5)  
% % Do the same for the quadratic classifier 
% err_qdc_baseline = prcrossval(a,qdc,5)  



% %(33).
% prload housing.mat;
% % Define an untrained linear classifier w/scaled input data
% w_sc = scalem([],'variance');
% w = w_sc*ldc;      
% % Perform 5-fold cross-validation 
% err_ldc_sc = prcrossval(a,w,5)  
% % Do the same for some other classifiers 
% err_qdc_sc = prcrossval(a,w_sc*qdc,5)  
% err_knnc_sc = prcrossval(a,w_sc*knnc,5)  
% err_parzenc_sc = prcrossval(a,w_sc*parzenc,5)  



% %(34).
% prload housing.mat;
% % Define a preprocessing
% w_pca = scalem([],'variance')*pcam([],0.9); %understand what i can do and why with scalem!!!
% % Define the classifier
% w = w_sc*ldc;      
% % Perform 5-fold cross-validation 
% err_ldc_pca = prcrossval(a,w,5)  



% %(35).
% % prload the housing dataset
% prload housing.mat;
% % Construct scaling and feature selection mapping
% w_fsf = featselo([],'in-in',5)*scalem([],'variance');
% % Calculate crossvalidation error for classifiers 
% %   trained on the optimal 5-feature set
% err_ldc_fsf = prcrossval(a,w_fsf*ldc,5)  
% err_qdc_fsf = prcrossval(a,w_fsf*qdc,5)
% err_knnc_fsf = prcrossval(a,w_fsf*knnc,5)
% err_parzenc_fsf = prcrossval(a,w_fsf*parzenc,5)



% %(36).
% % prload the housing dataset
% prload housing.mat;
% % Optimize feature set for ldc
% w_fsf = featself([],ldc,0)*scalem([],'variance');
% err_ldc_fsf = prcrossval(a,w_fsf*ldc,5)  
% % Optimize feature set for qdc
% w_fsf = featself([],qdc,0)*scalem([],'variance');
% err_qdc_fsf = prcrossval(a,w_fsf*qdc,5)
% % Optimize feature set for knnc
% w_fsf = featself([],knnc,0)*scalem([],'variance');
% err_knnc_fsf = prcrossval(a,w_fsf*knnc,5)
% % Optimize feature set for parzenc
% w_fsf = featself([],parzenc,0)*scalem([],'variance');
% err_parzenc_fsf = prcrossval(a,w_fsf*parzenc,5)



% %(37).
% prload housing.mat;                  % prload the housing dataset
% w_pre = scalem([],'variance');     % Scaling mapping
% degree = 1:3;                      % Set range of parameters
% radius = 1:0.25:3;
% for i = 1:length(degree)
%   err_svc_p(i) = ...               % Train polynomial SVC
%     prcrossval(a,w_pre*svc([],'p',degree(i)),5);
% end;
% for i = 1:length(radius)
%   err_svc_r(i) = ...               % Train radial basis SVC
%     prcrossval(a,w_pre*svc([],'r',radius(i)),5);
% end;
% figure; clf; plot(degree,err_svc_p);
% figure; clf; plot(radius,err_svc_r);



% %(38).
% prload housing.mat;                  % prload the housing dataset
% w_pre = scalem([],'variance');     % Scaling mapping
% networks = { bpxnc, neurc };       % Set range of parameters
% nlayers  = 1:2;                   
% nunits   = [4 8 12 16 20 30 40];
% for i = 1:length(networks)
%   for j = 1:length(nlayers)
%     for k = 1:length(nunits)       
%       % Train a neural network with nlayers(j) hidden layers 
%       % of nunits(k) units each, using algorithm network{i}
%       err_nn(i,j,k) = crossval(a, ...
%         w_pre*networks{i}([],ones(1,nlayers(j))*nunits(k)),5);
%     end;
%   end;
%   figure; clear all;               % Plot the errors
%   plot(nunits,err_nn(i,1,:),'-'); hold on;
%   plot(nunits,err_nn(i,2,:),'--');
%   legend('1 hidden layer','2 hidden layers');
% end;



% %(39). ????????????????? tofEstimator?!?!?!
% prload tofdata.mat; % prload tof dataset containing 150 waveforms
% Npart = 3;        % Number of partitions
% Nchunk = 50;      % Number of waveforms in one partition
% % Create 3 random partitions of the data set
% p = randperm(Npart*Nchunk); % Find random permutation of 1:150
% for n = 1:Npart
%     for i = 1:Nchunk
%         Zp{n,i} = Zraw{p((n-1)*Nchunk+i)};
%         Tp(n,i) = TOFindex(p((n-1)*Nchunk+i));
%     end
% end
% % Cross-validation
% for n = 1:Npart
%     % Create a learn set and an evaluation set
%     Zlearn = Zp;
%     Tlearn = Tp;
%     for i = 1:Npart
%         if (i == n)
%             Zlearn(i,:) = []; 
%             Tlearn(i,:) = [];
%             Zeval = Zp(i,:);  
%             Teval = Tp(i,:);
%         end
%     end
%     Zlearn = reshape(Zlearn,(Npart-1)*Nchunk,1);
%     Tlearn = reshape(Tlearn,1,(Npart-1)*Nchunk);
%     Zeval  = reshape(Zeval ,1,          Nchunk);
% 
%     % Optimize a ToF estimator
%     [parm,learn_variance,learn_bias] = ...
%        opt_ToF_estimator(Zlearn,Tlearn);
% 
%     % Evaluate the estimator
%     for i = 1:Nchunk
%         index(i) = ToF_estimator(Zeval{i},parm);
%     end
%     variance(n) = var(Teval-index);
%     bias(n) = mean(Teval-index-learn_bias);
% end




% %(40).
% data = rand(6,2);
% labels = [1 1 1, 2 2 2]';
% a = prdataset(data,labels);
% struct(a) %show stuff



% %(41).
% %delete all figures:
% delfigs;
% %reset random seed for repeatability
% %randreset(1)
% %generate in 2 dimensinos 3 normally distributed classes of 20 object for each class:
% a = prdataset(randn(60,2),genlab([20 20 20]));
% %60 by 2 dataset with 3 classes: [20 20 20]
% %give the features a name
% a = setfeatlab(a,char('size','intensity'));
% %make the distributions of the classes different and plot them
% a(1:20,:) = a(1:20,:)*0.5;
% a(21:40,1) = a(21:40,1)+4;
% a(41:60,2) = a(41:60,2)+4;
% figure
% scatterd(a);
% %create a subset of the second class:
% b = a(21:40,:);
% %20 by 2 dataset with 3 classes: [0 20 0]
% %add 4 to the second feature of this class
% b(:,2) = b(:,2) + 4*ones(20,1);
% %20 by 2 dataset with 3 classes [0 20 0]
% %concatenate this set to the original dataset
% c = [a;b]
% %80 by 2 dataset with 3 classes: [20 40 20]
% figure;
% scatterd(c);
% showfigs
% %for better annotation of the plot we may add some information on the
% %dataset, the classes and features in some recognizable way, e.g.:
% c = setname(c,'fruit dataset');
% c = setlablist(c,char('apple','banana','cherry'));
% c = setfeatlab(c,char('size','weight'));
% figure;
% scatterd(c);



% %(42).
% a = gendath([50 50]); %generate highleyman's classes, 50 objects/class
%                       %training set c (20 objects/class)
%                       %test set D (30 objects/class)
% [c,d] = gendat(a,[20 20]);
% %compute classifiers:
% w1 = ldc(c); %linear
% w2 = qdc(c); %quadratic
% w3 = parzenc(c); %parzen
% w4 = bpxnc(c,3); %neural net with 3 hidden units 
% %compute and display classification errors:
% disp([testc(d*w1),testc(d*w2),testc(d*w3),testc(d*w4)]);
% %plot data and classifiers:
% scatterd(a); %scatter plot
% %plot the 4 discriminant functions:
% plotc({w1,w2,w3,w4});


% %(43).
% %generate 10 dimensional data:
% a = gendatd([100,100],10);
% %select the training set of 40 = 2X20 objects and the test set of 160=2X80 objects
% [b,c] = gendat(a,0.2);
% %define 5 untrained classifiers, (re)set their names 
% %w1 is a linear discriminant (ldc) in the space reduces by pca:
% w1 = klm([],0.95)*ldc;
% w1 = setname(w1,'klm - ldc');
% %w2 is an ldc on the best (1-nn leave-one-out error) 3 features:
% w2 = featself([],'NN',3)*ldc;
% w2 = setname(w2,'NN-FFS-ldc');
% %w3 is an ldc on the best (ldc apparent error) 3 features:
% w3 = featself([],ldc,3)*ldc;
% w3 = setname(w3,'LDC-FFS-ldc');
% %w4 is an ldc:
% w4 = ldc;
% w4 = setname(w4,'ldc');
% %w5 is a 1-NN:
% w5 = knnc([],1);
% w5 = setname(w5,'1-NN');
% %store classifiers in a cell:
% W = {w1,w2,w3,w4,w5};
% %train them all:
% v = b*W;
% %test them all:
% disp([newline 'errors for individual classifiers'])
% testc(c,v);
% %construct combined classifier:
% v_all = [v{:}];
% %define combiners:
% wc = {prodc,meanc,medianc,maxc,minc,votec};
% %combine (result is cell array of combined classifiers):
% vc = v_all*wc;
% %test them all:
% disp([newline 'errors for combining rules']);
% testc(c,vc);




% %(44).
% a = prdataset([1,2,3;2,3,4;3,4,5;4,5,6],[3,3,5,5]')
% a = prdataset(randn(100,3),genlab([50,50],[3,5]'));
% a = setfeatlab(a,['r1';'r2';'r3'])
% struct(a)
% getfeatlab(a)
% a = addlabels(a,char('apple','pear','apple','banana'),'fruitnames')
% a = changelablist(a,1)
% getnlab(a)
% getlablist(a)
% a = changlablist(a,'fruitnames')
% getnlab(a)  
% getlablist(a)
% getlabels(a)
% nlab = getnlab(a);
% labels = lablist(nlab,:);
% [m,k] = size(a);
% [m,k,c] = getsize(a);




% %(45).
% data = [rand(3,2) ; rand(3,2)+0.5];
% labs = ['A';'A';'A';'B';'B';'B'];
% a = prdataset(data,labs);
% struct(a)
% a = prdataset(data,'labels','prior',[0.4,0.6],'featlist',['AA','BB']);
% a = set(a,'prior',[0.4,0.6],'featlist',['AA';'BB']);
% a.prior = [0.4,0.6];
% a.featlist = ['AA';'BB'];
% [n,lablist] = classsizes(a);
% prior = a.prior
% featlist = a.featlist
% %training a classifier
% w1 = ldc(a); %normal densities based linear classifier
% w2 = knnc(a,3); %3 nearest neighbor rule
% w3 = svc(a,'p',2); %support vector classifier based on a 2nd order plynomial kernel
% %untrained classifiers:
% v1 = ldc;
% v2 = knnc([],a);
% v3 = svc([],'p',2);
% %untrained classifiers (mappings) can be trained by:
% w1 = a*v1;
% w2 = a*v2;
% w3 = a*v3;
% %an equivalent way of doing this:
% V = {ldc,knnc([],a),svc([],'p',2)};
% W = a*V;
% %affine mappings may be transposed. this is useful for back projection of
% %data into the original space, for instance:
% w = klm(a,3); %computes a 3 dimensional KL transform
% b = a*w; %maps a on w, resulting in b
% c = b*w'; %back projection of b in the original space
% %train a 3-NN classifier on the generated data:
% w = knnc([],3); %untrained classifier
% v = gendatd([50,50])*w; %trained classifier
% d = b*v;
% %more examples:
% a = gendatd([50,50],10); %generate random 10D datasets
% b = gendatd([50,50],10);
% w = klm([],0.9); %untrained mapping, karhunen-loeve projection
% v = a*w; %trained mapping v
% d = b*v; %the result of the projection of b onto v
% d = a*w; %fixed
% %normalize the distances of all objects in a such that their city block
% %distances to the origin are one:
% a = gendatb([50,50]);
% w = normm;
% d = a*w;
% u = v*w; %combiner
% %more examples:
% a = gendatd([50,50],10); %generate random 10D datasets
% b = gendatd([50,50],10);
% v = klm([],0.9); %untrained karhunen loeve projection
% w = ldc; %untrained linear classifier ldc
% u = v*w; %untrained combiner
% t = a*u; %trained combiner
% d = b*t; %apply the combiner (first KL projection, then ldc) to b




% %(46).
% a = iris;
% w = pcam(a,2);
% struct(w)




% %(47).
% %generate 10 objects in 5D, mean is [1,2,3,4,5], small variances:
% a = gauss(10,[1:5],0.01*eye(5));
% %show the rounded values of the mean of a:
% disp(round(mean(a)));
% %select features 1,2 and 5
% b = featsel(a,[1,2,5]);
% %show the rounded values of the mean of b:
% disp(round(mean(b)));
% %the following statements are equivalent:
% b = featsel(a,[1,2,5]);
% b = a*featsel([],[1,2,5]);
% w = featsel([],[1,2,5]); b = a*w;




% %(48).
% %delete all figures:
% delfigs;
% %download and unpack famous 'faces' database:
% url = 'http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data';
% %url is too long to fit on a single line
% url = [url, '/att_faces.zip'];
% prdownload(url,'faces');
% %remove non-imags files:
% delete faces/README
% %this database consists of a directory of one-person subdirectories:
% %the prdatafile command interprets these as classes:
% a = prdatafile('faces');
% %faes, raw datafile with 400 objects in 40 crisp classes
% %show the first 10 classes, 10 images on a row:
% show(selclass(a,[1:10]),10);
% %compute a PCA, convert to dataset first
% %(this is easy here as all images have the same size)
% w = a*datasetm*pcam([],2);
% %show the first two eigenfaces:
% figure;
% show(w);
% %project all images on the eigenfaces and show the scatter:
% figure;
% scatterd(a*w); 
% axis equal
% %show all:
% showfigs
% %correct eigenface image for equal axis:
% figure(2);
% axis equal;





% %(49).
% a = gendatb;
% scatterd(a);
% plotc(a*parzenc);
% a = gendatb([1000,1000]);
% scatterd(a);
% plotc(a*parzenc);
% %multiple classifiers:
% a = gendatb;
% w1 = a*fisherc;
% w2 = a*qdc;
% w3 = a*knnc;
% w4 = a*dtc;
% scatterd(a);
% plotc({w1,w2,w3,w4});
% a = gendatb;
% scatterd(a);
% plotc(a*{fisherc,qdc,knnc,dtc});
% %
% a = gendatb;
% w1 = a*fisherc;
% w2 = a*qdc;
% w2 = setname(w2,'qdc');
% w3 = a*knnc;
% w4 = a*dtc;
% w4 = setname(w4,'dtc');
% scatterd(a);
% plotc({w1,w2,w3,w4});




% %(50).
% a = gendatb;
% w1 = a*fisherc;
% w2 = a*qdc;
% w3 = a*knnc;
% w4 = a*dtc;
% scatterd(a);
% plotc({w1,w2,w3,w4});
% %another way of doing it:
% a = gendatb;
% scatterd(a);
% plotc(a*{fisherc,qdc,knnc,dtc});



% %(51).
% %Densities:
% a = gendatgauss(100,[0,0]); %generate a 2D gaussian distribution
% w = a*gaussm; %estimate the density from the data
% scatterd(a); %scatter plot of the data
% plotm(w); %plot the estimated density



% %(52).
% delfigs;
% a = gendatb;
% scatterd(a);
% plotm(a*qdc);
% title('gaussian densities');
% figure;
% scatterd(a);
% plotm(a*parzenc);
% title('parzen densities'); %understand again what this means!!@!!@
% showfigs;




% %(53).
% delfigs
% a = gendatm;
% scatterd(a);
% plotc(a*qdc,'col');
% %use gridsize to increase plotting resolution:
% figure;
% scatterd(a);
% gridsize(100);
% plotc(a*qdc,'col');
% showfigs;
% %show a 2D pca space
% delfigs
% prdatasets;
% prload iris;
% scatterd(a*pcam(a,2));
% title('pca');
% %show all combinations of 2 features:
% delfigs
% figure;
% scatterd(a(:,[1,2])); title('1 2');
% scatterd(a(:,[1,3])); title('1 3');
% scatterd(a(:,[1,4])); title('1 4');
% scatterd(a(:,[2,3])); title('2 3');
% scatterd(a(:,[2,4])); title('2 4');
% scatterd(a(:,[3,4])); title('3 4');
% showfigs;




% %(54).
% delfigs;
% scatterdui(a); %?????????????????
% scatterdui(a*pcam(a));




% %(55).
% which ldc;
% delfigs;
% a = gendatb; 
% scatterd(a);
% struct(a)
% %the 2 feature values of the first 5 objects can be inspected by:
% +a(1:5,:)
% %mark them in the scatterplot:
% hold on;
% scatterd(a(1:5,:),'o');
% w1 = a*fisherc;
% %the error on the training set a can be found by:
% a*w1*testc
% plotc(w1);
% %compute a 3rd degree polynomial classifier based on fisherc and plot it:
% w2 = a*polyc([],fisherc,3);
% plotc(w2,'r');
% a*w2*testc
% %now let us split the dataset in a separate part for training and one for testing:
% [at,as] = gendat(a,0.5); %50/50 split
% w = at*{fisherc,polyc([],fisherc,3)}
% testc(as,w);




% %(56).
% %FEATURE SELECTION:
% a = malaysia;
% a = setprior(a,getprior(a)); %avoid heaps of warnings?
% [t,s] = gendat(a,0.5); %generate train/test set
% w1 = a*featseli([],'NN',4); 
% w1 = setname(w1,'ISel NN');
% w2 = a*featseli([],'maha-s',4);
% w2 = setname(w2,'ISel maha-s');
% w3 = a*featseli([],'maha-m',4);
% w3 = setname(w3,'ISel maha-m');
% w4 = a*featseli([],'in-in',4);
% w4 = setname(w4,'ISel in-in');
% w5 = a*featseli([],ldc,4,5);
% w5 = setname(w5,'ISel wrapper');
% disp([+w1;+w2;+w3;+w4;+w5]); %show selected features
% v = t*({w1,w2,w3,w4,w5}*ldc); %train all selectors and classifiers
% s*v*testc %show test result
% %
% u1 = featseli([],'maha-s',4); 
% u1 = setname(u1,'ISel maha-s');
% u2 = featself([],'maha-s',4);
% u2 = setname(u2,'FSel maha-s');
% u3 = featselb([],'maha-s',4);
% u3 = setname(u3,'BSel maha-s');
% u4 = featselo([],'maha-s',4);
% u4 = setname(u4,'OSel maha-s');
% w = t*{u1,u2,u3,u4}; %use train set for feature selection
% disp([+w{1};+w{2};+w{3};+w{4}]); %show selected features
% v = t*(w*ldc); %use train set also for classifier training
% s*v*testc %show test results
% %these results are based on a single split of the data. a full 8-fold
% %cross-validation can be executed by:
% prcrossval(a,{u1,u2,u3,u4}*ldc,8,'DPS');




% %(57).
% delfigs
% prdatafiles; %make sure that prdatafiles is in the path
% a = faces; %the ORL database of faces
% a = prdataset(a); %convert datafile into a dataset
% w1 = pcam(a,10);
% figure;
% show(w1); %show first 10 eigenfaces
% w2 = fisherm(a,10);
% figure;
% show(w2); %show first 10 fisherfacesa
% w3 = klm(a,10);
% figure;
% show(w3); %show first 10 KL-faces
% showfigs;




% %(58).
% %DISSIMILARITIES:
% a = gendatb([10,10]); %given set of 20 objects in a 2D space
% b = gendatb([50,50]); %new set of 100 objects in the same space
% w = a*proxm('m',1); %define proximity mapping, 2 to 20
% d = b*w %resulting dissimilarity matrix 100X20
% d1 = b*(a*proxm('m',2));
% d2 = b*(a*proxm('d',1));
% d3 = distm(b,a).^(0.5);
% max(max(abs(+d1 - +d2)))
% max(max(abs(+d1 - +d3)))
% max(max(abs(+d2 - +d3)))
% %-> d2 and d3 are identical due to the fact that proxm internally uses
% %distm for the 'd' option. note, however, the huge difference in computation times:
% x = rand(1000,100); %generate 1000 objects in 100D
% tic; d1 = x*(x*proxm('m',2)); toc
% tic; d3 = distm(x).^0.5; toc




% %(59).
% a = gendatb([20,20]); %training set
% b = gendatb([50,50]); %test set
% d = b*(a*proxm('d',1)); %euclidean dissimilarity matrix
% (1-d)*testc %classification - 
% b*(a*knnc([],1))*testc %1-NN classification
% d = b*(a*proxm('m',1)); %euclidean dissimilarity matrix
% (1-d)*testc %classification




%(60).
delfigs
prload sonar; %60 dimensional dataset
%compute feature curve for the original feature ranking
e = clevalf(a,{nmc,ldc,qdc,knnc},[1:5,7,10,15,20,30,45,60],0.7,25);
figure;
plote(e);
title('original');
axis([1,60,0,0.5]);
%compute feature curve for a randomized feature ranking:
r = randperm(60);
e = clevalf(a(:,r),{nmc,ldc,qdc,knnc},[1:5,7,10,15,20,30,45,60],0.7,25);
figure;
plote(e);
title('random');
axis([1,60,0,0.5]);
%compute feature curve or an optimized feature ranking:
w = a*featself('maha-m',60);
e = clevalf(a*w,{nmc,ldc,qdc,knnc},[1:5,7,10,15,20,30,45,60],0.7,25);
figure;
plote(e);
title('optimized (maha)');
axis([1,60,0,0.5]);
showfigs;
%
u = featself('',60);
e = clevalf(a,u*{nmc,ldc,qdc,knnc},[1:5,7,10,15,20,30,45,60],0.7,25);
u = featself('',60);
e = clevalf(a,u,{nmc,ldc,qdc,knnc},[1:5,7,10,15,20,30,45,60],0.7,25);





% %(61).
% %PREX_CLEVAL   PRTools example on learning curves
% %
% % Presents the learning curves for Highleyman's classes
% delfigs
% echo on
% % Set desired learning sizes
% learnsize = [3 5 10 15 20 30];
% % Generate Highleyman's classes
% A = gendath([100,100]);
% % Define classifiers (untrained)
% W = {ldc,qdc,knnc([],1),treec};
% % Average error over 10 repetitions (it may take a while)
% % Test set is the complementary part of the training set
% E = cleval(A,W,learnsize,10);
% % Output E is a structure, specially designed for plotr
% plote(E,'noapperror')   % plot without apparent error for clarity
% echo off
% 
% 
% %(62).
% %PREX_COMBINING   PRTools example on classifier combining
% %
% % Presents the use of various fixed combiners for some
% % classifiers on the 'difficult data'.
% echo on
% % Generate 10-dimensional data
% A = gendatd([100,100],10);
% % Select the training set of 40 = 2x20 objects
% % and the test set of 160 = 2x80 objects
% [B,C] = gendat(A,0.2);
% % Define 5 untrained classifiers, (re)set their names
% % w1 is a linear discriminant (LDC) in the space reduced by PCA
% w1 = klm([],0.95)*ldc;
% w1 = setname(w1,'klm - ldc');
% % w2 is an LDC on the best (1-NN leave-one-out error) 3 features
% w2 = featself([],'NN',3)*ldc;
% w2 = setname(w2,'NN-FFS - ldc');
% % w3 is an LDC on the best (LDC leave-one-out error) 3 features
% w3 = featself([],ldc,3)*ldc;
% w3 = setname(w3,'LDC-FFS - ldc');
% % w4 is an LDC
% w4 = ldc;
% w4 = setname(w4,'ldc');
% % w5 is a 1-NN
% w5 = knnc([],1);
% w5 = setname(w5,'1-NN');
% % Store classifiers in a cell
% W = {w1,w2,w3,w4,w5};
% % Train them all
% V = B*W;
% % Test them all
% disp([newline 'Errors for individual classifiers'])
% testc(C,V);
% % Construct combined classifier
% VALL = [V{:}];
% % Define combiners
% WC = {prodc,meanc,medianc,maxc,minc,votec};
% % Combine (result is cell array of combined classifiers)
% VC = VALL * WC;
% % Test them all
% disp([newline 'Errors for combining rules'])
% testc(C,VC)
% echo off
% 
% 
% %(63).
% %PREX_CONFMAT PRTools example on confusion matrix, scatterplot and gridsize
% % Prtools example code to show the use of confusion matrix,
% % scatterplot and gridsize.
% delfigs
% randstate = randreset;
% echo on
% % Load 8-class 2D problem
% randn('state',1);
% rand('state',1);
% a = gendatm;
% % Compute the Nearest Mean Classifier
% w = nmc(a);
% % Scatterplot
% figure;
% gridsize(30);
% scatterd(a,'legend');
% % Plot the classifier
% plotc(w);
% title([getname(a) ', Gridsize 30']);
% % Set higher gridsize
% gridsize(100);
% figure;
% scatterd(a,'legend');
% plotc(w);
% title([getname(a) ', Gridsize 100']);
% % Classify training set
% d = a*w;
% % Look at the confusion matrix and compare it to the scatterplot
% confmat(d);
% echo off
% randreset(randstate);
% showfigs
% c = num2str(gridsize);
% disp(' ')
% disp('   Classifier plots are inaccurate for small gridsizes. The standard');
% disp('   value of 30 is chosen because of the speed, but it is too low to');
% disp('   ensure good plots. Other gridsizes may be set by gridsize(n).')
% disp('   Compare the two figures and appreciate the difference.')
% 
% 
% 
% %(64).
% %PREX_DENSITY Various density plots
% % Prtools example to show the use of density estimators and how to
% % visualize them.
% delfigs
% figure
% echo on
% % Generate one-class data
% a = gencirc(200);
% % Parzen desity estimation
% w = parzenm(a);
% % scatterplot
% subplot(2,2,1);
% scatterd(a,[10,5]);
% plotm(w);
% title('Parzen Density')
% % 3D density plot
% subplot(2,2,2);
% scatterd(a,[10,5]);
% plotm(w,3);
% % Mixture of Gaussians (5)
% w = gaussm(a,5);
% % scatterplot
% subplot(2,2,3);
% scatterd(a,[10,5]);
% plotm(w);
% title('Mixture of 5 Gaussians')
% % 3D density plot
% subplot(2,2,4);
% scatterd(a,[10,5]);
% plotm(w,3);
% drawnow
% disp([newline '  Study figure at full screen, shrink and hit return'])
% pause
% figure
% % Define, name and store four density esimators
% W1 = gaussm;       W1 = setname(W1,'Gaussian');
% W2 = gaussm([],2); W2 = setname(W2,'Mixture of 2 Gaussians');
% W3 = parzenm;
% W4 = knnm([],10);   W4 = setname(W4,'10-Nearest Neighbor');
% W = {W1 W2 W3 W4};
% % generate data
% a = +gendath;
% % plot densities and estimator name
% for j=1:4
%     subplot(2,2,j)
%     scatterd(a,[10,5])
%     plotm(a*W{j})
%     title([getname(W{j}) ' density estimation'])
% end
% echo off
% showfigs
% 
% 
% 
% %(65).
% %PREX_EIGENFACES   PRTools example on the use of images and eigenfaces
% echo on
% % Load all faces (may take a while)
% faces = prdataset(orl);
% faces = setprior(faces,0);     % give them equal priors
% a = gendat(faces,ones(1,40));  % select one image per class
% % Compute the eigenfaces
% w = pcam(a);
% % Display them
% newfig(1,3); show(w); drawnow;
% % Project all faces onto the eigenface space
% b = [];
% for j = 1:40
%     a = seldat(faces,j);
%     b = [b;a*w];
%     % Don't echo loops
%     echo off
% end
% echo on
% % Show a scatterplot of the first two eigenfaces
% newfig(2,3)
% scatterd(b)
% title('Scatterplot of the first two eigenfaces')
% % Compute leave-one-out error curve
% featsizes = [1 2 3 5 7 10 15 20 30 39];
% e = zeros(1,length(featsizes));
% for j = 1:length(featsizes)
%     k = featsizes(j);
%     e(j) = testk(b(:,1:k),1);
%     echo off
% end
% echo on
% % Plot error curve
% newfig(3,3)
% plot(featsizes,e)
% xlabel('Number of eigenfaces')
% ylabel('Error')
% echo off
% 
% 
% 
% %(66).
% %PREX_MATCHLAB   PRTools example on K-MEANS clustering and matching labels
% % Illustrates the use of K-MEANS clustering and the match of labels.
% delfigs
% randstate = randreset;
% echo on
% rand('state',5);   % Set up the random generator (used in K-MEANS)
% a = iris;
% % Find clusters in the Iris dataset
% J1 = prkmeans(a,3);
% % Find about the same clusters, but they are
% J2 = prkmeans(a,3);
% % labeled differently due to a random initialisation.
% confmat(J1,J2);
% % Match the labels. 'Best' rotation of label names
% [J3,C] = matchlab(J1,J2);
% % since the confusion matrix is now almost diagonal.
% confmat(J1,J3);
% % Conversion from J2 to J3: J3 = C(J2,:);
% C
% echo off
% randreset(randstate);
% 
% 
% %(67).
% %PREX-MCPLOT   PRTools example on a multi-class classifier plot
% help prex_mcplot
% echo on
% gridsize(100)
% % Generate twice normally distributed 2-class data in 2D
% a = +gendath([20,20]);    % data only
% b = +gendath([20,20]);    % data only
% % Shift the second data by a vector [5,5]
% % and combine it with the first dataset in to A
% A = [a; b+5];
% % Generate 4-class labels
% lab = genlab([20 20 20 20],[1 2 3 4]');
% % Construct a 4-class dataset A
% A = prdataset(A,lab);
% A = setname(A,'4-class dataset')
% % Plot this 4-class dataset
% figure
% % Make a scatter-plot of the right size
% scatterd(A,'.'); drawnow;
% % Compute normal densities based quadratic classifier
% w = qdc(A);
% % Plot filled classification regions
% plotc(w,'col'); drawnow;
% hold on;
% % Redraw the scatter-plot
% scatterd(A);
% hold off
% echo off
% 
% 
% %(68).
% %PREX_PLOTC  PRTools example on the dataset scatter and classifier plot
% help prex_plotc
% echo on
% % Generate Higleyman data
% A = gendath([100 100]);
% % Split the data into the training and test sets
% [C,D] = gendat(A,[20 20]);
% % Compute classifiers
% w1 = ldc(C);        % linear
% w2 = qdc(C);        % quadratic
% w3 = parzenc(C);    % Parzen
% w4 = dtc(C);        % decision tree
% % Compute and display errors
% % Store classifiers in a cell
% W = {w1,w2,w3,w4};
% % Plot errors
% disp(D*W*testc);
% % Plot the data and classifiers
% figure
% % Make a scatter-plot
% scatterd(A);
% % Plot classifiers
% plotc({w1,w2,w3,w4});
% echo off
% 
% 
% %(69).
% %PREX_SPATM  PRTools example on spatial smoothing of image classification
% % Uses a multi-band image which is first segmented (classified) into 3
% % classes based on the EM-clustering applied to a feature space
% % defined by the bands. The result is then smoothed by using the spatial
% % information on the neighboring pixels.
% help prex_spatm
% delfigs
% echo on
% % Load the image
% a = emim;
% figure            % Show the multi-band image
% 
% show(a);
% % Extract a small training set
% b = gendat(a,500);
% % Use it for finding 3 clusters
% [d,w] = emclust(b,nmc,3);
% % Classify the entire image and show it
% c = a*w;
% figure; classim(c);
% title('Original classification')
% % Smooth the image and
% % Combine the spectral and spatial classifier
% % Show it
% e = spatm(c)*maxc;
% figure;
% classim(e);
% title('Smoothed classification')
% echo off
% showfigs
% 
% 
% %(70).
% %PREX_COST PRTools example on cost matrices and rejection
% % Prtools example code to show the use of cost matrices and how
% % to introduce a reject class.
% help prex_cost
% randstate = randreset;
% echo on
% % Generate a three class problem
% randn('state',1);
% rand('state',1);
% n = 30;
% class_labels = char('apple','pear','banana');
% a = [gendatb([n,n]);  gendatgauss(n,[-2 6])];
% laba = genlab([n n n],class_labels);
% a = setlabels(a,laba);
% % Compute a simple ldc
% w = ldc(a);
% % Scatterplot and classifier
% figure;
% gridsize(30);
% scatterd(a,'legend');
% plotc(w);
% % Define a classifier with a new cost matrix,
% % which puts a high cost on misclassifying
% % pears to apples
% cost = [0.0  1.0  1.0;
%     9.0  0.0  1.0;
%     1.0  1.0  0.0];
% wc = w*classc*costm([],cost,class_labels);
% plotc(wc,'b');
% % Define a classifier with a cost matrix where
% % an outlier class is introduced. For this an
% % extra column in the cost matrix has to be defined.
% % Furthermore, the class labels have to be supplied
% % to give the new class a name.
% cost = [0.0  1.0  1.0  0.2;
%     9.0  0.0  1.0  0.2;
%     1.0  1.0  0.0  0.2];
% class_labels = char('apple','pear','banana','reject');
% wr = w*classc*costm([],cost,class_labels);
% plotc(wr,'--')
% echo off
% randreset(randstate);
% disp(' ')
% disp('   The black decision boundary shows the standard ldc classifier');
% disp('   for this data. When the misclassification cost of a pear to an');
% disp('   apple is increased, we obtain the blue classifier. When on top');
% disp('   of that a rejection class is introduced, we get the blue dashed');
% disp('   classifier. In that case, all objects between the dashed lines');
% disp('   are rejected.');
% fprintf('\n');
% fprintf('  Cost of basic classifier  =  %4.2f\n',...
%     a*w*testcost([],cost,class_labels));
% fprintf('  Cost of cost classifier   =  %4.2f\n',...
%     a*wc*testcost([],cost,class_labels));
% fprintf('  Cost of reject classifier =  %4.2f\n',...
%     a*wr*testcost([],cost,class_labels));
% 
% 
% 
% %(71).
% %PREX_LOGDENS PRTools example on density based classifier improvement
% % This example shows the use and results of LOGDENS for improving
% % the classification in the tail of the distributions
% % Note that the use of CLASSC now includes the use of logaritmic densities
% % in the tails of distributions. There is no need anymore to call LOGDENS
% % explicitely.
% help prex_logdens
% delfigs
% figure
% echo on
% % Generate a small two-class problem
% randreset(7);
% a = gendatb([20 20]);
% % Compute two classifiers: Mixture of Gaussians and Parzen
% w_mogc = mogc(a)*classc;    w_mogc = setname(w_mogc,'MoG');
% w_parz = parzenc(a)*classc; w_parz = setname(w_parz,'Parzen');
% % Scatterplot with MoG classifier
% subplot(3,2,1);
% scatterd(a);
% plotc(w_mogc); xlabel(''); ylabel('');
% set(gca,'xtick',[],'ytick',[])
% title('MoG density classifier','fontsize',12)
% drawnow
% % Scatterplot with Parzen classifier
% subplot(3,2,2);
% scatterd(a);
% plotc(w_parz); xlabel(''); ylabel('');
% set(gca,'xtick',[],'ytick',[])
% title('Parzen density classifier','fontsize',12)
% drawnow
% % Scatterplot from a distance :
% % far away points are inaccurately classified
% subplot(3,2,3);
% %scatterd([a; [150 100]; [-150 -100]]);
% scatterd([a; [75 50]; [-75 -50]]);
% plotc(w_mogc); xlabel(''); ylabel('');
% set(gca,'xtick',[],'ytick',[])
% title('MoG: bad for remote points','fontsize',12)
% drawnow
% % Scatterplot from a distance :
% % far away points are inaccurately classified
% subplot(3,2,4);
% scatterd([a; [20 12]; [-20 -12]]);
% plotc(w_parz); xlabel(''); ylabel('');
% set(gca,'xtick',[],'ytick',[])
% title('Parzen: bad for remote points','fontsize',12)
% drawnow
% % Improvement of MOGC by LOGDENS
% subplot(3,2,5);
% %scatterd([a; [150 100]; [-150 -100]]);
% scatterd([a; [75 50]; [-75 -50]]);
% plotc({w_mogc,logdens(w_mogc)},['k--';'r- ']); legend off
% xlabel(''); ylabel(''); set(gca,'xtick',[],'ytick',[])
% title('MoG improved by Log-densities','fontsize',12)
% drawnow
% % Improvement of PARZEN by LOGDENS
% subplot(3,2,6);
% scatterd([a; [20 12]; [-20 -12]]);
% plotc({w_parz,logdens(w_parz)},['k--';'r- ']); legend off
% xlabel(''); ylabel(''); set(gca,'xtick',[],'ytick',[])
% title('Parzen improved by Log-densities','fontsize',12)
% echo off
% disp(' ')
% disp('    This example shows the use of the logdens() routine. It')
% disp('    improves the classification in the tails of the distribution,')
% disp('    which is especially important in high-dimensional spaces.')
% disp('    To this end it is combined with normalisation, generating')
% disp('    posterior probabilities. Logdens() can only be applied to')
% disp('    classifiers based on normal densities and Parzen estimates.')
% disp(' ')
% showfigs
% 
% 
% 
% %(72).
% %PREX_DATASETS  PRTools example of the standard datasets
% %
% % Shows quickly the scatter-plots for each pair of the
% % consecutive features from the standard datasets.
% 
% %addph datasets
% prdatasets all
% % makes sure that datafiles are available
% % will download them from the PRTools website when needed
% data = str2mat(...
%     'gauss', ...
%     'gendatb', ...
%     'gendatc', ...
%     'gendatd', ...
%     'gendath', ...
%     'gendatl', ...
%     'gendatm', ...
%     'gendats', ...
%     'gencirc', ...
%     'lines5d', ...
%     'x80', ...
%     'auto_mpg', ...
%     'malaysia', ...
%     'biomed', ...
%     'breast', ...
%     'cbands', ...
%     'chromo', ...
%     'circles3d', ...
%     'diabetes', ...
%     'ecoli', ...
%     'glass', ...
%     'heart', ...
%     'imox', ...
%     'glass', ...
%     'heart', ...
%     'imox', ...
%     'iris', ...
%     'ionosphere', ...
%     'liver', ...
%     'ringnorm', ...
%     'sonar', ...
%     'soybean1', ...
%     'soybean2', ...
%     'twonorm', ...
%     'wine');
% delfigs
% figure
% disp(' ')
% disp('   Consecutive feature pairs of the standard datasets are shown')
% disp('   in a scatter-plot. This may take a few minutes.')
% disp(' ')
% for j=1:size(data,1);
%     a = feval(deblank(data(j,:)));
%     a
%     if size(a,2) == 1
%         scatterd(a,1);
%         drawnow
%         pause(0.2)
%     else
%         for i=2:size(a,2)
%             scatterd(a(:,[i-1,i]));
%             drawnow
%             pause(0.1)
%         end
%     end
% end
% 
% 
% %(73).
% %PREX_DATAFILE  PRTools example of the datafile usage
% delfigs
% echo on
% prdatafiles
% % makes sure that datafiles are available
% % will download them from the PRTools website when needed
% a = highway
% % load datafile
% % 100 observations of 5 images and a label image
% % R,G,B
% % and two pixel features (comparisons with previous frames)
% b = gendat(a,0.06)
% % random selection of 6 observations
% figure; show(b); drawnow
% c = selectim(b,[1 2 3])
% % select RGB
% figure; show(c); drawnow
% showfigs
% x = b(2,:)
% % select one observation
% figure; show(x,3); drawnow
% showfigs
% y = data2im(x);
% % select features by retrieving images
% data = squeeze(y(:,:,1:5)); % data
% lab  = squeeze(y(:,:,6));   % labels
% datasize = size(data);
% z = im2feat(data);  % store images as feature, pixels become objects
% z = setlabels(z,lab(:));
% z = setobjsize(z,datasize(1:2));
% % stores pixels as objects with 5 features
% trainset = gendat(z,[1000,1000]);
% w = qdc(trainset) % train classifier on trainset only
% d = z*w*classc;   % classify entire image
% figure; show(z*w*classc);
% figure; imagesc(d*classim);
% figure; imagesc(lab);
% showfigs
% echo off
% 
% 
% 
% %(74).
% %PREX_MDS   PRTools example on multi-dimensional scaling
% % Show the training, generalisation of some non-linear mappings for
% % visualisation.
% help prex_mds
% %delfigs
% echo on
% a = satellite;         % 36D dataset, 6 classes, 6435 objects
% [x,y] = gendat(a,0.1); % split in train and test set
% % TSNEM
% wt = x*tsnem;          % train TSNEM
% figure;
% scattern(x*wt);        % show 2d result for trainset
% title('tSNEM trainset')
% figure;
% scattern(y*wt);        % show 2d result for testset
% title('tSNEM testset')
% showfigs
% % SAMMONM
% ws = x*sammonm;         % train SAMMONM
% figure;
% scattern(x*ws);         % show 2d result for trainset
% title('Sammon trainset')
% figure;
% scattern(y*ws);         % show 2d result for testset
% title('Sammon testset')
% showfigs
% % MDS
% dxx = sqrt(distm(x,x)); % dissimilarity matrix of trainset
% wm = mds(dxx);          % train MDS
% figure;
% scattern(dxx*wm);       % show 2d result for trainset
% title('MDS trainset')
% figure;
% dyx = sqrt(distm(y,x)); % dissimilarity between testset and trainset
% scattern(dyx*wm);       % show 2d result for testset
% title('MDS testset')
% showfigs
% echo off
% 
% 
% %(75).
% %PREX_PARZEN Parzen based denisities and classifiers
% % PRTools example to show the differences between various ways to use the
% % PARZEN procedures for estimating densities and classifiers.
% help prex_parzen
% delfigs
% figure
% echo on
% delfigs
% a = gendath;   % two normally distributed classes, different covariances
% w = a*parzenc; % Parzen classifier, single smoothing parameter optimizing
% % the classification error
% figure(1); scatterd(a); % show scatterplot
% plotm(w);  plotc(w);    % show densities and classifier
% title('Densities and classifier by PARZENC')
% w = a*parzendc;% Parzen classifier, smoothing parameter per class
% % optimizing class densities
% figure(2); scatterd(a); % show scatterplot
% plotm(w);  plotc(w);    % show densities and classifier
% title('Densities and classifier by PARZENDC')
% w = a*parzenm; % Parzen density, smoothing parameter per class
% % optimizing class densities, combined to single density
% figure(3); scatterd(a); % show scatterplot
% plotm(w);  plotc(w);    % show density
% title('Density by parzenm on labeled data')
% w = +a*parzenm; % Parzen density, classes combined, so just a single
% % smoothing parameter optimizing overall density
% figure(4); scatterd(+a);% show scatterplot
% plotm(w);               % show density
% title('Density by parzenm on unlabeled data')
% echo off
% showfigs
% 
% 
% %(76).
% %PREX_SOFT Simple example of handling soft labels in PRTools
% %
% % Soft labels are implemented next to the 'crisp' and 'targets' labels.
% % Like 'targets' labels they are stored in the target field of a dataset.
% % Their values should be between 0 and 1. For every class a soft label
% % values should be given. The density based classifiers can handle soft
% % labels, interpreting them as class weights for every objects in the
% % density estimation.
% %
% % The posterior probabilities found by classifying objects can be
% % interpreted as soft labels. They, however, sum to one (over the classes),
% % while this is not necessary for training and test objects.
% %
% % Note that the routine CLASSSIZES returns the sum of the soft labels over
% % the dataset for every class separately. In contrast to crisp labels the
% % sum over the classes of the output of CLASSSIZES is not necessarily
% % equal to number of objects in the dataset.
% %
% % The routine SELDATA(A,N) returns the entire dataset in case of a soft
% % labeled dataset A for every value of N and not just class N, as all
% % objects may participate in all classes.
% help prex_soft;
% echo on
% % Generate artificial soft labeled dataset using posteriors as soft labels
% a = gendath([100 100]);
% % retrieve a dataset with posteriors to be used for soft labels
% labels = a*qdc(a)*classc;
% % create a new dataset with soft labels
% s = prdataset(+a);
% % we just need the values of 'labels'
% s = setlabtype(s,'soft',+labels);
% % give the classes a name (optional, just to show how this is done)
% s = setlablist(s,{'A','B'});
% % experiment: % generate train set and test set
% [train_s,test_s] = gendat(s,0.5);
% % compute classifier that outputs posteriors
% w_s = parzenc(train_s)*classc;
% % apply classifier on testdata
% d_s = test_s*w_s;
% % result, by default for soft labeled data
% % the 'soft' test type is used in testc
% testc(d_s)
% % compare with crisp labeling, convert train and test set to crisp labels
% train_c = setlabtype(train_s,'crisp');
% test_c  = setlabtype(test_s,'crisp');
% % compute classifier
% w_c = parzenc(train_c)*classc;
% % apply classifier on testdata
% d_c = test_c*w_c;
% % result, by default for crisp labeled data
% % the 'crisp' test type is used in testc
% testc(d_c)
% echo off
% 
% 
% %(77).
% %PREX_SOM   PRTools example on training SelfOrganizing Maps
% % Show the training and plotting of 1- or 2-D Self-Organizing Maps.
% help prex_som
% delfigs
% echo on
% % Set size of the SOM
% k = [5 1];
% % Set the number of iterations
% nrruns = [20 40 40];
% % Set desired learning rates
% eta = [0.5 0.1 0.1];
% % Set the neighborhood widths
% h = [0.6 0.2 0.01];
% % Generate one banana class:
% A = gendatb([100,100]);
% A = seldat(A,1);
% % Train a 1D SOM:
% W = som(A,k);  % May take some time
% % Show the results in a scatter plot
% figure(1); clf;
% scatterd(A); hold on;
% prplotsom(W);
% title('One-dimensional SOM');
% drawnow
% % Train a 2D SOM:
% k = [5 5];
% W = som(A,k);  % Will take some time
% % Show the results in a scatter plot
% figure(2); clf;
% scatterd(A); hold on;
% prplotsom(W);
% title('Two-dimensional SOM');
% drawnow
% echo off
% showfigs
% 
% 
% %(78).
% %PREX_SILHOUETTE_CLASSIFICATION PRTools introductory example
% % Presents the construction of a dataset from a set of images,
% % builds a classifier and performs an evaluation
% help prex_silhouette_classification
% delfigs
% echo on
% % load the Kimia image collection
% a = kimia;
% show(a,18);
% % Select 5 classes and show them
% b = seldat(a,[1 3 5 7,13]);
% figure; show(b);
% % compute the area
% area = im_stat(b,'sum');
% % compute the perimeter
% bx = abs(filtim(b,'conv2',{[-1 1],'same'}));
% by = abs(filtim(b,'conv2',{[-1 1]','same'}));
% bor = or(bx,by);
% figure; show(bor);
% perimeter = im_stat(bor,'sum');
% % construct a dataset with equal class priors and label the features
% c = prdataset([area perimeter]);
% c = setprior(c,0);
% c = setfeatlab(c,char('area','perimeter'));
% % Show the 2D scatterplot
% figure; scatterd(c,'legend');
% % Compute a linear classifier and show it in the scatterplot
% w = ldc(c);
% plotc(w,'col');
% showfigs
% % Compute and print classification errors
% train_error = c*w*testc;
% test_error = crossval(c,ldc,2);
% echo off
% fprintf('\nerror in training set:        %5.3f\n',train_error)
% fprintf('2-fold crossvalidation error: %5.3f\n',test_error)
% 
% 
% %(79).
% %PREX_REGR PRTools regression example
% % Show the regression functions that are available in Prtools in a 1-D
% % example.
% help prex_regr;
% % Define the dataset parameters and generate the data:
% n = 25; sig = 0.2;
% a = gendatsinc(n,sig); % train
% b = gendatsinc(n,sig); % test
% % Train several regression functions:
% w1 = a*linearr([],1e-9);
% w2a = a*nu_svr([],'r',2,0.01,'epsilon',0.1);
% w2b = a*svmr([],0.01,'r',2);
% w3 = a*ridger([],10);
% w4a = a*lassor([],10);
% w4b = a*lassor([],1);
% w5a = a*ksmoothr([],1);
% w6 = a*pinvr;
% w7 = a*plsr;
% w8 = a*knnr([],1);
% w9 = a*gpr([],proxm([],'r',1),0.1);
% % Plot the functions in the scatterplot of the data:
% figure(1); clf; hold on;
% scatterr(a);
% plotr(w1,'b-');
% plotr(w2a,'r-');
% plotr(w2b,'r--');
% plotr(w3,'g-');
% plotr(w4a,'k-');
% plotr(w4b,'k--');
% plotr(w5a,'m-');
% plotr(w6,'y-');
% plotr(w7,'c-');
% plotr(w8,'b--');
% plotr(w9,'k:');
% % Show the MSE results:
% fprintf('                           MSE\n');
% fprintf('linear regression      : %f\n', b*w1*testr);
% fprintf('nu-svm regression      : %f\n', b*w2a*testr);
% fprintf('svm regression         : %f\n', b*w2b*testr);
% fprintf('ridge regression       : %f\n', b*w3*testr);
% fprintf('lasso regression (C=10): %f\n', b*w4a*testr);
% fprintf('lasso regression (C=1) : %f\n', b*w4b*testr);
% fprintf('smoother regression    : %f\n', b*w5a*testr);
% fprintf('pseudo-inv regression  : %f\n', b*w6*testr);
% fprintf('partial least squares  : %f\n', b*w7*testr);
% fprintf('kNN regression         : %f\n', b*w8*testr);
% fprintf('Gaussian Process regr. : %f\n', b*w9*testr);
% 
% 








































































