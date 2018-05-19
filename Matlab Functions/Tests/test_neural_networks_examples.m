%test_neural_networks_examples

% %%%% Define inputs and outputs:
% input_vec = [1:6]'; % 6-dimensional input vec (column vec), NOT 6 inputs
% target_vec = [1,2]';
% 
% %%%% Define and custom networks:
% number_of_inputs = 1;
% number_of_layers = 2;
% bias_connections = [1;0]; %(number_of_layers X 1) boolean
% input_connections = [1;0]; %(number_of_layers X number_of_inputs) boolean
% layer_connections = [0,0;1,0]; %(number_of_layers X number_of_layers) boolean matrix
% output_connecitons = [0;1]; 
% net = network(number_of_inputs,number_of_layers,bias_connections,input_connections,layer_connections,output_connecitons);
% view(net);
% 
% %%%%%  Define topology and transfer function:
% %number of hidden layer neurons:
% net.layers{1}.size = 5;
% %hidden layer transfer function:
% net.layers{1}.transferFcn = 'logsig';
% view(net);
% 
% %%%% Configure network:
% net = configure(net,input_vec,target_vec);
% view(net);
% 
% %%%% Train net and calculate neuron output:
% initial_output_without_training = net(inputs);
% %train network:
% net.trainFcn = 'trainlm';
% net.performFcn = 'mse';
% net = train(net,input_vec,output_vec);
% %network response after training:
% final_output = net(inputs);
% 
% 
% 
% %Classification of linearly separable data with a perceptron:
% number_of_samples_per_class = 20;
% offset_for_second_class = 5;
% inputs_vec = [randn(2,number_of_samples_per_class) , randn(2,number_of_samples_per_class) + offset_for_second_class];
% targets_vec = [zeros(1,number_of_samples_per_class) , ones(1,number_of_samples_per_class)];
% figure(1);
% plotpv(inputs_vec,targets_vec); %plotpv = plot perceptron input/target vectors
% 
% %Create and train perceptron:
% net = perceptron;
% net = train(net,inputs_vec,targets_vec);
% view(net);
% 
% %plot decision boundary:
% figure(1);
% plotpc(net.IW{1},net.b{1}); %!!!!!!!!??????



% 
% %Classification of a 4-class problem with a perceptron:
% number_of_samples_per_class = 30;
% % number_of_samples_per_class = 1;
% classes_offset = 0.6;
% q = 1;
% A = [rand(1,number_of_samples_per_class)-classes_offset; rand(1,number_of_samples_per_class)+classes_offset];
% B = [rand(1,number_of_samples_per_class)+classes_offset; rand(1,number_of_samples_per_class)+classes_offset];
% C = [rand(1,number_of_samples_per_class)+classes_offset; rand(1,number_of_samples_per_class)-classes_offset];
% D = [rand(1,number_of_samples_per_class)-classes_offset; rand(1,number_of_samples_per_class)-classes_offset];
% %plot classes:
% hold on;
% grid on;
% plot(A(1,:),A(2,:),'bs');
% plot(A(1,:),A(2,:),'bs');
% plot(B(1,:),B(2,:),'r+');
% plot(C(1,:),C(2,:),'go');
% plot(D(1,:),D(2,:),'m*');
% %text labels for classe:
% text(0.5-classes_offset,0.5+2*classes_offset,'class A');
% text(0.5+classes_offset,0.5+2*classes_offset,'class B');
% text(0.5+classes_offset,0.5-2*classes_offset,'class C');
% text(0.5-classes_offset,0.5-2*classes_offset,'class D');
% %define output coding for classes:
% a = [0,1]';
% b = [1,1]';
% c = [1,0]';
% d = [0,0]';
% %Prepare inputs and outputs for perceptron training:
% %define inputs (combine samples from all four classes):
% P = [A,B,C,D];
% %define targets:
% T = [repmat(a,1,length(A)), repmat(b,1,length(B)),repmat(c,1,length(C)),repmat(d,1,length(D))];
% %create a peceptron:
% net = perceptron;
% %train:
% E = 1;
% net.adaptParam.passes = 1; %???????????????????
% % net.layers{1}.transferFcn = 'logsig';
% % net.layers{1}.transferFcn = 'purelin';
% net.layers{1}.transferFcn = 'hardlim';
% % net.layers{1}.transferFcn = 'tansig';
% linehandle = plotpc(net.IW{1},net.b{1});
% number_of_iterations = 0;
% while (sse(E) && number_of_iterations<1000)
%    number_of_iterations = number_of_iterations+1;
%    [net,Y,E] = adapt(net,P,T);
%    linehandle = plotpc(net.IW{1},net.b{1},linehandle);
%    drawnow;
% end
% %show perceptron structure:
% view(net);




% %%%%% TIME SERIES PREDICTION:
% %define segments of time vecotrs:
% dt = 0.01;
% t1 = 0:dt:3;
% t2 = 3+dt:dt:6;
% t = [t1,t2];
% %define signal:
% y = [sin(4.1*pi*t1), 0.8*sin(8.3*pi*t2)];
% %plot signal:
% plot(t,y,'.-');
% xlabel('Time [Sec]');
% ylabel('Target Signal');
% grid on;
% ylim([-1.2,1.2]);
% 
% %PREPARE DATA FOR SEQUENTIAL NETWORK!!!!:
% % there are two basic tyes of input vectors: those that occur concurrently
% % (at the same time, or in no particular time sequence), and those that
% % ocur sequentially. for concurrent vectors, the order is not important.
% % for sequential vectors, the order in which the vectors appear is important:
% p = con2seq(y); %!!!!!!!!
% %Define ADALINE neural network:
% input_delays = 1:5;
% learning_rate = 0.2;
% %define adaline:
% net = linearlayer(input_delays,learning_rate); %!!!
% view(net);
% %adaptive learning of adaline:
% [net,output_signal,error_signal] = adapt(net,p,p); %INPUTS AND OUTPUTS ARE THE SAME!!!! - PREDICTION NETWORK!!!
% view(net);
% %check final network parameters:
% disp('weights and bias of the adaline after adaptation');
% net.IW{1};
% net.b{1};



% %%%% SOLVING XOR PROBLEM WITH A MULTILAYER PERCEPTRON:
% number_of_samples_per_class = 100;
% class_seperation = 0.6;
% A = [rand(1,number_of_samples_per_class)-class_seperation; rand(1,number_of_samples_per_class)+class_seperation];
% B = [rand(1,number_of_samples_per_class)+class_seperation; rand(1,number_of_samples_per_class)+class_seperation];
% C = [rand(1,number_of_samples_per_class)+class_seperation; rand(1,number_of_samples_per_class)-class_seperation];
% D = [rand(1,number_of_samples_per_class)-class_seperation; rand(1,number_of_samples_per_class)-class_seperation];
% figure();
% hold on;
% grid on;
% plot(A(1,:),A(2,:),'k+');
% plot(B(1,:),B(2,:),'bd');
% plot(C(1,:),C(2,:),'k+');
% plot(D(1,:),D(2,:),'bd');
% %text labels for clusters:
% text(0.5-class_seperation,0.5+2*class_seperation,'class A');
% text(0.5+class_seperation,0.5+2*class_seperation,'class B');
% text(0.5+class_seperation,0.5-2*class_seperation,'class A');
% text(0.5-class_seperation,0.5-2*class_seperation,'class B');
% %encode clusters a&c as one class, and d and d as another:
% a = -1;
% c = -1;
% b = 1;
% d = 1;
% %prepare inputs and outputs for network training:
% inputs_mat = [A,B,C,D];
% targets_vec = [repmat(a,1,length(A)),repmat(b,1,length(B)),repmat(c,1,length(C)),repmat(d,1,length(D))];
% %create a neural network:
% net = feedforwardnet([5,3]);
% %train network:
% net.divideParam.trainRatio = 1;
% net.divideParam.valRatio = 0;
% net.divieParam.testRatio = 0;
% [net,tr,Y,E] = train(net,inputs_mat,targets_vec); % tr?!?!?!?!?!?!
% view(net);
% %plot classificaiton results for the complete input space:
% span = -1:0.005:2;
% [P1,P2] = meshgrid(span,span);
% pp = [P1(:),P2(:)]';
% %simulate neural network on grid:
% aa = net(pp);
% %plot classification regions:
% figure(1);
% mesh(P1,P2,reshape(aa,length(span),length(span))-5);
% colormap cool;



% %%%% CLASSIFICATION OF A 4 CLASS PROBLEM USING A MULTILAYER PERCEPTRON:
% number_of_samples_per_class = 100;
% class_seperation = 0.6;
% A = [rand(1,number_of_samples_per_class)-class_seperation; rand(1,number_of_samples_per_class)+class_seperation];
% B = [rand(1,number_of_samples_per_class)+class_seperation; rand(1,number_of_samples_per_class)+class_seperation];
% C = [rand(1,number_of_samples_per_class)+class_seperation; rand(1,number_of_samples_per_class)-class_seperation];
% D = [rand(1,number_of_samples_per_class)-class_seperation; rand(1,number_of_samples_per_class)-class_seperation];
% %coding of 4 separate classes:
% a = [-1,-1,-1,+1]';
% b = [-1,-1,+1,-1]';
% c = [-1,+1,-1,-1]';
% d = [+1,-1,-1,-1]';
% %Define inputs (combine samples from all four classes):
% P = [A,B,C,D];
% %Define targets:
% T = [repmat(a,1,length(A)),repmat(b,1,length(B)),repmat(c,1,length(C)),repmat(d,1,length(D))];
% %create and train a multilayer perceptron:
% net = feedforwardnet([4,3]);
% %train net:
% net.divideParam.trainRatio = 1;
% net.divideParam.valRatio = 0;
% net.divideParam.testRatio = 0;
% [net,tr,Y,E] = train(net,P,T);
% view(net);
% %evaluate network performance and plot results:
% [m,i] = max(T); %target class
% [m,j] = max(Y); %prediction class
% N = length(Y); %number of samples
% number_of_misclassified_samples = 0;
% if find(i-j)
%     number_of_misclassified_samples = length(find(i-j));
% end;
% fprintf('correct classified samples: %.1f%% samples\n',100*(N-number_of_misclassified_samples)/N);




% %%%% INDUSTRIAL DIAGNOSTIC OF COMPRESSOR CONNECITON ROD DEFECTS:
% %industrial data:
% load data2.mat
% whos
% %show data for class 1:OK:
% figure;
% plot(force','c');
% grid on;
% hold on;
% plot(force(find(target==1),:)','b');
% xlabel('Time');
% ylabel('Force');
% title(notes{1});
% %show data for class 2: overload
% figure;
% plot(force','c');
% grid on;
% hold on;
% plot(force(find(target==2),:)','r');
% xlabel('Time');
% ylabel('Force');
% title(notes{2});
% %show data for class 3: Crack:
% figure;
% plot(force','c');
% grid on;
% hold on;
% plot(force(find(target==3),:)','m');
% xlabel('Time');
% ylabel('Force');
% title(notes{3});
% %downsample- include only every step-th data:
% step = 10;
% force = force(:,1:step:size(force,2));
% %Define binary output coding: 0=ok, 1=error:
% target = double(target>1);
% %Create and train a multilayer perceptron:
% %create a neural network:
% net = feedforwardnet([4]);
% %set early stopping parameters:
% net.divideParam.trainRatio = 0.7;
% net.divideParam.valRatio = 0.15;
% net.divideParam.testRatio = 0.15;
% %train a neural network:
% [net,tr,Y,E] = train(net,force',target');
% %evaluate network performance:
% threshold = 0.5;
% Y = double(Y>threshold)'; %DIGITIZE NETWORK RESPONSE!!!!
% %percentage of correct classifications:
% correct_classifications = 100*length(find(Y==target))/length(target);



% %PREDICTION OF CHAOTIC TIME SERIES WITH NAR neural network:
% N = 700; %number of samples:
% Nu = 300; %number of learning samples
% %Mackay-Glass time series:
% b = 0.1;
% c = 0.2;
% tau = 17;
% %initialization:
% y = [0.9697 0.9699 0.9794 1.0003 1.0319 1.0793 1.1076 1.1352 1.1485 1.1482 1.1383 1.1234 1.1072 1.0928 1.0820 1.0756 1.0739 1.0759]';
% %Generate Mackay-Glass time series:
% for n = 18:N+99
%     y(n+1) = y(n) - b*y(n) + c*y(n-tau)/(1+y(n-tau).^10);
% end
% %remove initial values:
% y(1:100) = [];
% %plot training and validation data:
% plot(y,'m-');
% grid on; hold on;
% plot(y(1:Nu),'b');
% plot(y,'+k','markersize',2);
% legend('validation data','training data','sampling markers','location','southwest');
% xlabel('time [steps]');
% ylabel('y');
% ylim([-0.5,1.5]);
% set(gcf,'position',[1,60,800,400]);
% %prepare training data:
% yt = con2seq(y(1:Nu)');
% %prepare validation data:
% yv = con2seq(y(Nu+1:end)');
% %Define nonlinear autoregressive neural network:
% inputDelays = 1:6:19; %input delay vector
% hiddenSizes = [6,3]; %network structure (number of neurons)
% net = narnet(inputDelays,hiddenSizes);
% %Prepare input and target time series data for network training
% % [Xs,Xi,Ai,Ts,EWs,shift] = preparets(net,Xnf,Tnf,Tf,EW);
% % the above function simplifies the normally complex and error prone tast
% % of reformatting input and target timeseries. it automatically shifts
% % input and target time series as many steps as are needed to fill the
% % initial input and layer delay states. if the network has open loop
% % feedback, then it copies feedback targets into the inputs as needed to
% % define the open loop inputs.
% %(1). INPUTS:
% % net = neural network
% % Xnf = non feedback inputs
% % Tnf = non-feedback targets
% % Tf = feedback targets
% % EW = error weights (default = 1)
% % %(2). OUTPUTS:
% % Xs = shifted inputs 
% % Xi = initial input delay states
% % Ai = initial layer delay states
% % Ts = shifted targets:
% [Xs,Xi,Ai,Ts] = preparets(net,{},{},yt); %IMPORTANT TO SEE HOW THEY DID THIS, WITHOUT NON-FEEDBACK INPUTS
% %Train net:
% net = train(net,Xs,Ts,Xi,Ai);
% view(net)
% %TRANSFORM NETWORK INTO A CLOSED LOOP NAR NETWORK:
% %close feedback for recursive prediction:
% net = closeloop(net);
% %view closeloop version of the net:
% view(net)
% %RECURSIVE PREDICTION ON VALIDATION DATA:
% %prepare validation data for network simulation:
% yini = yt(end-max(inputDelays)+1:end);  %initial values from training data
% %combine initial values and validation data 'yv'
% [Xs,Xi,Ai] = preparets(net,{},{},[yini,yv]);
% %predict on validation data:
% predict = net(Xs,Xi,Ai);
% %validation data:
% Yv = cell2mat(yv);
% %prediction:
% Yp = cell2mat(predict);
% %Error:
% e = Yv - Yp;
% %plot results of recursive simulation:
% figure(1)
% plot(Nu+1:N,Yp,'r');
% plot(Nu+1:N,e,'g');
% legend('validation data','training data','sampling markers','prediction','error','location','southwest');



% %FUNCTION APPROXIMATION WITH RBFN:
% %generate data:
% [X,Xtrain,Ytrain,fig] = data_generator(); %WHAT IS THIS FUNCTION!?!?!?!?!?
% %no hidden layers:
% net = feedforwardnet([]); %a way to have no hidden layers!!!
% %SET EARLY STOPING PARAMETERS:
% net.divideParam.trainRatio = 1;
% net.divideParam.valRatio = 0;
% net.divideParam.testRatio = 0;
% %train a neural network:
% net.trainParam.epochs = 200; %!!@!#!# a way to set epochs!!!!
% net = train(net,Xtrain,Ytrain);
% view(net);
% %simulate network over complete input range:
% Y = net(X);
% %plot network reponse:
% figure(fig);
% plot(X,Y,'color',[1,0.4,0]);
% legend('original function','available data','linear regression','location','northwest');
% %NOW FOR RBFN:
% %choose a spread constant:
% spread = 0.4;
% %create a neural network:
% net = newrbe(Xtrain,Ytrain,spread); %!!$!#$#@$@$@#$#@ RBFN
% view(net)
% %plot network reponse
% figure(fig);
% plot(X,Y,'r');
% legend('original function','available data','Exact RBFN','location','northwest'); %VERY BAD AT GENERALIZATION!!!
% %RBFN GENERALIZATION:
% %choose spread constant 
% spread = 0.2;
% %choose max number of neurons:
% K = 40;
% %performance goal (SSE);
% goal = 0;
% %number of neurons to add between displays
% Ki = 5;
% %create a neural network:
% net = newrb(Xtrain,Ytrain,goal,spread,K,Ki); %COOL WAY TO TRY SEVERAL NUMBER OF NEURONS!@#!#!
% view(net);
% Y = net(X);
% 
% 
% %GRNN!!!!!
% [X,Xtrain,Ytrain,fig] = data_generator();
% spread = 0.12;
% net = newgrnn(Xtrain,Ytrain,spread); %!!@#@#!#@!$!@ NEWGRNN
% view(net);
% Y = net(X);
% figure(fig);
% lot(X,Y,'r');
% legend('original function','available data','RBFN','location','northwest');
% 
% 
% %RBFN trained by BAYESIAN REGULARIZATION!@!#@!:
% [X,Xtrain,Ytrain,fig] = data_generator();
% spread = 0.2;
% %max number of neurons:
% K = 20;
% %performance goal (SSE)
% goal = 0;
% %number of neurons to add between displays:
% Ki = 20;
% %create a neural network:
% net = newrb(Xtrain,Ytrain,goal,spread,K,Ki);
% view(net);
% Y = net(X);
% figure(fig);
% plot(X,Y,'r');
% %show RBFN centers:
% c = net.iw{1};
% plot(c,zeros(size(c)),'rs');
% legend('original function','available data','RBFN','centers','location','northwest');
% %trainbr:
% %retrain a RBFN using bayesian regularization backpropagation:
% net.trainFcn = 'trainbr'; %@#$@#$@# trainbr = train bayesian regularization
% net.trainParam.epochs = 100;
% %perform Levenberg-Marquardt training with Bayesian regularization:
% net = train(net,Xtrain,Ytrain);
% %simulate a network over complete input range:
% Y = net(X);
% figure(fig);
% plot(X,Y,'m');
% %show RBFN centers:
% c = net.iw{1};
% plot(c,ones(size(c)),'ms');
% legend('original function','available data','RBFN','centers','RBFN + trainbr','new centers','location','northwest');
% 
% 
% 
% %MLP 
% % generate data:
% [X,Xtrain,Ytrain,fig] = data_generator();
% %create a neural network:
% net = feedforwardnet([12,6]);
% %set earily stopping parameters:
% net.divideParam.trainRatio = 1;
% net.divideParam.valRatio = 0;
% net.divideParam.testRatio = 0;
% %train a neural network:
% net.trainParam.epochs = 200;
% net = train(net,Xtrain,Ytrain);
% %view net:
% view(net);
% %simulate:
% Y = net(X);
% figure(fig);
% plot(X,Y,'color',[1,0.4,0]);
% legend('original function','available data','MLP','location','northwest');



% %CLASSIFICATION OF XOR PROBLEM WITH AN EXACT RBFN:
% K = 100;
% q = 0.6;
% A = [rand(1,K)-q,rand(1,K)+q ; ...
%     rand(1,K)+q,rand(1,K)-q];
% B = [rand(1,K)+q,rand(1,K)-q ; ...
%     rand(1,K)+q,rand(1,K)-q];
% plot(A(1,:),A(2,:),'k+',B(1,:),B(2,:),'b*');
% grid on;
% hold on;
% %coding (+1/-1) for 2-class XOR problem
% a = -1;
% b = 1;
% %define inputs ( combine samples from all four classes)
% P = [A,B];
% %define targets:
% T = [repmat(a,1,length(A)),repmat(b,1,length(B))];
% %CREATE AN EXACT RBFN:
% spread = 1;
% net = newrbe(P,T,spread);
% view(net);
% Y = net(P);
% correct = 100*length(find(T.*Y>0)) / length(T);
% figure;
% plot(T');
% hold on;
% grid on;
% plot(Y','r');
% ylim([-2,2]);
% set(gca,'ytick',[-2,0,2]);
% legend('Targets','Network response');
% xlabel('Sample No.');
% %generate a grid:
% span = -1:0.025:2;
% [P1,P2] = meshgrid(span,span);
% pp = [P1(:),P2(:)]';
% %simulate net:
% aa = sim(net,pp);
% %plot classification regions based on MAX activation:
% figure(1);
% ma = mesh(P1,P2,reshape(-aa,length(span),length(span))-5);
% mb = mesh(P1,P2,reshape(aa,length(span),length(span))-5);
% set(ma,'facecolor',[1,0.2,0.7],'linestyle','none');
% set(mb,'facecolor',[1,1,0.5],'linestyle','none');
% view(2);
% %PLOT RBFN centers:
% plot(net.iw{1}(:,1),net.iw{1}(:,2),'gs');



% %CLASSIFICATION OF XOR PROBLEM WITH AN RBFN:
% K = 100;
% q = 0.6;
% A = [rand(1,K)-q,rand(1,K)+q ; ...
%     rand(1,K)+q,rand(1,K)-q];
% B = [rand(1,K)+q,rand(1,K)-q ; ...
%     rand(1,K)+q,rand(1,K)-q];
% plot(A(1,:),A(2,:),'k+',B(1,:),B(2,:),'b*');
% grid on;
% hold on;
% %coding (+1/-1) for 2-class XOR problem
% a = -1;
% b = 1;
% %define inputs ( combine samples from all four classes)
% P = [A,B];
% %define targets:
% T = [repmat(a,1,length(A)),repmat(b,1,length(B))];
% %CREATE AN EXACT RBFN:
% spread = 1;
% %NEWRB ALGORITHM:
% %the following steps are repeated until the network's MSE falls below goal:
% % 1. the network is simulated
% % 2. the input vector with th egreatest error is found
% % 3. a radial basis neuron is added with weights equal to the vector
% % 4. the purelin layer weights are redesigned to minimize error
% %
% %max number of neurons:
% K = 20;
% goal = 0;
% Ki = 4; %number of neurons to add between displays:
% net = newrb(P,T,goal,spread,K,Ki);
% view(net);
% Y = net(P);
% correct = 100*length(find(T.*Y>0)) / length(T);
% figure;
% plot(T');
% hold on;
% grid on;
% plot(Y','r');
% ylim([-2,2]);
% set(gca,'ytick',[-2,0,2]);
% legend('Targets','Network response');
% xlabel('Sample No.');
% %generate a grid:
% span = -1:0.025:2;
% [P1,P2] = meshgrid(span,span);
% pp = [P1(:),P2(:)]';
% %simulate net:
% aa = sim(net,pp);
% %plot classification regions based on MAX activation:
% figure(1);
% ma = mesh(P1,P2,reshape(-aa,length(span),length(span))-5);
% mb = mesh(P1,P2,reshape(aa,length(span),length(span))-5);
% set(ma,'facecolor',[1,0.2,0.7],'linestyle','none');
% set(mb,'facecolor',[1,1,0.5],'linestyle','none');
% view(2);
% %PLOT RBFN centers:
% plot(net.iw{1}(:,1),net.iw{1}(:,2),'gs');




% %1D and 2D self organizing map:
% K = 200;
% q = 1.1;
% P = [rand(1,K)-q,rand(1,K)+q,rand(1,K)+q,rand(1,K)-q; ...
%      rand(1,K)+q,rand(1,K)+q,rand(1,K)-q,rand(1,K)-q];
%  plot clusters
%  plot(P(1,:),P(2,:),'g.');
%  hold on;
%  grid on;
% %SOM parameters:
% dimensions = [100];
% coverSteps = 100;
% iniNeighbor = 10;
% topologyFcn = 'gridtop';
% distanceFcn = 'linkdist';
% %define net:
% net1 = selforgmap(dimensions,coverSteps,iniNeighbor,topologyFcn,distanceFcn);
% %train:
% [net1,Y] = train(net1,P);
% %plot 1D-SOM results:
% %plot input data and SOM weight positions:
% plotsompos(net1,P);
% grid on;
% %Create and train 2D-SOM:
% %SOM parameters:
% dimensions = [10,10];
% coverSteps = 100;
% iniNeighbor = 4;
% topologyFcn = 'hextop';
% distanceFcn = 'linkdist';
% %define net:
% net2 = selforgmap(dimensions,coverSteps,iniNeighbor,topologyFcn,distanceFcn);
% %train:
% [net2,Y] = train(net2,P);
% %plot 2D-SOM results:
% %plot input data and SOM weight positions:
% plotsompos(net2,P);
% grid on;
% %plot SOM neighbor distances:
% plotsomnd(net2);
% %plot for each SOM neuron the number of input vectors that it classifies:
% figure;
% plotsomhits(net2,P);



% %Prepare inputs by PCA:
% %1. Standardize inputs to zero mean, variance one:
% [pn,ps1] = mapstd(force'); %!$$@#%$#%$#%$#%#$%@$############
% %2. Apply principal component analysis
% %inputs whose contribution to total variation are less than maxfrac are removed
% FP.maxfrac = 0.1;
% %process inputs with PCA:
% [ptrans,ps2] = processpca(pn,FP); %#$@$@$@#%@#$%%%%%%%%%%%%%%%%%%
% ps2
% %transformed inputs:
% force2 = ptrans';
% whos force force2
% %plot data in the space of first 2 PCA components:
% figure;
% plot(force2(:,1),force2(:,2),'.'); %OK
% grid on; 
% hold on;
% plot(force2(find(target>1),1), force2(find(target>1),2),'r.'); %NOT_OK
% xlabel('pca1');
% ylabel('pca2');
% legend('OK','NOT OK','location','nw');
% %binary coding 0/1:
% target = double(target > 1);
% %Create and train a multilayer perceptron:
% net = feedforwardnet([6,4]);
% %set early stopping parameters:
% net.divideParam.trainRatio = 0.7;
% net.divideParam.valRatio = 0.15;
% net.divideParam.testRatio = 0.15;
% %train a neural network:
% [net,tr,Y,E] = train(net,force2',target');
% %show net:
% view(net);
% %digitize performance:
% threshold = 0.5;
% Y = double(Y > threshold)';
% %find percentage:
% cc = 100*length(find(Y==target))/length(target);
% fprintf('correct classification: %.1f [%%]\n',cc);
% %plot classification results:
% figure(2);
% a = axis;
% xspan = a(1)-10:0.1:a(2)+10;
% yspan = a(3)-10:0.1:a(4)+10;
% [P1,P2] = meshgrid(xspan,yspan);
% pp = [P1(:),P2(:)]';
% %simulate neural network on a grid:
% aa = sim(net,pp);
% aa = double(aa > threshold);
% %plot classification regions based on MAX activation:
% ma = mesh(P1,P2,reshape(-aa,length(yspan),length(xspan))-4);
% mb = mesh(P1,P2,reshape(aa,length(yspan),length(xspan))-5);
% set(ma,'facecolor',[0.7,1,1],'linestyle','none');
% set(mb,'facecolor',[1,0.7,1],'linestyle','none');
% view(2);


















































