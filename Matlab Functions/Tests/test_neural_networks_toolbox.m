%Neural networks toolbox:

feed_forward_net = feedforwardnet;
% feed_forward_net =
% 
%     Neural Network
%  
%               name: 'Feed-Forward Neural Network'
%           userdata: (your custom info)
%  
%     dimensions:
%  
%          numInputs: 1
%          numLayers: 2
%         numOutputs: 1
%     numInputDelays: 0
%     numLayerDelays: 0
%  numFeedbackDelays: 0
%  numWeightElements: 10
%         sampleTime: 1
%  
%     connections:
%  
%        biasConnect: [1; 1]
%       inputConnect: [1; 0]
%       layerConnect: [0 0; 1 0]
%      outputConnect: [0 1]
%  
%     subobjects:
%  
%              input: Equivalent to inputs{1}
%             output: Equivalent to outputs{2}
%  
%             inputs: {1x1 cell array of 1 input}
%             layers: {2x1 cell array of 2 layers}
%            outputs: {1x2 cell array of 1 output}
%             biases: {2x1 cell array of 2 biases}
%       inputWeights: {2x1 cell array of 1 weight}
%       layerWeights: {2x2 cell array of 1 weight}
%  
%     functions:
%  
%           adaptFcn: 'adaptwb'
%         adaptParam: (none)
%           derivFcn: 'defaultderiv'
%          divideFcn: 'dividerand'
%        divideParam: .trainRatio, .valRatio, .testRatio
%         divideMode: 'sample'
%            initFcn: 'initlay'
%         performFcn: 'mse'
%       performParam: .regularization, .normalization
%           plotFcns: {'plotperform', plottrainstate, ploterrhist,
%                     plotregression}
%         plotParams: {1x4 cell array of 4 params}
%           trainFcn: 'trainlm'
%         trainParam: .showWindow, .showCommandLine, .show, .epochs,
%                     .time, .goal, .min_grad, .max_fail, .mu, .mu_dec,
%                     .mu_inc, .mu_max
%  
%     weight and bias values:
%  
%                 IW: {2x1 cell} containing 1 input weight matrix
%                 LW: {2x2 cell} containing 1 layer weight matrix
%                  b: {2x1 cell} containing 2 bias vectors
%  
%     methods:
%  
%              adapt: Learn while in continuous use
%          configure: Configure inputs & outputs
%             gensim: Generate Simulink model
%               init: Initialize weights & biases
%            perform: Calculate performance
%                sim: Evaluate network outputs given inputs
%              train: Train network with examples
%               view: View diagram
%        unconfigure: Unconfigure inputs & outputs
%  
%     evaluate:       outputs = feed_forward_net(inputs)
%  



feed_forward_net.layers{1}
% ans = 
% 
%     Neural Network Layer
%  
%               name: 'Hidden'
%         dimensions: 10
%        distanceFcn: (none)
%      distanceParam: (none)
%          distances: []
%            initFcn: 'initnw'
%        netInputFcn: 'netsum'
%      netInputParam: (none)
%          positions: []
%              range: [10x2 double]
%               size: 10
%        topologyFcn: (none)
%        transferFcn: 'tansig'
%      transferParam: (none)
%           userdata: (your custom info)
 


%CHANGE TRANSFER FUNCTION TO logsig:
feed_forward_net.layers{1}.transferFcn = 'logsig';

%VIEW layerWeights SUB-OBJECTS FOR THE WEIGHT BETWEEN LAYER 1 AND LAYER 2:
feed_forward_net.layerWeights{2,1}
% ans = 
% 
%     Neural Network Weight
% 
%             delays: 0
%            initFcn: (none)
%       initSettings: .range
%              learn: true
%           learnFcn: 'learngdm'
%         learnParam: .lr, .mc
%               size: [0 10]
%          weightFcn: 'dotprod'
%        weightParam: (none)
%           userdata: (your custom info)




%CONFIGURE THE NETWORK YOU CREATED USING A SET OF INPUTS AND TARGETS:
inputs_vec = -2: 0.1 : 2;
targets_vec = sin(pi*inputs_vec/2);
feed_forward_net1 = configure(feed_forward_net,inputs_vec,targets_vec);

feed_forward_net1.inputs{1};
% ans = 
% 
%     Neural Network Input
% 
%               name: 'Input'
%     feedbackOutput: []
%        processFcns: {'removeconstantrows', mapminmax}
%      processParams: {1x2 cell array of 2 params}
%    processSettings: {1x2 cell array of 2 settings}
%     processedRange: [1x2 double]
%      processedSize: 1
%              range: [1x2 double]
%               size: 1
%           userdata: (your custom info)



%SET UP A NETWORK WITH TWO INPUTS AND 1 OUTPUT WITH A CERTAIN WEIGHT MATRIX
%AND BIAS (== LINEAR FEEDFORWARD NETWORK!!!):
linear_feed_forward_net = linearlayer;
linear_feed_forward_net.inputs{1}.size = 2;
linear_feed_forward_net.layers{1}.dimensions = 1;
linear_feed_forward_net.IW{1,1} = [1,2];
linear_feed_forward_net.b{1} = 0;


%CREATE A DYNAMIC NETWORK WHERE THE WEIGHTS RECEIVE THE CURRENT AND
%PREVIOUS(!!!) INPUT (THAT IS == 1 DELAY):
delay_line = [0,1];
linear_dynamic_feed_forward_net = linearlayer(delay_line);
linear_dynamic_feed_forward_net.inputs{1}.size = 1; %ONLY 1 INPUT, THE NETWORK OBJECT TAKES CARE OF THE DELAY LINE
linear_dynamic_feed_forward_net.layers{1}.dimensions = 1;
linear_dynamic_feed_forward_net.biasConnect = 0; %NO BIAS CONNECTION!!!!


%TRAINING A NEURAL NETWORK:
%(1). INCREMENTAL TRAINING STATIC NETWORKS (incremental training == training/updating weights example after example):
inputs_mat = {[1;2] , [2;1], [2;3], [3;1]}; %each time sample of an input vec is a column vec: p1 = [1;
                                                                                                  %  2]
targets_vec = {4,5,7,7};
feed_forward_net2 = linearlayer(0,0);
feed_forward_net2 = configure(feed_forward_net2,inputs_mat,targets_vec);
feed_forward_net2.inputWeights{1,1}.learnParam.lr = 0.1; %learning rate for weights
feed_forward_net2.biases{1,1}.learnParam.lr = 0.1; %learning rate for bias
[feed_forward_net2,weights_cell_array,errors_cell_array,pf] = adapt(feed_forward_net2,inputs_mat,targets_vec);

%(2). INCREMENTAL TRAINING DYNAMIC NETWORKS:
initial_input = {1};
inputs_vec = {2,3,4};
targets_vec = {3,5,7};
learning_rate = 0.1;
feed_forward_net3 = linearlayer([0,1],learning_rate);
feed_forward_net3 = configure(feed_forward_net3,inputs_vec,targets_vec,initial_input);
feed_forward_net3.IW{1,1} = [0,0];
feed_forward_net3.biasConnect = 0;
[feed_forward_net3,weights_cell_array,errors_cell_array,pf] = adapt(feed_forward_net3,inputs_vec,targets_vec,initial_input);

%(3). BATCH TRAINING STATIC NETWORKS (training using a whole batch to speed up adaptation):
%INCREMENTAL TRAINING IS DONE WITH ADAPT!!!!!!
%BATCH TRAINING IS DONE WITH TRAIN!!!!!
inputs_mat = [1,2,2,3; 2,1,3,1]; %a special matrix for of the above for batch training
targets_vec = [4,5,7,7];
feed_forward_net4 = linearlayer(0,0.01); %probably learning rate is smaller because of batch training???
feed_forward_net4 = configure(feed_forward_net4,inputs_mat,targets_vec);
feed_forward_net4.IW{1,1} = [0,0];
feed_forward_net4.b{1} = 0;
feed_forward_net4.trainParam.epochs = 1; 
feed_forward_net4 = train(feed_forward_net4, inputs_mat, targets_vec);

%TRAIN ALWAYS PERFORMS BATCH TRAINING, REGARDLESS OF THE FORMAT OF THE INPUT
%ADAPT CAN IMPLEMENT INCREMENTAL OR BATCH TRAINING, DEPENDING ON THE FORMAT
%OF THE INPUT DATA. IF THE DATA IS PRESENTED AS A MATRIX OF CONCURRENT
%VECTORS THEN BATCH TRAINING OCCURS, AND IF THE DATA IS PRESENTED AS A
%SEQUENCE THEN INCREMENTAL TRAINING OCCURS.


%(4). BATCH TRAINING DYNAMIC NETWORKS
feed_forward_net5 = linearlayer([0,1], 0.02);
feed_forward_net5.inputs{1}.size = 1;
feed_forward_net5.layers{1}.dimensions = 1;
feed_forward_net5.IW{1,1} = [0,0];
feed_forward_net5.biasConnect = 0;
feed_forward_net5.trainParam.epochs = 1;
initial_input = {1};
inputs_vec = {2,3,4};
targets_vec = {3,5,6};
feed_forward_net5 = train(feed_forward_net5,inputs_vec,targets_vec,iitial_input);
%VIEW NETWORK!!!:
view(feed_forward_net5);
%NETWORK IS SIMULATED IN SEUQENTIAL MODE, BECAUSE THE INPUT IS A SEQUENCE,
%BUT THE WEIGHTS ARE UPDATED IN BATCH MODE!!!!???



%TRAINING FEEDBACK:
number_of_epochs_to_show_feedback_after = 35;
feed_forward_net5.trainParam.showwindow = false;
feed_forward_net5.trainParam.showCommandLine = true;
feed_forward_net5.trainParam.show = number_of_epochs_to_show_feedback_after;

%MANUALLY OPEN AND CLOSE THE TRAINING WINDOW:
nntraintool
nntraintool('close');



%CONSTRUCT DEEP NETWORK USING AUTOENCODERS:
%load the sample data:
[X,T] = wine_dataset;
hidden_layer_size = 10;
auto_encoder1 = trainAutoencoder(X, hidden_layer_size,...
                                 'L2WeightRegularization',0.001,...
                                 'SparsityRegularization',4,... %????
                                 'SparsiyProportion',0.05,... %????
                                 'DecoderTransferFunction','purelin');
%extract the features in the hidden layer:
features1 = encode(autoenc1,X);

%TRAIN A SECOND AUTOENCODER USING THE FEATURES FROM THE FIRST AUTOENCODER.
%DO NOT SCALE THE DATA:
hidden_layer_size = 10;
auto_encoder2 = trainAutoencoder(features1,hidden_layer_size,...
                                 'L2WeightRegularization',0.001,...
                                 'SparsityRegularization',4,... %????
                                 'SparsiyProportion',0.05,... %????
                                 'DecoderTransferFunction','purelin',...
                                 'ScaleData',false);
%extract features in the hidden layer:
features2 = encode(autoenc2, features1);

%train a softmax layer for classification using the features, features2,
%from the second autoencoder:
softmax_net = trainSoftmaxLayer(features2,T,'LossFunction','crossentropy');
%Stack the encoders and the softmax layer to form a deep network:
deepnet = stack(auto_encoder1, auto_encoder2, softmax_net);
%train the deep network on the wine data:
deep_network = train(deepnet,X,T);
%estimate the wine types using the deep network:
wind_type = deepnet(X);
%plot the confusino matrix:
plotconfusion(T,wine_type);


%FOCUSED TIME DELAY NEURAL NETWORK (FTDNN), a feedforward net with a time
%delay only at the input:
y = laser_dataset;
y = y(1:600);
ftdnn_net = timedelaynet([1:8],10);
ftdnn_net.trainParam.epochs = 1000;
ftdnn_net.divideFcn = '';
%because the network has a tapped delay line with a maximum delay of 8,
%begin by predicting the ninth value of the time series:
inputs_vec = y(9:end);
targets_vec = y(9:end);
initial_values_vec = y(1:8);
%which could be replaced with:
[inputs_vec,initial_values_vec,initial_outputs,targets_vec] = preparets(ftdnn_net,y,y);
%train the network:
ftdnn_net = train(ftdnn_net,inputs_vec,targets_vec,initial_values_vec);


%USING preparets:
[X,Xi,Ai,T,EW,shifts] = preparets(net,inputs,targets,feedback_targets,error_weights);
% X = input for training and simulation
% Xi = initial inputs for loading the tapped delay lines for input weights
% Ai = initial layer outputs for loading the tapped delay lines for layer weights
% T = training targets
% EW = error weights
% shifts = time shift between network inputs and outputs 



%TIME DISTRIBUTED DELAY NEURAL NETWORKS (TDNN):
time = 0:99;
y1 = sin(2*pi*time/10);
y2 = sin(2*pi*time/5);
y = [y1,y2,y1,y2];
t1 = ones(1,100);
t2 = -ones(1,100);
t = [t1,t2,t1,t2];
delay_line1 = 0:4;
delay_line2 = 0:3;
inputs_vec = con2seq(y);  %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
targets_vec = con2seq(t); %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dtdnn_net = distdelaynet({delay_line1,delay_line2},5);
dtdnn_net.trainFcn = 'trainbr';
dtdnn_net.divideFcn = '';
dtdnn_net.trainParam.epochs = 100;
dtdnn_net = train(dtdnn_net,inputs_vec,targets_vec);
yp = sim(dtdnn_net,p); %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
plotresponse(targets_vec,yp);



%Nonlinear AutoRegressive neural network with eXogeneous inputs (NARX):
load magdata;
y = con2seq(y);
u = con2seq(u);
d1 = [1:2];
d2 = [1:2];
narx_net = narxnet(d1,d2,10);
narx_net.divideFcn = '';
narx_net.trainParam.min_grad = 1e-10;
[p,Pi,Ai,t] = preparets(narx_net,u,{},y); %fourth argument is feedback-targets = y. 
narx_net = train(narx_net,p,t,Pi);
yp = sim(narx_net,p,Pi);
e = cell2mat(yp) - cell2mat(t);
plot(e);
%convert the network to parallel form (closed loop):
narx_net_closed = closeloop(narx_net);
view(narx_net);
view(narx_net_closed);
%perform an iterated prediction of 900 time steps:
y1 = y(1700:2600);
u1 = u(1700:2600);
[p1,Pi1,Ai1,t1] = preparets(narx_net_closed,u1,{},y1);
yp1 = narx_net_closed(p1,Pi1,Ai1);
TS = size(t1,2);
plot(1:TS,cell2mat(t1),'b',1:TS,cell2mat(yp1),'r');


%Multiple External Variables with NARX networks:
[X,T] = ph_dataset;
net = narxnet(10);
[x,xi,ai,t] = preparets(net,X,{},T);
net = train(net,x,t,xi,ai);
y = net(x,xi,ai);
e = gsubtract(t,y);



%Create reference model controller with matlab script:
[u,y] = robotarm_dataset;
d1 = [1:2];
d2 = [1:2];
S1 = 5;
narx_net = narxnet(d1,d2,S1);
narx_net.divideFcn = '';
narx_net.inputs{1}.processFcns = {};
narx_net.inputs{2}.processFcns = {};
narx_net.outputs{2}.processFcns = {};
narx_net.trainParam.min_grad = 1e-10;
[p,Pi,Ai,t] = preparets(narx_net,u,{},y);
narx_net = train(narx_net,p,t,Pi);
narx_net_closed = closeloop(narx_net);
view(narx_net_closed);
%now that the NARX plant modle is trained you can create the total MRAC
%(model reference adaptive control) system and insert the NARX modle
%inside. begin with a feedforward network, and then add the feedback
%connections. also, turn off learning in the plant model subnetwork, since
%it has already been trained. the next stage of training will train only
%the controller subnetwork.
mrac_net = feedforwardnet([S1,1,S1]);
mrac_net.layerConnect = [0,1,0,1; ...
                        1,0,0,0; ...
                        0,1,0,1; ...
                        0,0,1,0];
mrac_net.outputs{4}.feedbackMode = 'closed';
mrac_net.layers{2}.transferFcn = 'purelin';
mrac_net.layerWeights{3,4}.delays = 1:2;
mrac_net.layerWeights{3,2}.delays = 1:2;
mrac_net.layerWeights{3,2}.learn = 0;
mrac_net.layerWeights{3,4}.learn = 0;
mrac_net.layerWeights{4,3}.learn = 0;
mrac_net.biases{3}.learn = 0;
mrac_net.biases{4}.learn = 0;
%turn off data division and preprocessing:
mrac_net.divideFcn = '';
mrac_net.inputs{1}.processFcns = {};
mrac_net.outputs{4}.processFcns = {};
mrac_net.name = 'Model Reference Adaptive Control Network';
mrac_net.layerWeights{1,2}.delays = 1:2;
mrac_net.layerWeights{1,4}.delays = 1:2;
mrac_net.inputWeights{1}.delays = 1:2;
%to configure the network you need some sample training data. the following
%code loads and plots the training data and configures the network:
[refin,refout] = refmodel_dataset;
ind = 1:length(refin);
plot(ind,cell2mat(refin),ind,cell2mat(refout));
mrac_net = configure(mrac_net,refin,refout);
%now insert the weights from the trained plant model network into the
%appropriate location of the MRAC system:
mrac_net.LW{3,2} = narx_net_closed.IW{1};
mrac_net.LW{3,4} = narx_net_closed.LW{1,2};
mrac_net.b{3} = narx_net_closed.b{1};
mrac_net.LW{4,3} = narx_net_closed.LW{2,1};
mrac_net.b{4} = narx_net_closed.b{2};
%you can set the output weights of the controller network to zero, which
%will give the plant an initial input of zero:
mrac_net.LW{2,1} = zeros(size(mrac_net.LW{2,1}));
mrac_net.b{2} = 0;
%you can also associate any plots and training funciton that you desire to
%the network:
mrac_net.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotregression','plotresponse'};
mrac_net.trainFcn = 'trainlm';
view(mrac_net);
%you can now prepare the training data and train the networ:
[x_tot,xi_tot,ai_tot,t_tot] = preparets(mrac_net,refin,{},refout);
mrac_net.trainParam.epochs = 50;
mrac_net.trainParam.min_grad = 1e-10;
[mrac_net,tr] = train(mrac_net, x_tot, t_tot, xi_tot, ai_tot);




%TRAIN NEURAL NETWORKS WITH ERROR WEIGHTS!!!:
y = laser_dataset;
y = y(1:600);
ind = 1:600;
ew = 0.99.^(600-ind);
figure;
plot(ew);
ew = con2seq(ew);
ftdnn_net = timedelaynet([1:8],10);
ftdnn_net.trainParam.epochs = 1000;
ftdnn_net.divideFcn = '';
[p,Pi,Ai,t,ew1] = preparets(ftdnn_net,y,y,{},ew);
[ftdnn_net1,tr] = train(ftdnn_net,p,t,Pi,Ai,ew1);



%NORMALIZE ERRORS OF MULTIPLE OUTPUTS!!$!@$#@%@$$@#!@$#!
x = -1:0.01:1;
t1 = 100*sin(x);
t2 = 0.01*cos(x);
t = [t1;t2]; %the way to have several target elements!!!!!!!!!!
net = feedforwardnet(5);
net1 = train(net,x,t);
y = net1(x);
figure(1);
plot(x,y(1,:),x,t(1,:));
figure(2);
plot(x,y(2,:),x,t(2,:));
%now perform output error normalization!!!!!
net.performParam.normalization = 'standard'; %!!!!!!!!!!!!!!!!!!!!!
net2 = train(net,x,t);
y = net2(x);
figure(3);
plot(x,y(1,:),x,t(1,:));
figure(4);
plot(x,y(2,:),x,t(2,:));

























