%DOWNLOAD CIFAR-10 data to a remporary directory:
cifar10Data = tempdir;
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url, cifar10Data);
%Load the CIFAR-10 training and test data:
[trainingImages, trainingLabels, testImages, testLabels] = helperCIFAR10Data.load(cifar10Data);
%each image is a 32X32 RGB(3 channels) image and there are 50,000 training samples:
size(trainingImages)
%CIFAR-10 has 10 image categories. let's list them:
numImageCategories = 10;
categories(trainingLabels);
%display a few of the training images, resizing them for display:
figure;
thumbnails = trainingImages(:,:,:,1:100);
thumbnails = imresize(thumbnails, [64,64]);
montage(thumbnails);

%EXAMPLE LAYERS:
% imageInputLayer = image input layer
% convolutional2dLayer = 2D convolution layer for Convolutional Neural Networks
% reluLayer = Rectified linear unit (ReLU) layer
% maxPooling2dLayer = max pooling layer
% fullyConnectedLayer = Fully connected layer
% softmaxLayer = Softmax layer
% classificationLayer = Classification output layer for a neural network

%create the image input layer for 32X32X3 CIFAR-10 images:
[height,width,numChannels,~] = size(trainingImages);
imageSize = [height,width,numChannels];
inputLyaer = imageInputLayer(imageSize);

%the middle layers are made up of repeated blocks of convolutional, ReLU
%(rectified linear units), and pooling layers. these 3 layers form the core
%building blocks of convolutional neural networks!!!!.
%the convolutional layers define sets of filter weights, which are updated
%during network training. 
%the ReLU layer adds non-linearity to the network, which allow the network
%to approximate non-linear functions that map image pixels to the semantic
%content of the image.
%the pooling layers downsample data as it flows through the network. in a
%network with lots of layers, pooling layers should be used sparingly to
%avoid downsampling the data too early in the network:

%Convolution layer parameters:
filterSize = [5,5]; %2D filter size
numFilters = 32; %number of filters in the layer

middleLayers = [ 
    
%the first convolutional layer has a bank of 32 5X5X3 filters as stated
%above. a symmetric padding of 2 pixels is added to ensure that image
%borders are included in the processing. this is important to avoid
%information at the borders being washed away too early in the network.
convolution2dLayer(filterSize, numFilters, 'Padding', 2)

%note that the third dimension of the filter can be omitted because it is
%automatically deduced based on the connectivity of the network!!!!!!!.
%in this case because this layer follows the image layer, the third
%dimension must be 3 to match the number of channels in the input image.

%Next add the ReLU layer:
reluLayer()

%follow it with a max pooling layer that has a 3X3 spatial pooling area and
%a stride of 2 pixels. this downsamples the data dimensions from 32X32 to 15X15.
maxPooling2dLayer(3, 'Stride', 2)

%Repeat the 3 core layers to complete the middle of the network:
convolution2dLayer(filterSize, numFilters, 'Padding', 2);
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

convolution2dLayer(filterSize, 2*numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride', 2)

];




%the final layers of a CNN are typically composed of fully connected layers
%and a softmax loss layer:
finalLayer = [
    
%add a fully connected layer with 64 output neorons. the output size of
%this layer will be an array with a length of 64.
fullyConnectedLayer(64)

%Add a ReLU non-linearity:
reluLayer

%Add the last fully connected layer. at this point, the network must
%produce 10 signals that can be used to measure whether the input image
%belongs to one category or another. this measurement is made using the
%subsequent loss layers.
fullyConnectedLayer(numImageCategories)

%Add the softmax loss layer and classification layer. the final layers use
%the output of the fully connected layer to compute the categorical
%probability distribution over the image classes. during the training
%process, all the network weights are tuned to minimize the loss over this
%categorical distribution 
softmaxLayer
classificationLayer %cross-entropy !!!!!
]


%NOW COMBINE THE input, middle and final layers:
layers = [
 inputLayer
 middleLayers
 finalLayers
]

%Initialize the FIRST convolution layer weights using normally distributed
%random numbers with standard deviation of 0.0001. this helps improve the
%convergence of training.
layers(2).Weights = 0.0001 * randn([filterSize, numChannels,numFilters]);


%Set the network training options:
opts = trainingOptions(...
    'sgdm',...
    'Momentum',0.9,...
    'InitialLearnRate', 0.001,...
    'LearnRateSchedule', 'piecewise',... %?!@#?!@?#!
    'LearnRateDropFactor', 0.1,... %!?#?#!@?$@!?$
    'LearnRateDropPeriod', 8, ... %!?@#?!#?@!?#
    'L2Regularization', 0.004,...
    'MaxEpochs', 40,... %EPOCH = 1 complete pass through all the training data
    'MiniBatchSize', 128,...
    'Verbose', true); %!@?!?#!@?

%a trained network is loaded from disk to save time when running the
%example. set this flag to true to train the network:
doTraining = false;
if doTraining
   %Train a network:
   cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opt);
else
   %Load pre-trained detector for the example
   load('rcnnStopSigns.mat', 'cifar10Net');
end


%After the network is trained, it should be validated to ensure that
%training was successful. first, a quick visualization of the first
%convolutional layer's filter weights can help identify any immediate
%issues with training.
%Extract the first convolutional layer weights
w = cifar10Net.Layers(2).Weights;
%rescale and resize the weights for better visualization
w = mat2gray(w);
w = imresize(w, [100,100]);
figure;
montage(w);

%THE FIRST LAYER WEIGHTS SHOULD HAVE SOME WELL DEFINED STRUCTURE!.
%if the weights still look random, then that is an indication that the
%network may require additional training. in this case, as shown above, the
%first layer filters have learned edge-like features from the CIFAR-10
%training data.

%Run the network on the test set.
YTest = classify(cifar10Net, testImages);
%Calculate the accuracy.
accuracy = sum(YTest == testLabels)/numel(testLabels)


%LOAD TRAINING DATA:
%now that the network is working well for the CIFAR-10 classification task,
%the transfer learning approach can be used to fine-tune the network for
%stop sign detection.
%start by loading the ground truth data for stop signs.

% Load the ground truth data:
data = load('stopSignsAndCars.mat','stopSignsAndCars');
stopSignsAndCars = data.stopSignsAndCars;
%update the path to the image files to match the local file system:
visiondata = fullfile(toolboxdir('vision'),'visiondata');
stopSignsAndCars.imageFilename = fullfile(visiondata,stopSignsAndCars.imageFilename); %?!#?!@?#?!@#
%display a summary of the ground truth data:
summary(stopSignsAndCars)
%only keep the image file names and the stop sign ROI labels:
stopSigns = stopSignsAndCars(:, {'imageFilename','stopSign'});
%Display one trainig image and he ground truth bounding boxes:
I = imread(stopSigns.imageFilename{1});
I = insertObjectAnnotation(I, 'Rectangle', stopSigns.stopSign{1},'stop sign','LineWidth',8); %!$!?#!@?#!@
figure;
imshow(I);

%TRAIN R-CNN stop sign detector.
%during training, the input network weights are fine tuned using image
%patches extracted from the ground truth data. the 'PositiveOverlapRange'
%and 'NegativeOverlapRange' parameters control which image patches are used
%for training.
%positive training samples are those that overlap with the ground truth
%boxed by 0.5 to 1, as measured by the bounding box intersection over union
%metric. negative training samples are those that overlap by 0 to 0.3. the
%best values for these parameters should be chosen by testing the trained
%detector on a validation set. 
doTraining = false;
if doTraining
    %set training options:
    options = trainingOptions(...
        'sgdm',...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3,...
        'LearnRateSchedule', 'piecewise',...
        'LearnRateDropFactor', 0.1,...
        'LearnRateDropPeriod',100,...
        'MaxEpochs',100,...
        'Verbose',true);
    
    %Train an R-CNN object detector. this will take several minutes.
    rcnn = trainRCNNObjectDetector(stopSigns, cifar10Net, ...
        options, 'NegativeOverlapRange',[0,0.3],'PositiveOverlapRange',[0.5,1]);
else
    %Load pre-trained network for the example:
    load('rcnnStopSigns.mat','rcnn');
end


%Test R-CNN stop sign detector:
%read test image:
testImage = imread('stopSignTest.jpg');
%detect stop signs:
[bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128); %why is minibatchsize relevant here!@!?#!@?#?!#

%extract the activations from the softmax layer. these are the
%the trained network is stored within the RCNN detector:
rcnn.Network
%classification scores produced by the network as it scans the image:
featureMap = activations(rcnn.Network,testImage,'softmax','OutputAs','channels');
%the softmax activations are stored in a 3D array:
size(featureMap);
%the 3-rd dimension in featureMap corresponds to the object classes:
rcnn.ClassNames
%the stop sign feature map is stored in the first channel:
stopSignMap = featureMap(:,:,1);
%THE SIZE OF THE ACTIVATIONS OUTPUT IS SMALLER THAN THE INPUT IMAGE DUE TO
%THE DOWNSAMPLING OPERATIONS IN THE NETWORK. TO GENERATE A NICER
%VISUALIZATION, RESIZE STOPSIGNMAP TO THE SIZE OF THE INPUT IMAGE:
%resize stopSignMap for visualization:
[height,width,~] = size(testImage);
stopSignMap = imresize(stopSignMap, [height,width]);
%visualize the feature map superimposed on the test image:
featureMapOnImage = imfuse(testImage, stopSignMap); %@$#@$@#$@$ IMFUSE!!@#@!#!
figure;
imshow(featureMapOnImage);











































