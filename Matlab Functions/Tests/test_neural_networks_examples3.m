%Image category classification using deep learning:
funciton DeepLearningImageClassificationExample

%get GPU device information:
deviceInfo = gpuDevice;

%check the GPU compute capability:
computeCapability = str2double(deviceInfo.ComputeCapability);
assert(computeCapability >= 3, 'this example requires a gpu device with compute capability 3 or higher');

%the category classifier will be trained on images from Caltech101, which
%is one of the most widely cited and used image datasets.
%download the compressed dataset from the following location:
url = 'http://www.vision.caltech.eduImage_Datasets/Caltech101/101_ObjectCategories.tar.gz';
%store the output in a temporary folder:
outputFolder = fullfile(tempdir, 'caltech101'); %define output folder
if ~exist(outputFolder,'dir')
   %disp('Downloading 126MB caltech101 data set');
   untar(url,outputFolder);
end

%instead of operating on all of caltech101, which is time consuming, use
%three of the categories and the image category classifier will be trained
%to distinguish amonst these:
rootFolder = fullfile(outputFolder, '101_ObjectCategories');
categories = {'airplanes','ferry','laptop'};

%Create an ImageDatastore to help you manage the data. Because
%ImageDatastore operates on image file locations, images are not loaded
%into memory until read, making it efficient for use with large image
%collections.
imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tbl = countEachLabel(imds)
%because imds above contains an unequal nuber of images per category, let's
%first adjust it, so that the number of images in the training set is
%balanced:
miniSetCount = min(tbl{:,2}); %determine the smallest amount of images in a category
%Use splitEachLabel method to trim the set:
imds = splitEachLabel(imds,minSetCount,'randomize');
%Notice that each set now has exactly the same number of images:
countEachLabel(imds) %one can see that the dataset has been trimmed to fit the lowest number of labels

%fIND THE FIRST INSTANCE OF AN IMAGE FOR EACH CATEGORY:
airplanes = find(imds.labels == 'airplanes',1);
ferry = find(imds.Labels == 'ferry',1);
laptop = find(imds.Labels == 'laptop',1);
figure;
subplot(1,3,1);
imshow(readimage(imds,airplanes)); %READIMAGE !@#!#!#!#
subplot(1,3,2);
imshow(readimage(imds,ferry));
subplot(1,3,3);
imshow(readimage(imds,laptop));




%LOAD PRE-TRAINED ALEXNET NETWORK:
%load alexnet:
net = alexnet()
%other popular networks trained on imagenet include vgg-16 and vgg-19 which
%can be loaded using vgg16 and vgg19 from the neural network toolbox
%view the CNN architecture:
net.Layers
%Inspect the first layer:
net.Layers(1)
%Inspect the last layer:
net.Layers(end)
%number of class names for imagenet classification task:
numel(net.Layers(end).ClassNames)

%Pre=process images for CNN:
%as mentioned above, the net can only process RGB images that are 227X227.
%to avoid resaving all the images in caltech 101 to this format, setup the
%imds read function, imds.ReadFcn, to pre-process image on-the-fly. the
%imds.ReadFcn is called every time an image is read from the ImageDatastore!!!!!:
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
%note that other CNN models will have different input size constraints, and
%may require other pre-processing steps:
% function Iout = readAndPreprocessImage(filename)
%     I = imread(filename);
%     
%     %some images may be grayscale.:
%     if ismatrix(I)
%        I = cat(3,I,I,I); 
%     end
%     
%     %Resize the image as required for the CNN:
%     Iout = imresize(I, [227,227]);
%     
%     %the that the aspect ratio is not preserved. in caltech 101, the object
%     %of interest is centered in the image and occupies a majority of the
%     %image scene. therefore, preerving the aspect ratio is not critical.
%     %however, for other data sets, it may prove beneficial to preserve the
%     %aspect ratio of the original image when resizing.
% end


%Prepare training and test image sets:
[trainingSet,testSet] = splitEachLabel(imds, 0.3, 'randomize');
w1 = net.Layers(2).Weights;
%scale and resize the weights for visualization:
w1 = mat2gray(w1);
w1 = imresize(w1,5);
%display a montage of network weights. there are 96 individual sets of
%weights in the first layer:
figure
montage(w1);
title('first convolutional layer weights');
%extract from layer before classification:
featureLayer = 'fc7';
trainingFeatures = activations(net, trainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns'); %$!$@#$@#$

%Train a multiclass SVM classifier using CNN features:
%get training labels from the trainingSet:
trainingLabels = trainingSet.Labels;
%train multiclass SVM classifier using a fast linear solver:
%#@$#$#@$ fitcecoc????
classifier = fitcecoc(trainingFeatures,trainingLabels, 'Learners','Linear','Coding','onevsall','ObservationIn','columns');
%evaluate classifier:
%extract test features using the CNN:
testFeatures = activations(net,testSet,featureLayer,'MiniBatchSize',32);
%pass CNN image features to trained classifier:
predictedLabels = predict(classifier,testFeatures);
%get the known labels:
testLabels = testSet.Labels;
%tabulate the results using a confusion matrix:
confMat = confusionmat(testLabels, predictedLabels);
%Convert confusion matrix into percentage form:
confMat = bsxfun(@rdivide,confMaat,sum(confMat,2))
%display the mean accuracy:
mean(diag(confMat))

%Try the newly trained classifier on test images:
newImage = fullfile(rootFolder, 'airplanes', 'image_0690.jpg');
%pre-process the images as required for the CNN:
img = readAndPreprocessImage(newImage);
%extract image features using the CNN:
imageFeatures = activations(net,img,featureLayer);
%Make a prediction using the classifier:
label = predict(classifier, imageFeatures)

























