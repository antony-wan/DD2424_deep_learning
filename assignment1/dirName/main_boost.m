addpath Datasets/cifar-10-batches-mat/;
filename1 = 'data_batch_1.mat';
filename2 = 'data_batch_2.mat';
filename3 = 'data_batch_3.mat';
filename4 = 'data_batch_4.mat';
filename5 = 'data_batch_5.mat';

filetest = 'test_batch.mat';

[trainX1, trainY1, trainy1] = LoadBatch(filename1);
[trainX2, trainY2, trainy2] = LoadBatch(filename2);
[trainX3, trainY3, trainy3] = LoadBatch(filename3);
[trainX4, trainY4, trainy4] = LoadBatch(filename4);
[trainXint, trainYint, trainyint] = LoadBatch(filename5);

trainX5 = trainXint(1:9000);
trainY5 = trainYint(1:9000);
trainy5 = trainyint(1:9000);

trainX = [trainX1 trainX2 trainX3 trainX4 trainX5];

validationX = trainXint(9001:end);
validationY = trainYint(9001:end);
validationy = trainyint(9001:end);

[testX, testY, testy] = LoadBatch(filetest);

[d,N] = size(trainX);
K = size(trainY,1);

%% Initialization

std = 0.01;
W = randn(K,d)*std;
b = randn(K,1)*std;


%% Gradient test

%Comparison of the gradient computed in an efficient way with the numerical
%way

%{
lambda = 0; %test without regularization 
P = EvaluateClassifier(trainX(:, 1), W, b);
[grad_W, grad_b] = ComputeGradients(trainX(:,1), trainY(:,1), P, W, lambda);
[ngrad_b, ngrad_W] = ComputeGradsNum(trainX(:,1), trainY(:,1), W, b, lambda, 1e-6);

diffGrad.b = abs( (grad_b - ngrad_b) )./max(eps, abs(grad_b) + abs(ngrad_b)) ;
diffGrad.W = abs( (grad_W - ngrad_W) )./max(eps, abs(grad_W) + abs(ngrad_W)) ;

max(diffGrad.b, [], 'all')
max(diffGrad.W, [], 'all')
%}

GDparams = initParam(100, 0.01, 40); %n_batch, eta, n_epochs%

lambda = 0;

[Wstar, bstar] = MiniBatchGD(trainX, trainY, validationX, validationY, GDparams, W, b, lambda);

disp("accuracy = ");
acc = ComputeAccuracy(testX, testy, Wstar, bstar)
