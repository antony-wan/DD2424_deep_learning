addpath Datasets/cifar-10-batches-mat/;
filename1 = 'data_batch_1.mat';
filename2 = 'data_batch_2.mat';
filename3 = 'test_batch.mat';

[trainX, trainY, trainy] = LoadBatch(filename1);
[validationX, validationY, validationy] = LoadBatch(filename2);
[testX, testY, testy] = LoadBatch(filename3);

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


