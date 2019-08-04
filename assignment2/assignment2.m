%% Load files
DirData = 'Datasets/cifar-10-batches-mat';
addpath(DirData)
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[validationX, validationY, validationy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');

%% Preprocessing
mean_X = mean(trainX, 2); 
std_X = std(trainX, 0, 2);
trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]); 
trainX = trainX ./ repmat(std_X, [1, size(trainX, 2)]);

mean_X = mean(validationX, 2); 
std_X = std(validationX, 0, 2);
validationX = validationX - repmat(mean_X, [1, size(validationX, 2)]); 
validationX = validationX ./ repmat(std_X, [1, size(validationX, 2)]);

mean_X = mean(testX, 2); 
std_X = std(testX, 0, 2);
testX = testX - repmat(mean_X, [1, size(testX, 2)]); 
testX = testX ./ repmat(std_X, [1, size(testX, 2)]);

%% Initialization
m = 50; K = 10; 
[d,~] = size(trainX);
lambda = 0.01;

%% i) Gradient test
% W{1} = W{1}(:,1:20);
% P = EvaluateClassifier(trainX(1:20, 1:2), W, b);
% [grad_b,grad_W] = ComputeGradients(trainX(1:20, 1:2), trainY(:, 1:2), P, W, b, lambda);
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(1:20, 1:2), trainY(:, 1:2), W, b, lambda, 1e-5);
% [mgrad_b, mgrad_W] = ComputeGradsNum(trainX(1:20, 1:2), trainY(:, 1:2), W, b, lambda, 1e-5);
% diffb{1} = max(abs(grad_b{1} - ngrad_b{1}))/max(eps, max(abs(grad_b{1})+abs(ngrad_b{1})));
% diffb{2} = max(abs(grad_b{2} - ngrad_b{2}))/max(eps, max(abs(grad_b{2})+abs(ngrad_b{2})));
% diffW{1} = max(abs(grad_W{1} - ngrad_W{1}))/max(eps, max(abs(grad_W{1})+abs(ngrad_W{1})));
% diffW{2} = max(abs(grad_W{2} - ngrad_W{2}))/max(eps, max(abs(grad_W{2})+abs(ngrad_W{2})));

%% ii) Network evaluation
[GDparams] = setGDparams(100,10,0.01); %n_batch,n_epochs,eta
[b{1},W{1}] = setparams(m,d);
[b{2},W{2}] = setparams(K,m);
[Wstar, bstar] = MiniBatchGD(trainX, trainY, validationX, validationY, trainy, validationy, GDparams, W, b, lambda);
acc = ComputeAccuracy(testX, testy, Wstar, bstar);


%% iii) Lambda search

% [trainX1, trainY1, trainy1] = LoadBatch('data_batch_1.mat');
% [trainX2, trainY2, trainy2] = LoadBatch('data_batch_2.mat');
% [trainX3, trainY3, trainy3] = LoadBatch('data_batch_3.mat');
% [trainX4, trainY4, trainy4] = LoadBatch('data_batch_4.mat');
% [trainX5, trainY5, trainy5] = LoadBatch('data_batch_5.mat');
% trainX = [trainX1 trainX2 trainX3 trainX4 trainX5(:,1:9000)];
% trainY = [trainY1 trainY2 trainY3 trainY4 trainY5(:,1:9000)];
% trainy = [trainy1; trainy2 ;trainy3 ;trainy4 ;trainy5(1:9000)];
% validationX = trainX5(:,9001:10000);
% validationY = trainY5(:,9001:10000);
% validationy = trainy5(9001:10000);
% [testX, testY, testy] = LoadBatch('test_batch.mat');

%% Initialization
% lmin = log(0.000027166787899)/log(10);
% lmax = log(0.000040913452224)/log(10);
% n_sample = 10;
% l = lmin + (lmax - lmin)*rand(1, n_sample);
% lambda = 10.^l;
% lambda = sort(lambda);
%acc = zeros(1,n_sample);
% lambda = 3.4149648025e-5;
% [GDparams] = setGDparams(100,24,0.01);
% [d,~] = size(trainX);
% m = 50; K = 10;
% 
% %initialization de W et b
% [b,W] = setparams(m,K,d);
% [Wstar, bstar] = MiniBatchGD3(trainX, trainY, validationX, validationY,GDparams, W, b, lambda);
% acc = ComputeAccuracy(testX, testy, Wstar, bstar);














