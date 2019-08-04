function [Wstar, bstar] = MiniBatchGD3(trainX, trainY, validationX, validationY,GDparams, W, b, lambda)
n_batch = GDparams.n_batch; 
n_epochs = GDparams.n_epochs;
[~,N] = size(trainX);
n_s = 4*floor(N/n_batch); %1960
eta_min = 1e-5;
eta_max = 1e-1;
%eta = GDparams.eta;
eta = [eta_min+(0:n_s-1)*(eta_max-eta_min)/n_s eta_max-(0:n_s-1)*(eta_max-eta_min)/n_s];
eta = [eta eta eta];
k = 1;
train_loss = zeros(1,n_epochs);
validation_loss = zeros(1,n_epochs);
for i = 1:n_epochs
    train_loss(i) = ComputeLoss2(trainX, trainY, W, b);
    validation_loss(i) = ComputeLoss2(validationX, validationY, W, b);
    for j=1:N/n_batch %490 step 
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = trainX(:, inds); Ybatch = trainY(:, inds);
        P = EvaluateClassifier(Xbatch, W, b);
        [grad_b,grad_W] = ComputeGradients(Xbatch, Ybatch, P, W, b, lambda);
        W{1} = W{1} - eta(k)*grad_W{1};
        W{2} = W{2} - eta(k)*grad_W{2};
        b{1} = b{1} - eta(k)*grad_b{1};
        b{2} = b{2} - eta(k)*grad_b{2};
        k = k+1
    end
end
Wstar = W;
bstar = b;
figure(1)
plot(1:n_epochs, train_loss);
hold on
plot(1:n_epochs, validation_loss);
legend({'training','validation'})
xlabel('epoch')
ylabel('loss')
end