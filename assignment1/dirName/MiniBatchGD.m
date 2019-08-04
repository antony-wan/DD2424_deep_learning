function [Wstar, bstar] = MiniBatchGD(trainX,trainY,validationX, validationY, GDparams, W, b, lambda)

eta = GDparams.eta;
n_batch = GDparams.n_batch;
n_epochs = GDparams.n_epochs;
N = size(trainX,2);
train_cost = zeros(1,n_epochs);
validation_cost = zeros(1,n_epochs);

for i = 1:n_epochs
    for j = 1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        X_batch = trainX(:,inds);
        Y_batch = trainY(:,inds);
        
        P = EvaluateClassifier(X_batch, W, b);
        [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, P, W, lambda);
        
        W = W - eta * grad_W;
        b = b - eta * grad_b;
    end
    train_cost(i) = ComputeCost(trainX, trainY, W, b, lambda);
    validation_cost(i) = ComputeCost(validationX, validationY, W, b, lambda);
    
    figure(1)
    plot(1:i,train_cost(1:i))
    hold on
    plot(1:i,validation_cost(1:i))
    legend({'train','validation'},'Location','southwest')
end

Wstar = W;
bstar = b;
xlim([1,n_epochs])


end