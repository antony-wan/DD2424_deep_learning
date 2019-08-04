function J = ComputeCost(X, Y, W, b, lambda)
%COMPUTECOST computes the cost function given by equation (5) for a set of images.
%   Detailed explanation goes here
[~,n] = size(X);
P = EvaluateClassifier(X,W,b);
W1 = W{1};
W2 = W{2};
loss = (1/n) * sum( -log(diag(Y'*P)));
regularization = lambda*( sum(W1.^2,'all') + sum(W2.^2,'all') );

J = loss + regularization;
end