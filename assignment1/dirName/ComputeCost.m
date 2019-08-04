function J = ComputeCost(X, Y, W, b, lambda)
%COMPUTECOST computes the cost function given by equation (5) for a set of images.
%   Detailed explanation goes here
[~,n] = size(X);
P = EvaluateClassifier(X, W, b);

loss = (1/n) * sum( -log(diag(Y'*P)));
regularization = lambda*sum(W.^2,'all');

J = loss + regularization;
end

