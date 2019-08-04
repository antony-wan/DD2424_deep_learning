function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    
[~,n] = size(X);
G = -(Y-P);
grad_W = G*X'/n;
grad_b = G*ones(n,1)/n;

grad_W = grad_W + 2*lambda*W;

end