function [grad_b,grad_W] = ComputeGradients(X, Y, P, W, b, lambda)
%COMPUTEGRADIENTS Summary of this function goes here
%   Detailed explanation goes here
grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);
[~,nb] = size(X);

G = -(Y-P);
H = max(0, W{1}*X+b{1});
grad_W{2} = (G*H')/nb+2*lambda*W{2};
grad_b{2} = G*ones(nb,1)/nb;

G = W{2}'* G;
G = G .* (H>0);
grad_W{1} = (G*X')/nb+2*lambda*W{1};
grad_b{1} = G*ones(nb,1)/nb;
end


