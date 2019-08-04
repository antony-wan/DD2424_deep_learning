function acc = ComputeAccuracy(X, y, W, b)
%COMPUTEACCURACY computes the accuracy of the network’s predictions given by equation (4) on a set of data
%   Detailed explanation goes here
P = EvaluateClassifier(X, W, b);
[maxvalues,ind] = max(P,[],1);
acc = sum(ind==y')/length(y)*100;
end

