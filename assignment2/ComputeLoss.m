function loss = ComputeLoss(X, Y, W, b)
P = EvaluateClassifier(X,W,b);
loss = sum(-log(diag(Y'*P)));
end