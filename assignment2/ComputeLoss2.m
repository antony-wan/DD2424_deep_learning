function loss = ComputeLoss2(X, Y, W, b)
P = EvaluateClassifier(X,W,b);
[~,ind] = max(Y);
[~,n] = size(Y);
loss = 0;
for i = 1:n
    loss = loss -log(P(ind(i),i));
end
end