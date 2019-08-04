function loss = ComputeLoss(X, Y, NetParams)
P = EvaluateClassifier(X, NetParams);
[~,ind] = max(Y);
n = size(Y,2);
loss = 0;
for i = 1:n
    loss = loss - log(P(ind(i),i));
end
end