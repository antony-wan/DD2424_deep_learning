function P = EvaluateClassifier(X,W,b)
%EVALUATECLASSIFIER evaluates the network function, i.e. equations (1, 2), on multiple images and returns the results.
S1 = W{1}*X+b{1};
H = max(S1,0);
S = W{2}*H+b{2};
P = exp(S)./sum(exp(S),1);
end


