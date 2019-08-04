function [b,W] = setparams(m,d)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here 
b = zeros(m,1); 
W = randn(m,d).*(1/sqrt(d)); 
end

