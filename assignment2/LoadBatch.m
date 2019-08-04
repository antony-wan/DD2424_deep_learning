function [X, Y, y] = LoadBatch(filename)
%LOADBATCH function that reads in the data from a CIFAR-10 batch file and returns the image and label data in separate files.
%   X contains the image pixel data, has size d×N, is of type double or 
%	single and has entries between 0 and 1. N is the number of images (10000)
%	and d the dimensionality of each image (3072=32×32×3).
%	Y is K×N (K= # of labels = 10) and contains the one-hot representation 
%	of the label for each image.
%   y is a vector of length N containing the label for each image. 
%   A note of caution. CIFAR-10 encodes the labels as integers between 0-9 
%   but Matlab indexes matrices and vectors starting at 1. 
%   Therefore it may be easier to encode the labels between 1-10.
A = load(filename);
[NbImages, ~] = size(A.data);
X = ( (double(A.data))/255 )'; %conversion of A.data in double precision, then divided by 255 and finally transpose
K = 10; % # of labels
Y = zeros(10, NbImages);
y = A.labels + 1;
for idx = 1:NbImages
    Y(y(idx),idx) = 1;
end
end

