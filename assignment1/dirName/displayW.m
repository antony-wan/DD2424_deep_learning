function displayW(Wstar)
[K,n] = size(Wstar);
figure()
for i=1:K
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:))); 
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
    subplot(1,10,i)
    imshow(s_im{i})
end

end
