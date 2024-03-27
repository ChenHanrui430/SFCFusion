% chr 2023.2.7
function [label, outputImage]=superpixel(inputimage)
    image = im2uint8(inputimage);
    if ndims(image) ~= 3
        image = repmat(image,[1 1 3]);
    end
    gaus = fspecial('gaussian',3);
    image_filtered = imfilter(image,gaus);
    superpixelNum = 800;
    ratio = 0.035;
    label = su(image_filtered,superpixelNum,ratio);
    [numRows ,numCols]=size(label); 
    idx = label2idx(label);
    [~, N] = size(idx);
    outputImage = zeros(size(image),'like',image);
    for labelVal = 1:N
        redIdx = idx{labelVal};
        greenIdx = idx{labelVal}+numRows*numCols;
        blueIdx = idx{labelVal}+2*numRows*numCols;
        outputImage(redIdx) = mean(image(redIdx));
        outputImage(greenIdx) = mean(image(greenIdx));
        outputImage(blueIdx) = mean(image(blueIdx));
    end       
end