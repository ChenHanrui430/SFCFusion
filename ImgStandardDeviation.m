function standard_deviation = ImgStandardDeviation(img)
    % 获取图像标准差
    if ndims(img)==2
        %灰度图
        I = double(img);
        % 方差
%         var_ = var(I(:));
        % 标准差
        standard_deviation = std2(I(:));
    elseif ndims(img)==3
        R = double(img(:,:,1));
        G = double(img(:,:,2));
        B = double(img(:,:,3));
        standard_deviation = sum(std2(R)+std2(G)+std2(B))/3.0;
    end
end