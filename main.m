clear 
close all
clc
addpath(genpath('superpixel'))
addpath(genpath('nsst_toolbox'));

IRdocStruct = getAlldoc(".\dataset\ir");
VISdocStruct = getAlldoc(".\dataset\vi");
IRdocPath = IRdocStruct.folder;
IRdocNames = {IRdocStruct.name};
VISdocPath = VISdocStruct.folder;
VISdocNames = {VISdocStruct.name};

fprintf("\t Completion: ")
[m, num] = size(VISdocNames);
showTimeToCompletion;startTime=tic;
p = parfor_progress(num);
parfor i=1:length(IRdocNames)
    p = parfor_progress;
    showTimeToCompletion(p/100,[],[],startTime);

    inf_name = [IRdocPath filesep IRdocNames{i}];
    rgb_name = [VISdocPath filesep VISdocNames{i}];
    final_name = ['fused' filesep  IRdocNames{i}];
    
    image1=imread(rgb_name);% read rgb figure
    image2=imread(inf_name);% read inf figure
    
    % Convert the number of channels to grayscale
    if ndims(image1)==3
        vis=rgb2gray(image1);
    else
        vis=image1;
    end
    if ndims(image2)==3
        ir=rgb2gray(image2);
    else
        ir=image2;
    end
    % Determine whether the image size is consistent
    if size(vis)~=size(ir)
        error('two images are not the same size.');
    end
    [VIS_label,VIS_sig] =superpixel(vis);
    [IR_label,IR_sig] =  superpixel(ir);

    fused = nsst_fuse(vis,ir,double(VIS_sig),double(IR_sig),i);
    imwrite(uint8(fused),final_name);
end
disp('Done!')