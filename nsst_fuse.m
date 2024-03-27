function xr=nsst_fuse(vis,ir,I1_m,I2_m,ai)
    %    Input:
    %    I1 - input image A
    %    I2 - input image B
    %    I1_m - input image A superpixel
    %    I2_m - input image B superpixel
    %    nlevels - number of directions in each decomposition level
    %    Output:
    %    y  - fused image  
    deep=0;
    % setup parameters for shearlet transform
    lpfilt= 'pyrexc';
    % .dcomp(i) indicates there will be 2^dcomp(i) directions 
    shear_parameters.dcomp =[3 3 3 4];
    shear_parameters.dsize =[8 4 8 16];
    [dst_vis,shear_f_vis]=nsst_dec1e(vis,shear_parameters,lpfilt);
    [dst_ir,shear_f_ir]=nsst_dec1e(ir,shear_parameters,lpfilt);
    %% detail layers fusion
    y = dst_ir;
    for i=2:numel(shear_parameters.dcomp)+1
        for j=1:(2^(shear_parameters.dcomp(i-1)))
            E1 = abs(dst_vis{i}(:,:,j));
            E2 = abs(dst_ir{i}(:,:,j));
            um = 3;
            A1 = ordfilt2(abs(es2(E1,floor(um/2))),um*um, ones(um));
            A2 = ordfilt2(abs(es2(E2,floor(um/2))),um*um, ones(um));
            map = (conv2(double(A1>A2),ones(um),'valid'))>floor(um*um/2);
            temp_y = y{i}(:,:,j);
            temp_vis = dst_vis{i}(:,:,j);
            temp_y(map)=temp_vis(map);
            y{i}(:,:,j)=temp_y;
        end
    end
   
    %% base layers fusion
    if deep==1
        imagename = strcat('.\SFCFusionDeepfuse\test_result\',num2str(ai),'.png');
        y{1,1}=double(imread(imagename)); 
    else
        locate ='.\deep';
        baseIRsavename = strcat(locate,'\baIR', num2str(ai),'.jpg');
        baseVISsavename = strcat(locate,'\baVI', num2str(ai),'.jpg'); 
        saIRsavename =  strcat(locate,'\saIR', num2str(ai),'.jpg');
        saVISsavename =  strcat(locate,'\saVI', num2str(ai),'.jpg');
        srIRsavename =  strcat(locate,'\srIR', num2str(ai),'.jpg');
        srVISsavename =  strcat(locate,'\srVI', num2str(ai),'.jpg');
        imwrite(uint8(dst_vis{1,1}),baseVISsavename)
        imwrite(uint8(dst_ir{1,1}),baseIRsavename)
        imwrite(uint8(I1_m(:,:,1)),saVISsavename)
        imwrite(uint8(I2_m(:,:,1)),saIRsavename)
        imwrite(uint8(vis),srVISsavename)
        imwrite(uint8(ir),srIRsavename)

        I1 = double(vis);
        I2 = double(ir);
        I1_d = ImgStandardDeviation(I1);
        I2_d = ImgStandardDeviation(I2);
        I1_m_d = ImgStandardDeviation(I1_m);
        I2_m_d = ImgStandardDeviation(I2_m);
        Iz = min(I1_m_d,I2_m_d)/(I1_m_d + I2_m_d)*max(I1_m,I2_m) + max(I1_m_d,I2_m_d)/(I1_m_d + I2_m_d)*min(I1_m,I2_m);
        Izf = (Iz + repmat((min(dst_vis{1,1},dst_ir{1,1})./max(dst_vis{1,1},dst_ir{1,1})).*(max(dst_vis{1,1},dst_ir{1,1})-min(dst_vis{1,1},dst_ir{1,1})),[1 1 3]));
        W = 1+ (sqrt(I1_m_d+I2_m_d) - sqrt(I1_d+I2_d))/(sqrt(I1_d + I2_d));
        W1 = W*(0.5+ (I1_m_d - I2_m_d)/(I1_d + I2_d));
        W2 = W*(0.5+ (I2_m_d - I1_m_d)/(I1_d + I2_d));
        If = Izf + W1*I1/W + W2*I2/W -W1*(I1_m) - W2*(I2_m);
        y{1,1}=If(:,:,1); 
    end
    %% reconstruct the image from the shearlet coefficients
    xr=nsst_rec1(y,lpfilt);  
end