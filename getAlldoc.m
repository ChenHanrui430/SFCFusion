function AllFile= getAlldoc(inputdir)
    if ~exist('inputdir')
        AidDir = uigetdir(); 	% 通过交互的方式选择一个文件夹
        if AidDir == 0 			% 用户取消选择
            fprintf('Please Select a New Folder!\n');
        else
            workpath = pwd;
            cd(AidDir)
            RawFile = dir('**/*.*'); %主要是这个结构，可以提取所有文件
            AllFile = RawFile([RawFile.isdir]==0);
            if isempty(fieldnames(AllFile))
	            fprintf('There are no files in this folder!\n');
            else	% 当前文件夹下有文件，反馈文件数量
	            fprintf('Number of Files: %i \n',size(AllFile,1));
            end
            cd(workpath);
        end
    else
        AidDir = inputdir;
        workpath = pwd;
        cd(AidDir)
        RawFile = dir('**/*.*'); %主要是这个结构，可以提取所有文件
        AllFile = RawFile([RawFile.isdir]==0);
        if isempty(fieldnames(AllFile))
            fprintf('There are no files in this folder!\n');
        else	% 当前文件夹下有文件，反馈文件数量
            fprintf('Number of Files: %i \n',size(AllFile,1));
        end
        cd(workpath);
end
