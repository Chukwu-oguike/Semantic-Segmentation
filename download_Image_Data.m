function download_Image_Data(url, destination)
%Function downloads multispectral image data used in training a U-Net
%network

filename = 'rit18_data.mat';
image_Dir_FullPath = fullfile(destination,'rit18_data');
image_File_FullPath = fullfile(image_Dir_FullPath,filename);

if ~exist(image_File_FullPath,'file')
    mkdir(image_Dir_FullPath);
    websave(image_File_FullPath,url);
end
end