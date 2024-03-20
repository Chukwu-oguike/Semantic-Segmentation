  %download training and validation data
image_Dir = tempdir;
url = 'http://www.cis.rit.edu/~rmk6217/rit18_data.mat';
download_Image_Data(url,image_Dir);

  %load data with matlab
%load(fullfile(imageDir,'rit18_data','rit18_data.mat'));

  %data matrix
%whos train_data val_data test_data

train_data = switch_Channels_Z_plane(train_data);
val_data   = switch_Channels_Z_plane(val_data);
test_data  = switch_Channels_Z_plane(test_data);

  %create vector of names for pixel classes
classNames = [ "RoadMarkings","Tree","Building","Vehicle","Person", ...
               "LifeguardChair","PicnicTable","BlackWoodPanel",...
               "WhiteWoodPanel","OrangeLandingPad","Buoy","Rocks",...
               "LowLevelVegetation","Grass_Lawn","Sand_Beach",...
               "Water_Lake","Water_Pond","Asphalt"];

  %Overlay the labels on the histogram-equalized RGB training image
%cmap = jet(numel(classNames));
%B = labeloverlay(histeq(train_data(:,:,4:6)),train_labels,'Transparency',0.8,'Colormap',cmap);


%Save the training data as a MAT file and the training labels as a PNG file
save('train_data.mat','train_data');
imwrite(train_labels,'train_labels.png');


%Here a random patch data extraction was used to collate data for network

%store images from training data in a datastore
imds = imageDatastore('train_data.mat','FileExtensions','.mat','ReadFcn',@matReader);

%Create a pixelLabelDatastore to store the label patches containing the 18 labeled region
pixel_Label_Ids = 1:18;
pxds = pixelLabelDatastore('train_labels.png',classNames,pixel_Label_Ids);

%create a random patch extraction datastore from image and pixel datastores
dsTrain = randomPatchExtractionDatastore(imds,pxds,[256,256],'PatchesPerImage',16000);

  %display current datastore
%inputBatch = preview(dsTrain);
%disp(inputBatch)


%build U-Net
input_Tile_Size = [256,256,6];
U_net = build_Unet(input_Tile_Size);
%disp(U_net.Layers)


%Set training options

initialLearningRate = 0.05;
maxEpochs = 150;
minibatchSize = 16;
l2reg = 0.0001;

options = trainingOptions('sgdm','InitialLearnRate',initialLearningRate,...
                          'Momentum',0.9,'L2Regularization',l2reg,...
                          'MaxEpochs',maxEpochs,'MiniBatchSize',minibatchSize,...
                          'LearnRateSchedule','piecewise','Shuffle','every-epoch',...
                          'GradientThresholdMethod','l2norm','GradientThreshold',0.05, ...
                          'Plots','training-progress', 'VerboseFrequency',20);

%Training network (training usually takes 24 hrs+)
modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
[net,info] = trainNetwork(dsTrain,U_net,options);
save(['multispectralUnet-' modelDateTime '-Epoch-' num2str(maxEpochs) '.mat'],...
     'net','options');


%Forward pass on the trained network with validation data set
predict_Patch_Size = [1024 1024];
segmented_Image = segment_Image(val_data,net,predict_Patch_Size);

%extract valid portion of segmentation and multiply the segmented image by
%the mask channel of the validation data
segmented_Image = uint8(val_data(:,:,7)~=0) .* segmented_Image;

%figure
%imshow(segmentedImage,[])

%Filter noise from segmentation
segmented_Image = medfilt2(segmented_Image,[7,7]);
%imshow(segmentedImage,[]);
%title('Segmented Image  with Noise Removed')


%{
B = labeloverlay(histeq(val_data(:,:,[3 2 1])),segmented_Image,...
                'Transparency',0.8,'Colormap',cmap);
figure
imshow(B)
title('Labeled Validation Image')
colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,'TickLength',0,'TickL%abelInterpreter','none');
colormap(cmap)
%}


%Save the segmented image and ground truth labels as PNG files
imwrite(segmented_Image,'results.png');
imwrite(val_labels,'gtruth.png');

%PixelLabelDatastore for the segmentation results and the ground truth labels
pxds_Results = pixelLabelDatastore('results.png',classNames,pixel_Label_Ids);
pxds_Truth = pixelLabelDatastore('gtruth.png',classNames,pixel_Label_Ids);

%Display the accuracy of the semantic segmentation
ssm = evaluateSemanticSegmentation(pxds_Results,pxds_Truth,'Metrics','global-accuracy');

%Calculate the percentage of vegetation cover
vegetation_Class_Ids = uint8([2,13,14]);
vegetation_Pixels = ismember(segmented_Image(:),vegetation_Class_Ids);
validPixels = (segmented_Image~=0);

num_Vegetation_Pixels = sum(vegetation_Pixels(:));
num_Valid_Pixels = sum(validPixels(:));


%Display the calculated percentage
percentVegetationCover = (num_Vegetation_Pixels/num_Valid_Pixels)*100;
fprintf('The percentage of vegetation cover is %3.2f%%.',percentVegetationCover);












