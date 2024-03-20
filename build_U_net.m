
function out_put = build_U_net(input_Tile_Size)

%Function builds U-Net network

encoder_Depth = 4;
initial_Encoder_NumChannels = 64;
input_Numchannels = input_Tile_Size(3);
conv_Filter_Size = 3;
Up_conv_Filter_Size = 2;

layers = imageInputLayer(input_Tile_Size,'Name','ImageInputLayer');
layerIndex = 1;

% Encoder layers
for sections = 1:encoder_Depth
    
    encoder_NumChannels = initial_Encoder_NumChannels * 2^(sections-1);
        
    if sections == 1
        conv1 = convolution2dLayer(conv_Filter_Size,encoder_NumChannels,...
                                   'Padding',[1 1],'NumChannels',input_Numchannels,...
                                   'BiasL2Factor',0,'Name',['Encoder-Section-' ...
                                    num2str(sections) '-Conv-1']);
        
        conv1.Weights = sqrt(2/((conv_Filter_Size^2)*input_Numchannels*encoder_NumChannels)) ...
                             * randn(conv_Filter_Size,conv_Filter_Size,...
                             input_Numchannels,encoder_NumChannels);
    else
        conv1 = convolution2dLayer(conv_Filter_Size,encoder_NumChannels,'Padding',[1 1],...
                                   'BiasL2Factor',0,'Name',['Encoder-Section-' ...
                                   num2str(sections) '-Conv-1']);
        
        conv1.Weights = sqrt(2/((conv_Filter_Size^2)*encoder_NumChannels/2*encoder_NumChannels)) ...
                             * randn(conv_Filter_Size,conv_Filter_Size,encoder_NumChannels/2,...
                             encoder_NumChannels);
    end
    
    conv1.Bias = randn(1,1,encoder_NumChannels)*0.00001 + 1;
    
    relu1 = reluLayer('Name',['Encoder-Section-' num2str(sections) '-ReLU-1']);
    
    conv2 = convolution2dLayer(conv_Filter_Size,encoder_NumChannels,'Padding',[1 1],...
                               'BiasL2Factor',0,'Name',['Encoder-Section-' num2str(sections) ...
                               '-Conv-2']);
    
    conv2.Weights = sqrt(2/((conv_Filter_Size^2)*encoder_NumChannels)) ...
                         * randn(conv_Filter_Size,conv_Filter_Size,...
                         encoder_NumChannels,encoder_NumChannels);
                     
    conv2.Bias = randn(1,1,encoder_NumChannels)*0.00001 + 1;
    
    relu2 = reluLayer('Name',['Encoder-Section-' num2str(sections) '-ReLU-2']);
       
    layers = [layers; conv1; relu1; conv2; relu2];   
    layerIndex = layerIndex + 4;
    
    if sections == encoder_Depth
        drop_Out_Layer = dropoutLayer(0.5,'Name',['Encoder-Section-' num2str(sections) '-DropOut']);
        layers = [layers; drop_Out_Layer];
        layerIndex = layerIndex +1;
    end
    
    maxPoolLayer = maxPooling2dLayer(2, 'Stride', 2, 'Name',['Encoder-Section-' num2str(sections) '-MaxPool']);
    
    layers = [layers; maxPoolLayer];
    layerIndex = layerIndex +1;
    
end
%Middle layers
conv1 = convolution2dLayer(conv_Filter_Size,2*encoder_NumChannels,'Padding',[1 1],...
                           'BiasL2Factor',0,'Name','Mid-Conv-1');

conv1.Weights = sqrt(2/((conv_Filter_Size^2)*2*encoder_NumChannels)) ...
                     * randn(conv_Filter_Size,conv_Filter_Size,....
                     encoder_NumChannels,2*encoder_NumChannels);
                 
conv1.Bias = randn(1,1,2*encoder_NumChannels)*0.00001 + 1;

relu1 = reluLayer('Name','Mid-ReLU-1');

conv2 = convolution2dLayer(conv_Filter_Size,2*encoder_NumChannels,'Padding',[1 1],...
                           'BiasL2Factor',0,'Name','Mid-Conv-2');

conv2.Weights = sqrt(2/((conv_Filter_Size^2)*2*encoder_NumChannels)) ...
                     * randn(conv_Filter_Size,conv_Filter_Size,...
                     2*encoder_NumChannels,2*encoder_NumChannels);
                 
conv2.Bias = zeros(1,1,2*encoder_NumChannels)*0.00001 + 1;

relu2 = reluLayer('Name','Mid-ReLU-2');
layers = [layers; conv1; relu1; conv2; relu2];
layerIndex = layerIndex + 4;

%Drop-out Layer
drop_Out_Layer = dropoutLayer(0.5,'Name','Mid-DropOut');

layers = [layers; drop_Out_Layer];
layerIndex = layerIndex + 1;

initialDecoderNumChannels = encoder_NumChannels;

%Decoder layers
for sections = 1:encoder_Depth
    
    decoder_NumChannels = initialDecoderNumChannels / 2^(sections-1);
    
    upConv = transposedConv2dLayer(Up_conv_Filter_Size, decoder_NumChannels,...
                                  'Stride',2,'BiasL2Factor',0,'Name',...
                                  ['Decoder-Section-' num2str(sections) '-UpConv']);

    upConv.Weights = sqrt(2/((Up_conv_Filter_Size^2)*2*decoder_NumChannels)) ...
                          * randn(Up_conv_Filter_Size,Up_conv_Filter_Size,...
                          decoder_NumChannels,2*decoder_NumChannels);
                      
    upConv.Bias = randn(1,1,decoder_NumChannels)*0.00001 + 1;    
    
    upReLU = reluLayer('Name',['Decoder-Section-' num2str(sections) '-UpReLU']);
    
    depthConcatLayer = depthConcatenationLayer(2,'Name',...
        ['Decoder-Section-' num2str(sections) '-DepthConcatenation']);
    
    conv1 = convolution2dLayer(conv_Filter_Size,decoder_NumChannels,...
                               'Padding',[1 1],'BiasL2Factor',0,'Name',...
                               ['Decoder-Section-' num2str(sections) '-Conv-1']);
    
    conv1.Weights = sqrt(2/((conv_Filter_Size^2)*2*decoder_NumChannels)) ...
                         * randn(conv_Filter_Size,conv_Filter_Size,...
                         2*decoder_NumChannels,decoder_NumChannels);
                     
    conv1.Bias = randn(1,1,decoder_NumChannels)*0.00001 + 1;
    
    relu1 = reluLayer('Name',['Decoder-Section-' num2str(sections) '-ReLU-1']);
    
    conv2 = convolution2dLayer(conv_Filter_Size,decoder_NumChannels,'Padding',[1 1],...
                               'BiasL2Factor',0,'Name',...
                               ['Decoder-Section-' num2str(sections) '-Conv-2']);
    
    conv2.Weights = sqrt(2/((conv_Filter_Size^2)*decoder_NumChannels)) ...
                         * randn(conv_Filter_Size,conv_Filter_Size,...
                         decoder_NumChannels,decoder_NumChannels);
                     
    conv2.Bias = randn(1,1,decoder_NumChannels)*0.00001 + 1;
    
    relu2 = reluLayer('Name',['Decoder-Section-' num2str(sections) '-ReLU-2']);
    
    layers = [layers; upConv; upReLU; depthConcatLayer; conv1; relu1; conv2; relu2];
    
    layerIndex = layerIndex + 7;
end

finalConv = convolution2dLayer(1,18,'BiasL2Factor',0,'Name',...
                               'Final-ConvolutionLayer');

finalConv.Weights = randn(1,1,decoder_NumChannels,18);
finalConv.Bias = randn(1,1,18)*0.00001 + 1;

soft_Max_Layer = softmaxLayer('Name','Softmax-Layer');

pixelClassLayer = pixelClassificationLayer('Name','Segmentation-Layer'); 

layers = [layers; finalConv; soft_Max_Layer; pixelClassLayer];

% Create the layer graph and create connections in the graph
out_put = layerGraph(layers);

% Connect layers for U-Net structure
out_put = connectLayers(out_put, 'Encoder-Section-1-ReLU-2','Decoder-Section-4-DepthConcatenation/in2');
out_put = connectLayers(out_put, 'Encoder-Section-2-ReLU-2','Decoder-Section-3-DepthConcatenation/in2');
out_put = connectLayers(out_put, 'Encoder-Section-3-ReLU-2','Decoder-Section-2-DepthConcatenation/in2');
out_put = connectLayers(out_put, 'Encoder-Section-4-DropOut','Decoder-Section-1-DepthConcatenation/in2');
end