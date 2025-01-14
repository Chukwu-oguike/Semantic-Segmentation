function out = segment_Image(im, net, patchSize)
%This Function performs segmentation on image patches using the semanticseg function

[height, width, nChannel] = size(im);
patch = zeros([patchSize, nChannel-1], 'like', im);

% pad image to have dimensions as multiples of patchSize
padSize(1) = patchSize(1) - mod(height, patchSize(1));
padSize(2) = patchSize(2) - mod(width, patchSize(2));

im_pad = padarray (im, padSize, 0, 'post');
[height_pad, width_pad, nChannel_pad] = size(im_pad);

out = zeros([size(im_pad,1), size(im_pad,2)], 'uint8');

for i = 1:patchSize(1):height_pad
    
    for j =1:patchSize(2):width_pad
        
        for p = 1:nChannel-1
               
            patch(:,:,p) = squeeze( im_pad( i:i+patchSize(1)-1,...
                                            j:j+patchSize(2)-1,...
                                            p));
            
        end
        
        patch_seg = semanticseg(patch, net, 'outputtype', 'uint8');
        
        out(i:i+patchSize(1)-1, j:j+patchSize(2)-1) = patch_seg;
       
    end
end

% Remove the padding
out = out(1:height, 1:width);
