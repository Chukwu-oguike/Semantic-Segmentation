

function out = switch_Channels_Z_plane(im)
%This Function converts image data to from numChannels-by-width-by-height arrays
% to width-by-height-by-numChannels arrays
    
    for i = 1:size(im,1)
        out(:,:,i) = im(i,:,:);
    end
end