This files train a U-Net convolutional neural network to perform semantic 
segmentation of a multispectral image with seven channels: three color channels, three 
near-infrared channels, and a mask. The goal is to recognize forest cover and determine the 
percentage of forest cover in an image

Run the Image_seg_Veg_Cover_Script to view results. The training process takes 24+ hrs. Ensure 
that the supporting functions build_U_net; segment_image; download_image_data; mat_Reader and 
switch_Channels_Z_plane are all in the same directory. Consult the function files for documentation.