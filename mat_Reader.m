
function data = mat_Reader(filename)
%This function extracts the first six channels from the training data and 
%omits the last channel containing the mask

    d = load(filename);
    f = fields(d);
    data = d.(f{1})(:,:,1:6);