function hGrayProj = grayProjOnX( inputRows,inputCols,image )
%GRAYPROJONX 
% inputRows: num of rows of the image
% inputCols: num of cols of the image
% image: the image to be projected on X axis
%   此处显示详细说明
% this function is to calculate the gray projection horizonally
for x = uint64(1:inputRows)
    hGrayProj(x) = sum(image(x,1:inputCols));
end

end

