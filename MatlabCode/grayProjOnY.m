function vGrayProj = grayProjOnY( inputRows,inputCols,image )
%GRAYPROJONX 
% inputRows: num of rows of the image
% inputCols: num of cols of the image
% image: the image to be projected on Y axis
%   此处显示详细说明
% this function is to calculate the gray projection vertically
for y = uint64(1:inputCols)
    vGrayProj(y) = sum(image(1:inputRows,y));
end

end

