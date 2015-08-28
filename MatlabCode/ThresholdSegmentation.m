function [] = ThresholdSegmentation( input_args )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
clear all
clc
close all
input = imread('test6.jpg');
figure;
imshow(input);
title('input');
sizeOfInput = size(input);

if numel(sizeOfInput)>2
    input = rgb2gray(input);
end
grayimgToBeGaus = input;
result = grayimgToBeGaus;
figure;
imshow(result);
title('first result');
[inputRows,inputCols] = size(input);
%%
%--------------------------------filtering--------------------------------%
gausFilter = fspecial('gaussian',[35 35],20);
blur=imfilter(grayimgToBeGaus,gausFilter);
figure;
imshow(blur);
title('Gaussian filtering');
%%
%--------------------------threshold segmentation-------------------------%
ThresValue=graythresh(blur);
J=im2bw(grayimgToBeGaus,ThresValue);
figure;
imshow(J);
title('lung segmentation bw');
%%
%--------------------------close-open operation---------------------------%
%openedJ = bwmorph(J,'open');
disk = strel('disk',3);
line = strel('line',6,0);
riberosion = imerode(J,disk);
ribdilation = imdilate(riberosion,disk);
openedJ = imdilate(ribdilation,disk);
figure;
imshow(openedJ);
title('open');
% colcounter = 0;
% for y = 1:inputCols
%     for x = 1:inputRows
%         if openedJ(x,y) == 0
%             colcounter = colcounter + 1;
%         end
%         
%     end
%     if colcounter < 175
%         for newx = 1:inputRows
%             if newx > 403 && newx <450
%                 result(newx,y) = result(newx,y)-rem((100*rand()),13);
%             end
%         end
%     end
%     colcounter = 0;
% % end
% figure;
% imshow(result);
%%
[outsideLungX,outsideLungY] = find(openedJ == 1);
Xsize = size(outsideLungX);
for pointid = 1:Xsize
    x = outsideLungX(pointid,1);
    y = outsideLungY(pointid,1);
    if x< 404
        result(x,y) = result(x,y) - rem((100*rand()),28);
    end
    if x> 449
        result(x,y) = result(x,y) - rem((100*rand()),28);
    end 
    if x > 403 && x <450
        if y >170 || y<30
            result(x,y) = result(x,y) - rem((100*rand()),28);
        end
    end
end

figure;
imshow(result);
title('ouside lung noising')
%%
%---------------------disturb the image in a low-mod way------------------%
[whiteAreaRows,whiteAreaCols] = find(J==1);
whiteAreaCoor = [whiteAreaRows,whiteAreaCols];
whiteAreasize = size(whiteAreaRows,1);
for whitePoint = 1:whiteAreasize
    whiteX = whiteAreaCoor(whitePoint,1);
    whiteY = whiteAreaCoor(whitePoint,2);
    result(whiteX,whiteY) = result(whiteX,whiteY)+mod(rand()*100,5);
end
hGrayInput = grayProjOnX(inputRows,inputCols,J);
vGrayInput = grayProjOnY(inputRows,inputCols,J);
figure;
x = 1:inputRows;
plot(x,hGrayInput(x));
title('horizonal projection');
y = 1:inputCols;
figure;
plot(y,vGrayInput(y));
title('vertical projection');
segmentedim = grayimgToBeGaus;
for point=1:inputRows
    for y=1:inputCols
        if J(point,y) ==1
            segmentedim(point,y) = 0;
            blur(point,y) = 0;
            %             result(point,y) = result(point,y) + mod(100*rand(),20);
            %             result(point,y) = 255;
        end
    end
end
% for point2=1:inputRows
%     for y=1:10
%%
%----------------------put the area without lung into black---------------%
intRand = 1000*rand();
randPositiveOrNegtive = (-1).^intRand;
segmentedim(:,1:10) = 0;
blur(:,1:10) = 0;
for x1 = 1:inputRows
    for y1 = 1:10
        if 1000*rand()>500
            result(x1,y1) = result(x1,y1) + rem...
                ((100*rand()),10);
        else
            result(x1,y1) = result(x1,y1) - rem...
                ((100*rand()),10);
        end
    end
end

%%
upper = input(1:145,1:inputCols);
aveFilter = fspecial('gaussian',[20 20]);
upper = imfilter(upper,aveFilter);
figure;
imshow(upper)
%%
[upperRows,upperCols] = find(upper<80);
upperPixNum = size(upperRows,1);
for pid = 1:upperPixNum
    x = upperRows(pid,1);
    y = upperCols(pid,1);
    if x < 145
        result(x,y) =0;
    end
    
end
figure;
imshow(result);
%%
segmentedim(1:185,:) = 0;
blur(1:185,:) = 0;
segmentedim(515:inputRows,:) = 0;
blur(515:inputRows,:) = 0;
for x3 = 515:inputRows
    for y3 = 1:inputCols
        if grayimgToBeGaus(x3,y3)<30
            result(x3,y3) = result(x3,y3) + rem((100*rand()),10);
        elseif 1000*rand()>500
            result(x3,y3) = result(x3,y3) + rem((100*rand()),10);
        else
            result(x3,y3) = result(x3,y3) - rem((100*rand()),10);
        end
        %         if 1000*rand()>500
        %             result(x2,y2) = result(x2,y2) + rem((100*rand()),30);
        %         else
        %             result(x2,y2) = result(x2,y2) - rem((100*rand()),30);
        %         end
    end
end
%         result(x,point4) = result(x,point4) - mod(100*rand(),10);
%         result(x,point4) = 255;
%     end
% end
% for point=1:inputRows
%     for y=526:inputCols
segmentedim(:,510:inputCols) = 0;
blur(:,510:inputCols) = 0;
for x4 = 1:inputRows
    for y4 = 510:inputCols
        if 1000*rand()>500
            result(x4,y4) = result(x4,y4) + rem...
                ((100*rand()),10);
        else
            result(x4,y4) = result(x4,y4) - rem...
                ((100*rand()),10);
        end
    end
end
%         result(point,y) = 255;
%             result(point,y) = result(point,y) - mod(100*rand(),10);
%     end
% end
figure;
imshow(segmentedim);
title('边际切除后');
imwrite(segmentedim,'edge_segmentation.jpg');

segmentedbw = im2bw(segmentedim,0);
segmentlabel = bwlabel(segmentedbw,4);
maxLabel = max(max(segmentlabel));
for sLabel = 1:maxLabel
    [sBlockRow,sBlockCol] = find(segmentlabel==sLabel);
    sBlockCoor = [sBlockRow,sBlockCol];
    sBlocksize = size(sBlockRow,1);
    if sBlocksize<300
        for sBlockpoint = 1:sBlocksize
            sBlockX = sBlockCoor(sBlockpoint,1);
            sBlockY = sBlockCoor(sBlockpoint,2);
            segmentedim(sBlockX,sBlockY) = 0;
        end
    end
end
figure;
imshow(segmentedbw);
figure;
subplot(1,2,1);imshow(segmentedim);
title('lung segmentation origin');
subplot(1,2,2);imshow(blur);
title('Gaussian Segmentation');
imwrite(segmentedim,'lung_final.jpg');
%%
%------------------Gaussian filtering to segment lung area----------------%
% newblur=imfilter(segmentedim,gausFilter);
% figure;
% imshow(newblur);
% title('Gaussian filtering');
% newThresValue=graythresh(newblur);
% newJ=im2bw(segmentedim,newThresValue);
% figure;
% imshow(newJ);
% title('lung segmentation bw');
%%
%------------calculate the horiaonal and vertical gray sum----------------%
%for vertical gray projection
% for y = uint64(1:inputCols)
%     vGrayProj(y) = sum(segmentedim(1:inputRows,y));
%     vGausGrayProj(y) = sum(blur(1:inputRows,y));
% %     vGrayProj(y) = sum(input(1:inputRows,y));
% end
vGrayProj = grayProjOnY( inputRows,inputCols,segmentedim);
vGausGrayProj = grayProjOnY( inputRows,inputCols,blur);
y = uint64(1:inputCols);
figure;
hold on
plot(y,vGrayProj(y));
plot(y,vGausGrayProj(y));
hold off
%for horizonal gray projection
% for x = uint64(1:inputRows)
%     hGrayProj(x) = sum(segmentedim(x,1:inputCols));
%     hGausGrayProj(x) = sum(blur(x,1:inputCols));
% %     vGrayProj(y) = sum(input(1:inputRows,y));
% end
hGrayProj = grayProjOnX( inputRows,inputCols,segmentedim);
hGausGrayProj = grayProjOnX( inputRows,inputCols,blur);
x = uint64(1:inputRows);
figure;
hold on
plot(x,hGrayProj(x));
plot(x,hGausGrayProj(x));
hold off
%%
%------------------------thresholding segmentation------------------------%
preribbw = im2bw(segmentedim);
for x = 1:inputRows
    for y = 1:inputCols
        variation = segmentedim(x,y)-blur(x,y);
        if variation>0
            preribbw(x,y) = 1;
        else
            preribbw(x,y) = 0;
        end
    end
end
figure;
imshow(preribbw);
title('Gaussian plane threshold segmentation');
imwrite(preribbw,'rough_rib.jpg');
%%
%-------------------------closing-opening filter--------------------------%
% disk = strel('disk',2);
% line = strel('line',6,0);
% riberosion = imerode(preribbw,line);
% ribdilation = imdilate(riberosion,line);
% ribdilationline = imdilate(ribdilation,line);
ribbwopened = bwmorph(preribbw,'open');
ribbwclosed = bwmorph(ribbwopened,'close');
figure;
imshow(ribbwclosed);
title('close-opened');
imwrite(ribbwclosed,'close-opend.jpg');
%%
%-----------------get the horizonal projection for new bw-----------------%
hBwProj = grayProjOnX(inputRows,inputCols,preribbw);
x = uint64(1:inputRows);
figure;
plot(x,hBwProj);
title('horizonal projection for new bw');
%%
%-------------------------labeling&remove the focus-----------------------%
labels = bwlabel(preribbw,4);
% labels = bwlabel(ribbwclosed,4);
reshapedLabels = reshape(labels,inputRows*inputCols,1);
numOfblocks = max(reshapedLabels);
tomedian = preribbw;
for blocknum = 1:numOfblocks
    [blockrows,blockcols] = find(labels==blocknum);
    blocksize = size(blockrows,1);
    if blocksize<300
        for blockpoints = 1:blocksize;
            x = uint64(blockrows(blockpoints,1));
            y = uint64(blockcols(blockpoints,1));
            tomedian(x,y) = 0;
        end
    end
end
figure;
imshow(tomedian);
title('remove the focus');
imwrite(tomedian,'focusRemoving.jpg');
[ribrows,ribcols] = find(tomedian==1);
pointsofribs = size(ribrows,1);
% for ribpoint = 1:pointsofribs
%     x = uint64(ribrows(pointsofribs,1));
%     y = uint64(ribcols(pointsofribs,1));
%     result(x,y) = result(x,y) + mod(100*rand(),10);
% end
figure;
imshow(result);
title('first result');
%%
%----------first time relabeling ribs & calcaulate the rib center---------%
newlabels = bwlabel(tomedian,4);
newreshapedLabels = reshape(labels,inputRows*inputCols,1);
newnumOfblocks = max(newreshapedLabels);
median = tomedian;
labelim = uint64(median);
label = 2;
for newblocknum = 1:numOfblocks
    [blockrows,blockcols] = find(newlabels==newblocknum);
    blocksize = size(blockrows,1);
    for blockpoints = 1:blocksize;
        x = uint64(blockrows(blockpoints,1));
        y = uint64(blockcols(blockpoints,1));
        median(x,y) = 0;
        sum = 0;
        ribcenterYid = 1;
        while (ribcenterYid<=blocksize)
            ribcenterY = blockcols(ribcenterYid,1);
            %blockincolCols should all be 1 theoritically
            [blockincolRows,blockincolCols] = ...
                find(blockcols==ribcenterY);
            sum = 0;
            sizeOfCol = size(blockincolRows,1);
            ribcenterYid = ribcenterYid+sizeOfCol;
            for bincolXid = 1:sizeOfCol;
                bincolX = blockincolRows(bincolXid,1);
                sum = sum+blockrows(bincolX,1);
            end
            aveX = sum/sizeOfCol;
            ribcenterX = uint64(aveX);
            median(ribcenterX,ribcenterY) = 1;
            labelim(ribcenterX,ribcenterY) = label;
        end
    end
    label = label+1;
end
figure;
imshow(median);
title('rib center');
hGrayOfRibCenter = grayProjOnX(inputRows,inputCols,median);
x = 1:inputRows;
figure;
plot(x,hGrayOfRibCenter(x));
% median(360:inputRows,:) = 0;
% labelim(360:inputRows,:) = 0;
for labelx = 1:inputRows
    for labely = 1:inputCols
        if labelim(labelx,labely)==1
            labelim(labelx,labely) = 0;
        end
    end
end
figure;
imshow(median);
title('rib center changed');
imwrite(median,'ribcenter.jpg');
%%
%-------------------kmeans clustering the coordinates---------------------%
% [ribcenterRows,ribcenterCols] = find(median==1);
% allRibcentercoor = [ribcenterRows,ribcenterCols];
% numofClusters = 8;
% linecluster = kmeans(allRibcentercoor,numofClusters);
%%
%-------------------------line segments relabeling------------------------%
numofLines = max(max(labelim));
fitted = grayimgToBeGaus;
oldavelineRow = 0;
newlabel = 21;
for lineLabel = 1:numofLines
    [lineRows,lineCols] = find(labelim==lineLabel);
    avelineRow = mean(lineRows);
    rowVariation = abs(avelineRow-oldavelineRow);
    blocksize = size(lineRows,1);
    for lineblocklabel = 1:numofLines
        [newlineRows,newlineCols] = find(labelim==lineblocklabel);
        newavelineRow = mean(newlineRows);
        rowVariation = abs(avelineRow-newavelineRow);
        newblocksize = size(newlineRows,1);
        if rowVariation<20
            for newpointofblock = 1:newblocksize
                newlineX = newlineRows(newpointofblock,1);
                newlineY = newlineCols(newpointofblock,1);
                labelim(newlineX,newlineY) = newlabel;
            end
            for pointofblock = 1:blocksize
                lineX = lineRows(pointofblock,1);
                lineY = lineCols(pointofblock,1);
                %             newlabel = lineLabel-1;
                %             [newlineRows,newlineCols] = find(labelim==newlabel);
                %             while(isempty(newlineRows))
                %                 [newlineRows,newlineCols] = find(labelim==newlabel);
                %                 newlabel = newlabel-1
                %             end
                labelim(lineX,lineY) = newlabel;
            end
        end
    end
    newlabel = newlabel+1;
end
% for cluster = 1:numofClusters
%     colsinLinecluster = find(linecluster==cluster);
%     numofpointstobefit = size(colsinLinecluster,1);
%     tobefitRows = [];
%     tobefitCols = [];
%     for pointstobefit = 1:numofpointstobefit
%         tobefitX = allRibcentercoor(colsinLinecluster(pointstobefit,1),1);
%         tobefitY = allRibcentercoor(colsinLinecluster(pointstobefit,1),2);
%         tobefitRows = [tobefitRows,tobefitX];
%         tobefitCols = [tobefitCols,tobefitY];
%     end
%         params = polyfit(tobefitCols,tobefitRows,2);
%         fitRow = uint64(params(1)*tobefitCols(1,:).^2+params(2)...
%             *tobefitCols(1,:)+params(3));
%         numofFitRow = size(fitRow,2);
%         for pointofFitrow = 1:numofFitRow
%             pointofFitX = fitRow(1,pointofFitrow);
%             pointofFitY = tobefitCols(1,pointofFitrow);
%             fitted(pointofFitX,pointofFitY) = 255;
%         end
%         tobefitRows = [];
%         tobefitCols = [];
% end
%%
%-----------------rib center fitting & remove the improper lines----------%
newnumofLines = max(max(labelim));
fitCol = [];
ribmodel = zeros(inputRows,inputCols);
for i = 1:inputCols
    fitCol = [fitCol;i];
end
for lineLabel = 1:newnumofLines
    [lineRows,lineCols] = find(labelim==lineLabel);
    avelineRow = mean(lineRows);
    rowVariation = abs(avelineRow-oldavelineRow);
    numofLineCols = size(lineCols,1);
    params = polyfit(lineCols,lineRows,2);
    disp(params(:));
    fitRow = uint64(params(1)*fitCol(:,1).^2+params(2)*fitCol(:,1)...
        +params(3));
    numofFitRow = size(fitRow,1);
    for pointofFitrow = 1:numofFitRow
        pointofFitX = fitRow(pointofFitrow,1);
        pointofFitY = fitCol(pointofFitrow,1);
        %remove the lines by the 2nd param of the parabola
        
        if (abs(params(2)<3))&&(pointofFitX~=0)&&(pointofFitX<inputRows)
%         if (pointofFitX~=0)&&(pointofFitX<inputRows)
            if segmentedim(pointofFitX,pointofFitY) ~= 0
%                 disp(params(:));
                fitted(pointofFitX,pointofFitY) = 255;
                ribmodel(pointofFitX,pointofFitY) = 255;
            end
        end
    end
end
figure;
imshow(fitted);
title('the image fitted');
imwrite(fitted,'fitted.jpg');
se = strel('disk',10);
ribmodel = imdilate(ribmodel,se);
figure;
imshow(ribmodel);
title('rib model');
imwrite(ribmodel,'ribmodel.jpg');
%%
%----------------labeling the fit then to disturb the ribs----------------%
rib_input = input;
result_rib_input = result;
rib_input(:,:) = 0;
result_rib_input(:,:) = 0;
[ribmodelRow ribmodelCol] = find(ribmodel==255);
ribmodelCoor = [ribmodelRow,ribmodelCol];
ribmodelsize = size(ribmodelRow,1);
for ribmodelpoint = 1:ribmodelsize;
    ribmodelX = ribmodelCoor(ribmodelpoint,1);
    ribmodelY = ribmodelCoor(ribmodelpoint,2);
    result(ribmodelX,ribmodelY) = result(ribmodelX,ribmodelY)+mod(100*rand(),20);
    result_rib_input(ribmodelX,ribmodelY) = result(ribmodelX,ribmodelY);
    rib_input(ribmodelX,ribmodelY) = input(ribmodelX,ribmodelY);
end
figure;
imshow(result);
imwrite(result,'result-test6-[35 35]-sigma20.png');
figure;
imshow(rib_input);
imwrite(result_rib_input,'noised_rib.png');
figure;
imshow(result_rib_input);
imwrite(rib_input,'ribforinput.png');
%}