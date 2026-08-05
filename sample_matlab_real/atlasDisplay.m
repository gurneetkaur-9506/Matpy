function Eatlas = atlasDisplay(array_factor,xAxis,yAxis)
xAxisStep = -0.5:(1/(length(xAxis)-1)):0.5;
for rowNo = 1:length(yAxis)
    dataInRow = array_factor(rowNo,:);
    nonZeroIndex = find(dataInRow ~= 0);
    nonZeroData = dataInRow(nonZeroIndex);
    nonZeroXAxis = xAxis(nonZeroIndex);
    tempE = array_factor(rowNo,nonZeroIndex);
    interpPoints = 2*nonZeroXAxis(1)*xAxisStep;
    atlasRow(rowNo,:) = interp1(nonZeroXAxis,tempE,interpPoints);
end
Eatlas = atlasRow;