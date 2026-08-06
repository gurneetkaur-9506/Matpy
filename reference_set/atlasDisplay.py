import numpy as np

# MATLAB: function Eatlas = atlasDisplay(array_factor, xAxis, yAxis)
#      -> Python: def returns Eatlas, same parameter list
def atlasDisplay(array_factor, xAxis, yAxis):

    # MATLAB: xAxisStep = -0.5:(1/(length(xAxis)-1)):0.5;
    #      -> Python: uniform step 1/(L-1) from -0.5 to 0.5 is np.linspace
    xAxisStep = np.linspace(-0.5, 0.5, len(xAxis))

    # MATLAB: atlasRow(rowNo,:) = ... (implicit array growth on first
    #      assignment) -> Python: preallocate once, same shape/dtype
    atlasRow = np.zeros_like(array_factor)

    # MATLAB: for rowNo = 1:length(yAxis) -> Python: range starts at 0,
    #      1-based -> 0-based
    for rowNo in range(len(yAxis)):

        # MATLAB: dataInRow = array_factor(rowNo,:);
        #      -> Python: 1-based index converted to 0-based, ':' slice preserved
        dataInRow = array_factor[rowNo, :]

        # MATLAB: nonZeroIndex = find(dataInRow ~= 0);
        #      -> Python: np.where over the != 0 condition
        nonZeroIndex = np.where(dataInRow != 0)[0]

        # MATLAB: nonZeroData = dataInRow(nonZeroIndex);
        #      -> Python: fancy indexing (unused in the original as well)
        nonZeroData = dataInRow[nonZeroIndex]

        # MATLAB: nonZeroXAxis = xAxis(nonZeroIndex);
        #      -> Python: fancy indexing with the nonzero positions
        nonZeroXAxis = xAxis[nonZeroIndex]

        # MATLAB: tempE = array_factor(rowNo,nonZeroIndex);
        #      -> Python: 0-based row index + fancy column index
        tempE = array_factor[rowNo, nonZeroIndex]

        # MATLAB: interpPoints = 2*nonZeroXAxis(1)*xAxisStep;
        #      -> Python: '*' is element-wise, first element is index 0
        interpPoints = 2 * nonZeroXAxis[0] * xAxisStep

        # MATLAB: atlasRow(rowNo,:) = interp1(nonZeroXAxis,tempE,interpPoints);
        #      -> Python: np.interp(xq, x, v) with arguments reordered
        atlasRow[rowNo, :] = np.interp(interpPoints, nonZeroXAxis, tempE)

    # MATLAB: Eatlas = atlasRow; -> Python: assignment to the return variable
    Eatlas = atlasRow

    # MATLAB: end (function) -> Python: return the computed field
    return Eatlas
