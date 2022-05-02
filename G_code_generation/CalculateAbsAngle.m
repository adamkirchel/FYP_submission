function[abs_angle] = CalculateAbsAngle(WheelPath,i)
% % Calculates the angle of the wheel relative to the sheet at the current 
%   node
%
%   Inputs:
%   WheelPath - Positions relative to the sheet of the tracking pattern
%   (array)
%   i - Index of current node (int)
%   
%   Outputs:
%   abs_angle - Angle of the wheel (double)

if WheelPath(i,1) > 0
    abs_angle = atand((WheelPath(i,2) - WheelPath(i-1,2))/(WheelPath(i,1) - WheelPath(i-1,1)));
else
    abs_angle = 180 + atan((WheelPath(i,2) - WheelPath(i-1,2))/(WheelPath(i,1) - WheelPath(i-1,1)));
end

if abs_angle < 0
    abs_angle = abs_angle + 180;
end