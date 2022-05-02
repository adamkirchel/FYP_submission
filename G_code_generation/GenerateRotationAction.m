function[MS] = GenerateRotationAction(MS,i,j)
% % Generates angle and matrix for rotation transformation
%
% Inputs:
% MS - Manufacturing strategy (struct)
% i - Node index
% j - Action index
%
% Outputs:
% MS - Manufacturing strategy (struct)

% Angle of rotation
angle = MS.node(i+1).datum.global.angle - MS.node(i).datum.global.angle;

% Change so that wheel is always pointing north
if abs(angle) > 90 && angle < 0
    angle = angle + 180;
elseif abs(angle) > 90 && angle > 0
    angle = angle - 180;
end

% Set rotation angle and matrix
MS.node(i).datum.action(j).name = 'rotation';
MS.node(i).datum.action(j).angle = angle;
MS.node(i).datum.action(j).R = [cosd(MS.node(i).datum.action(j).angle), -sind(MS.node(i).datum.action(j).angle), 0; ...
    sind(MS.node(i).datum.action(j).angle), cosd(MS.node(i).datum.action(j).angle), 0; ...
    0,0,1];