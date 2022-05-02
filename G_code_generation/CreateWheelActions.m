function[MS] = CreateWheelActions(MS)
% % Generates list of transformations including translations and rotations
% associated with the movement of the wheel relative to the sheet.
%
% Inputs:
% MS - Manufacturing Strategy (struct)
%
% Outputs:
% MS - Manufacturing Strategy (struct)

% Initial transformation
MS = GenerateRotationAction(MS,1,1);
MS.node(1).datum.action(2).name = 'translation';
MS.node(1).datum.action(2).O = MS.node(2).datum.global.position - MS.node(1).datum.global.position;
MS.node(1).datum.T = [cosd(MS.node(1).datum.action(1).angle),-sind(MS.node(1).datum.action(1).angle),0,MS.node(1).datum.action(2).O(1); ...
    sind(MS.node(1).datum.action(1).angle),cosd(MS.node(1).datum.action(1).angle),0,MS.node(1).datum.action(2).O(2); ...
    0,0,1,MS.node(1).datum.action(2).O(3);0,0,0,1];

for i = 2:length(MS.node)-1
 
    % Next transformation
    MS = GenerateRotationAction(MS,i,1);
    MS = GenerateTranslationAction(MS,i,2);   
    MS.node(i).datum.T = [MS.node(i).datum.action(1).R, MS.node(i).datum.action(2).O ; [0 0 0 1]];
    
end   