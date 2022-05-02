function[MS] = GenerateTranslationAction(MS,i,j)
% % Generates vector for translation transformation
%
% Inputs:
% MS - Manufacturing strategy (struct)
% i - Node index
% j - Action index
%
% Outputs:
% MS - Manufacturing strategy (struct)

% Position of next vector relative to the initial position
position = [MS.node(i+1).datum.global.position - MS.node(1).datum.global.position ; 1];

% Calculate transformation matrix of current position relative to initial position
T_matrix = eye([4,4]);
for u = 1:i-1
    T_matrix = T_matrix * MS.node(u).datum.T;
end

% Calculate next position relative to current 
pos = CalculateTinv(T_matrix) * position;

% Set translation properties
MS.node(i).datum.action(j).name = 'translation';
MS.node(i).datum.action(j).O = pos(1:3);
MS.node(i).datum.action(j).dist = sqrt((MS.node(i).datum.action(j).O(1)^2)+(MS.node(i).datum.action(j).O(2)^2));