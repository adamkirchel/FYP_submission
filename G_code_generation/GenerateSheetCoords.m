function[SheetCoords] = GenerateSheetCoords(MS)
% % Generates the sheet coordinates for each action for display purposes
% 
% Inputs:
% MS - Manufacturing strategy (struct)
%
% Outputs:
% SheetCoords - Sheet coordinates (cell array)

% Initialise
SheetCoords = {};

% Start up operations
SheetCoords{1} = [1-MS.node(1).datum.global.position(1) 1; -MS.node(1).datum.global.position(1) 1; -MS.node(1).datum.global.position(1) 0; 1-MS.node(1).datum.global.position(1) 0; 1-MS.node(1).datum.global.position(1) 1];
SheetCoords{2} = SheetCoords{1};
SheetCoords{3} = SheetCoords{2};
SheetCoords{4} = SheetCoords{3};
SheetCoords{5} = SheetCoords{4};
SheetCoords{6} = [1-MS.node(1).datum.global.position(1) 1-MS.node(1).datum.global.position(2) ; -MS.node(1).datum.global.position(1) 1-MS.node(1).datum.global.position(2) ; -MS.node(1).datum.global.position(1) -MS.node(1).datum.global.position(2) ; 1-MS.node(1).datum.global.position(1) -MS.node(1).datum.global.position(2) ; 1-MS.node(1).datum.global.position(1) 1-MS.node(1).datum.global.position(2)];
SheetCoords{7} = SheetCoords{6};
SheetCoords{8} = SheetCoords{7};

T_inv = CalculateTinv(MS.node(1).datum.T);
pos1 = T_inv * [SheetCoords{8}(1,:)';0;1];
pos2 = T_inv * [SheetCoords{8}(2,:)';0;1];
pos3 = T_inv * [SheetCoords{8}(3,:)';0;1];
pos4 = T_inv * [SheetCoords{8}(4,:)';0;1];
SheetCoords{9} = [pos1(1:2)';pos2(1:2)';pos3(1:2)';pos4(1:2)';pos1(1:2)'];

% Index
n = 10;

% Loop through remaining nodes
for i = 2:length(MS.node)-1

    [nr,~] = size(MS.node(i).effectors.action);
    
    % Loop through actions for current node
    for j = 1:nr 
        [end1size,~] = size(MS.node(i).effectors.action{j,1});
        [end2size,~] = size(MS.node(i).effectors.action{j,2});
        
        % If there is a rotation
        if end1size == 0
            act1name = '';
            act1enabled = 0;
        else
            act1name = MS.node(i).effectors.action{j,1}.name;
            act1enabled = MS.node(i).effectors.action{j,1}.enabled;
        end
        
        if end2size == 0
            act2name = '';
            act2enabled = 0;
        else
            act2name = MS.node(i).effectors.action{j,2}.name;
            act2enabled = MS.node(i).effectors.action{j,2}.enabled;
        end
        
        % Calculate inverse transformation matrices
        % Pivot on either effector
        if strcmp(act1name,'pivot') || strcmp(act2name,'pivot')
            T_inv = CalculateTinv([MS.node(i).datum.action(1).R [0;0;0];[0 0 0 1]]);

        % Front translation
        elseif strcmp(act1name,'translation') && act1enabled
            T_inv = [eye(3,3) MS.node(i).effectors.action{j,1}.O ; [0 0 0 1]];
            
        % Back translation
        elseif strcmp(act2name,'translation') && act2enabled
            T_inv = [eye(3,3) MS.node(i).effectors.action{j,2}.O ; [0 0 0 1]];
           
        % No movement
        else
            T_inv = eye(4,4);
            
        end
        
        % Calculate sheet coordinates
        pos1 = T_inv * [SheetCoords{n-1}(1,:)';0;1];
        pos2 = T_inv * [SheetCoords{n-1}(2,:)';0;1];
        pos3 = T_inv * [SheetCoords{n-1}(3,:)';0;1];
        pos4 = T_inv * [SheetCoords{n-1}(4,:)';0;1];
        SheetCoords{n} = [pos1(1:2)';pos2(1:2)';pos3(1:2)';pos4(1:2)';pos1(1:2)'];
        n = n + 1;
    end
    
end