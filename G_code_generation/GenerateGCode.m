function[MS] = GenerateGCode(MS,Params)
% % Generates G-code for each action
%
% Inputs:
% MS - Manufacturing strategy (struct)
% Params - Process parameters (struct)
%
% Outputs:
% MS - Manufacturing strategy (struct)

% Initialise variable
code = {};

% Setup operations
% Calibrate
code{1,1} = 'M20';
code{1,2} = 'Calibrate axes';

% Home
code{2,1} = 'G40';
code{2,2} = 'Home axes';

% Raise lower tool
code{3,1}= append('G12 X',num2str(Params.Machine.RollGap));
code{3,2} = 'Raise lower tool';

% Extend front effector
code{4,1} = append('G13 X',num2str(num2str(MS.node(1).effectors.axes{5}(3))));
code{4,2} = 'Extend A3';

% Drive through boundary
code{5,1} = append('G01 X',num2str(Params.Path.DriveDist));
code{5,2} = 'Drive through boundary';

% Vacuum on
code{6,1} = 'M06';
code{6,2} = 'Front vacuum on';

% Attach front
code{7,1} = 'G27';
code{7,2} = 'Attach front';

% Initial tracking
code{8,1} = append('G21 X',num2str(round(100*MS.node(1).effectors.action{end,1}.O(2)*Params.Sheet.SheetSize)/100));
code{8,2} = 'Front enabled translation';

% Index
k = 9;

% Loop through nodes
for i = 2:length(MS.node)-1
    
    [nr,~] = size(MS.node(i).effectors.action);
    
    % Actions for current node
    for j = 1:nr
        
        [end1size,~] = size(MS.node(i).effectors.action{j,1});
        [end2size,~] = size(MS.node(i).effectors.action{j,2});
        
        % Action for front effector
        if end1size
            if strcmp(MS.node(i).effectors.action{j,1}.name,'transfer')
                code{k,1} = 'G31';
                code{k,2} = 'Front - Back transfer';
            elseif strcmp(MS.node(i).effectors.action{j,1}.name,'pivot')
                code{k,1} = append('G25 X',num2str(round(100*MS.node(i).effectors.action{j,1}.angle*60)/100));
                code{k,2} = 'Front pivot';
            elseif MS.node(i).effectors.action{j,1}.enabled && strcmp(MS.node(i).effectors.action{j,1}.name,'translation')
                code{k,1} = append('G21 X',num2str(round(100*MS.node(i).effectors.action{j,1}.O(2)*Params.Sheet.SheetSize)/100));
                code{k,2} = 'Front enabled translation';
            elseif strcmp(MS.node(i).effectors.action{j,1}.name,'translation')
                code{k,1} = 'G43 G44';
                code{k,2} = 'Home front';
            end
        % Action for back effector
        else
            if strcmp(MS.node(i).effectors.action{j,2}.name,'transfer')
                code{k,1} = 'G32';
                code{k,2} = 'Back - Front transfer';
            elseif strcmp(MS.node(i).effectors.action{j,2}.name,'pivot')
                code{k,1} = append('G26 X',num2str(round(100*MS.node(i).effectors.action{j,2}.angle*60)/100));
                code{k,2} = 'Back pivot';
            elseif MS.node(i).effectors.action{j,2}.enabled && strcmp(MS.node(i).effectors.action{j,2}.name,'translation')
                code{k,1} = append('G23 X',num2str(round(100*-MS.node(i).effectors.action{j,2}.O(2)*Params.Sheet.SheetSize)/100));
                code{k,2} = 'Back enabled translation';
            elseif strcmp(MS.node(i).effectors.action{j,2}.name,'translation')
                code{k,1} = 'G46 G47';
                code{k,2} = 'Home back';
            end
        end
        
        % Next action
        k = k + 1;

    end
end
MS.code = code;