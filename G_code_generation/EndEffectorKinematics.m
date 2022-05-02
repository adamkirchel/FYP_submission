function[MS] = EndEffectorKinematics(MS,Params)
% % Determines the transformations required by each end effector given the
% wheel transformations
%
% Inputs:
% MS - Manufacturing strategy (struct)
% Params - Parameters defining the wheel path (struct)
%
% Outputs:
% MS - Manufacturing strategy (struct)

% Initial position front
MS.node(1).effectors.position{1,1} = [0; Params.Fixed.WheelDistNorm; 50];
% Initial position back
MS.node(1).effectors.position{1,2} = [0; -Params.Fixed.WheelDistNorm; 50];

% Calibrate
MS.node(1).effectors.action{1,1}.name = 'calibrate';
MS.node(1).effectors.action{1,1}.position{1} = MS.node(1).effectors.position{1,1};
MS.node(1).effectors.action{1,1}.position{2} = MS.node(1).effectors.position{1,1};
MS.node(1).effectors.action{1,1}.enabled = 0;

% Home
MS.node(1).effectors.action{2,1}.name = 'home all';
MS.node(1).effectors.action{2,1}.position{1} = MS.node(1).effectors.position{1,1};
MS.node(1).effectors.action{2,1}.position{2} = MS.node(1).effectors.position{1,1};
MS.node(1).effectors.action{2,1}.enabled = 0;

% Adjust roll gap
MS.node(1).effectors.action{3,1}.name = 'adjust roll gap';
MS.node(1).effectors.action{3,1}.position{1} = MS.node(1).effectors.position{1,1};
MS.node(1).effectors.action{3,1}.position{2} = MS.node(1).effectors.position{1,1};
MS.node(1).effectors.action{3,1}.enabled = 0;

% Initial translation axis 3 from home into position
MS.node(1).effectors.action{4,1}.name = 'translation';
MS.node(1).effectors.action{4,1}.O = [0; Params.Fixed.WheelDistNorm + Params.Path.EffBoundNorm - MS.node(1).datum.global.position(2); 0];
MS.node(1).effectors.action{4,1}.position{1} = MS.node(1).effectors.position{1,1};
MS.node(1).effectors.action{4,1}.position{2} = [0; ...
    2*Params.Fixed.WheelDistNorm + Params.Path.EffBoundNorm - MS.node(1).datum.global.position(2);50];
MS.node(1).effectors.action{4,1}.enabled = 0;

% Drive through boundary
MS.node(1).effectors.action{5,1}.name = 'drive';
MS.node(1).effectors.action{5,1}.distance = MS.node(1).datum.global.position(2);
MS.node(1).effectors.action{5,1}.position{1} = MS.node(1).effectors.action{4,1}.position{2};
MS.node(1).effectors.action{5,1}.position{2} = MS.node(1).effectors.action{5,1}.position{1};
MS.node(1).effectors.action{5,1}.enabled = 0;

% Front effector vacuum on
MS.node(1).effectors.action{6,1}.name = 'vac on';
MS.node(1).effectors.action{6,1}.position{1} = MS.node(1).effectors.action{5,1}.position{2};
MS.node(1).effectors.action{6,1}.position{2} = MS.node(1).effectors.action{6,1}.position{1};
MS.node(1).effectors.action{6,1}.enabled = 0;

% Attach front 
MS.node(1).effectors.action{7,1}.name = 'active';
MS.node(1).effectors.action{7,1}.position{1} = MS.node(1).effectors.action{6,1}.position{2};
MS.node(1).effectors.action{7,1}.position{2} = [MS.node(1).effectors.action{7,1}.position{1}(1);MS.node(1).effectors.action{7,1}.position{1}(2);0];
MS.node(1).effectors.action{7,1}.enabled = 1;

% Initial tracking
MS.node(1).effectors.action{8,1}.name = 'translation';
MS.node(1).effectors.action{8,1}.O = -(MS.node(2).datum.global.position - MS.node(1).datum.global.position);
MS.node(1).effectors.action{8,1}.position{1} = MS.node(1).effectors.action{7,1}.position{2};
MS.node(1).effectors.action{8,1}.position{2} = MS.node(1).effectors.action{8,1}.position{1} + MS.node(1).effectors.action{8,1}.O;
MS.node(1).effectors.action{8,1}.enabled = 1;

% Final positions
MS.node(1).effectors.position{2,1} = MS.node(1).effectors.action{8,1}.position{2};
MS.node(1).effectors.position{2,2} = MS.node(1).effectors.position{1,2};

% Add parameters at node
MS.node(1).tooltype = Params.Machine.ToolType;

for i = 2:length(MS.node)-1
    
    % Reset action index
    j = 1;
    
    % Add parameters at node
    MS.node(i).tooltype = Params.Machine.ToolType;

    % Check transfer
    if MS.node(i).enabled ~= MS.node(i-1).enabled
        MS.node(i).effectors.transfer = 1;
    else
        MS.node(i).effectors.transfer = 0;
    end
    
    % action if there is a transfer
    if MS.node(i).effectors.transfer
        
        % Front effector enabled
        if MS.node(i).enabled(1)
            
            % Calculate initial end effector positions
            MS.node(i).effectors.position{1,1} = [0;Params.Fixed.WheelDistNorm;50];
            pos = MS.node(i-1).effectors.position{2,2};
            MS.node(i).effectors.position{1,2} = pos(1:3);
            
            % Rotation previous to transfer on node if required
            if ~isequal(MS.node(i).datum.action(1).R,eye(3,3))
                % New position of enabled end effector
                T_matrix = MS.node(i-1).datum.T * [MS.node(i).datum.action(1).R [0;0;0];[0 0 0 1]];      
                pos = CalculateTinv(T_matrix) * [MS.node(i-1).effectors.position{1,2};1];
            
                MS.node(i).effectors.action{j,2}.name = 'pivot';
                MS.node(i).effectors.action{j,2}.angle = MS.node(i).datum.action(1).angle;
                MS.node(i).effectors.action{j,2}.radius = abs(MS.node(i).effectors.position{1,2}(2));
                MS.node(i).effectors.action{j,2}.position{1} = MS.node(i).effectors.position{1,2};
                MS.node(i).effectors.action{j,2}.position{2} = pos(1:3);
                MS.node(i).effectors.action{j,2}.enabled = 1;
                
                j = j + 1;
            end
            
            % Transfer
            MS.node(i).effectors.action{j,2}.name = 'transfer';
            MS.node(i).effectors.action{j,2}.position{1} = pos(1:3);
            MS.node(i).effectors.action{j,2}.position{2} = [MS.node(i).effectors.action{j,2}.position{1}(1);MS.node(i).effectors.action{j,2}.position{1}(2);50];
            MS.node(i).effectors.action{j,2}.enabled = 0;
            j = j + 1;

            % Enabled effector translation after transfer
            MS.node(i).effectors.action{j,1}.name = 'translation';
            MS.node(i).effectors.action{j,1}.O = -MS.node(i).datum.action(1).R' * MS.node(i).datum.action(2).O;
            MS.node(i).effectors.action{j,1}.position{1} = [MS.node(i).effectors.position{1,1}(1);MS.node(i).effectors.position{1,1}(2);0];
            MS.node(i).effectors.action{j,1}.position{2} = MS.node(i).effectors.action{j,1}.position{1} + MS.node(i).effectors.action{j,1}.O;
            MS.node(i).effectors.action{j,1}.enabled = 1;
            j = j + 1;
            
            % Disabled effector translation
            MS.node(i).effectors.action{j,2}.name = 'translation';
            MS.node(i).effectors.action{j,2}.O = [0; -Params.Fixed.WheelDistNorm; 50] - MS.node(i).effectors.action{j-2,2}.position{2};
            MS.node(i).effectors.action{j,2}.position{1} = MS.node(i).effectors.action{j-2,2}.position{2};
            MS.node(i).effectors.action{j,2}.position{2} = MS.node(i).effectors.action{j,2}.position{1} + MS.node(i).effectors.action{j,2}.O;
            MS.node(i).effectors.action{j,2}.enabled = 0;
            
            % Final positions
            MS.node(i).effectors.position{2,1} = MS.node(i).effectors.action{j-1,1}.position{2};
            MS.node(i).effectors.position{2,2} = MS.node(i).effectors.action{j,2}.position{2};
            
        % Back effector enabled
        else
            % Calculate effector positions
            MS.node(i).effectors.position{1,2} = [0;-Params.Fixed.WheelDistNorm;50];
            pos = MS.node(i-1).effectors.position{2,1};
            MS.node(i).effectors.position{1,1} = pos(1:3);
            
            % Rotation previous to transfer on node if required
            if ~isequal(MS.node(i).datum.action(1).R,eye(3,3))
                % New position of enabled end effector
                T_matrix = MS.node(i-1).datum.T * [MS.node(i).datum.action(1).R [0;0;0];[0 0 0 1]];
                pos = CalculateTinv(T_matrix) * [MS.node(i-1).effectors.position{1,1};1];
                
                MS.node(i).effectors.action{j,1}.name = 'pivot';
                MS.node(i).effectors.action{j,1}.angle = MS.node(i).datum.action(1).angle;
                MS.node(i).effectors.action{j,1}.radius = abs(MS.node(i).effectors.position{1,1}(2));
                MS.node(i).effectors.action{j,1}.position{1} = MS.node(i).effectors.position{1,1};
                MS.node(i).effectors.action{j,1}.position{2} = pos(1:3);
                MS.node(i).effectors.action{j,1}.enabled = 1;
                
                j = j + 1;
            end
            
            % Transfer
            MS.node(i).effectors.action{j,1}.name = 'transfer';
            MS.node(i).effectors.action{j,1}.position{1} = pos(1:3);
            MS.node(i).effectors.action{j,1}.position{2} = [MS.node(i).effectors.action{j,1}.position{1}(1);MS.node(i).effectors.action{j,1}.position{1}(2);50];
            MS.node(i).effectors.action{j,1}.enabled = 0;
            j = j + 1;

            % Enabled effector translation after transfer
            MS.node(i).effectors.action{j,2}.name = 'translation';
            MS.node(i).effectors.action{j,2}.O = -MS.node(i).datum.action(1).R' * MS.node(i).datum.action(2).O;
            MS.node(i).effectors.action{j,2}.position{1} = [MS.node(i).effectors.position{1,2}(1);MS.node(i).effectors.position{1,2}(2);0];
            MS.node(i).effectors.action{j,2}.position{2} = MS.node(i).effectors.action{j,2}.position{1} + MS.node(i).effectors.action{j,2}.O;
            MS.node(i).effectors.action{j,2}.enabled = 1;
            j = j + 1;
            
            % Disabled effector translation
            MS.node(i).effectors.action{j,1}.name = 'translation';
            MS.node(i).effectors.action{j,1}.O = [0; Params.Fixed.WheelDistNorm; 50] - MS.node(i).effectors.action{j-2,1}.position{2};
            MS.node(i).effectors.action{j,1}.position{1} = MS.node(i).effectors.action{j-2,1}.position{2};
            MS.node(i).effectors.action{j,1}.position{2} = MS.node(i).effectors.action{j,1}.position{1} + MS.node(i).effectors.action{j,1}.O;
            MS.node(i).effectors.action{j,1}.enabled = 0;
            
            % Final positions
            MS.node(i).effectors.position{2,1} = MS.node(i).effectors.action{j,1}.position{2};
            MS.node(i).effectors.position{2,2} = MS.node(i).effectors.action{j-1,2}.position{2};
        end
        
    else
        % Front effector enabled
        if MS.node(i).enabled(1)
            
            % Calculate node start positions
            MS.node(i).effectors.position{1,1} = MS.node(i-1).effectors.position{2,1};
            MS.node(i).effectors.position{1,2} = MS.node(i-1).effectors.position{2,2};
            
            % Rotation
            MS.node(i).effectors.action{j,1}.name = 'pivot';
            MS.node(i).effectors.action{j,1}.angle = MS.node(i).datum.action(1).angle;
            MS.node(i).effectors.action{j,1}.radius = abs(MS.node(i).effectors.position{1,1}(2));
            MS.node(i).effectors.action{j,1}.position{1} = MS.node(i).effectors.position{1,1};
            MS.node(i).effectors.action{j,1}.position{2} = MS.node(i).datum.action(1).R' * MS.node(i).effectors.action{j,1}.position{1};
            MS.node(i).effectors.action{j,1}.enabled = 1;
            MS.node(i).effectors.action{j,2} = [];
            j = j + 1;
            
            % Translation
            MS.node(i).effectors.action{j,1}.name = 'translation';
            MS.node(i).effectors.action{j,1}.O = -MS.node(i).datum.action(1).R' * MS.node(i).datum.action(2).O;
            MS.node(i).effectors.action{j,1}.position{1} = MS.node(i).effectors.action{j-1,1}.position{2};
            MS.node(i).effectors.action{j,1}.position{2} = MS.node(i).effectors.action{j,1}.position{1} + MS.node(i).effectors.action{j,1}.O;
            MS.node(i).effectors.action{j,1}.enabled = 1;
            MS.node(i).effectors.action{j,2} = [];
            
            % Final position
            MS.node(i).effectors.position{2,1} = MS.node(i).effectors.action{j,1}.position{2};
            MS.node(i).effectors.position{2,2} = MS.node(i).effectors.position{1,2};
        else
            % Calculate node start positions
            MS.node(i).effectors.position{1,1} = MS.node(i-1).effectors.position{2,1};
            MS.node(i).effectors.position{1,2} = MS.node(i-1).effectors.position{2,2};

            % Rotation
            MS.node(i).effectors.action{j,2}.name = 'pivot';
            MS.node(i).effectors.action{j,2}.angle = MS.node(i).datum.action(1).angle;
            MS.node(i).effectors.action{j,2}.radius = abs(MS.node(i).effectors.position{1,2}(2));
            MS.node(i).effectors.action{j,2}.position{1} = MS.node(i).effectors.position{1,2};
            MS.node(i).effectors.action{j,2}.position{2} = MS.node(i).datum.action(1).R' * MS.node(i).effectors.action{j,2}.position{1};
            MS.node(i).effectors.action{j,2}.enabled = 1;
            MS.node(i).effectors.action{j,1} = [];
            j = j + 1;
            
            % Translation
            MS.node(i).effectors.action{j,2}.name = 'translation';
            MS.node(i).effectors.action{j,2}.O = -MS.node(i).datum.action(1).R' * MS.node(i).datum.action(2).O;
            MS.node(i).effectors.action{j,2}.position{1} = MS.node(i).effectors.action{j-1,2}.position{2};
            MS.node(i).effectors.action{j,2}.position{2} = MS.node(i).effectors.action{j,2}.position{1} + MS.node(i).effectors.action{j,2}.O;
            MS.node(i).effectors.action{j,2}.enabled = 1;
            MS.node(i).effectors.action{j,1} = [];
            
            % Final positions
            MS.node(i).effectors.position{2,1} = MS.node(i).effectors.position{1,1};
            MS.node(i).effectors.position{2,2} = MS.node(i).effectors.action{j,2}.position{2};
        end
        
    end
end