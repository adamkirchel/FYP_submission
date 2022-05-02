function[MS] = DefineAxes(MS,Params)
% % Define axes values for each action
%
% Inputs:
% MS - Manufacturing strategy (struct)
% Params - Process parameters (struct)
%
% Outputs:
% MS - Manufacturing strategy (struct)

% Loop through nodes
for i = 1:length(MS.node)-1
    
    % Define A1 excluding setup
    if i > 1
        [end1size,~] = size(MS.node(i-1).effectors.action{end,1});
        
        if end1size
            if i > 1 && MS.node(i-1).effectors.action{end,1}.enabled
                A1 = Params.Machine.FeedRate;
            else
                A1 = 0;
            end
        else
            if MS.node(i-1).effectors.action{end,2}.enabled
                A1 = Params.Machine.FeedRate;
            else
                A1 = 0;
            end
        end
    else
        A1 = 0;
    end
    
    % Define A2
    if i > 1
        A2 = Params.Machine.RollGap;
    else
        A2 = 12;
    end
    
    % Define other axes for first action of node
    A3 = round(100*(MS.node(i).effectors.position{1,1}(2) - Params.Fixed.WheelDistNorm)*Params.Sheet.SheetSize)/100;
    A4 = round(100*(MS.node(i).effectors.position{1,1}(1) + Params.Fixed.xlimNorm)*Params.Sheet.SheetSize)/100;
    A5 = round(100*MS.node(i).effectors.position{1,1}(3))/100;
    A6 = round(100*(- MS.node(i).effectors.position{1,2}(2) - Params.Fixed.WheelDistNorm)*Params.Sheet.SheetSize)/100;
    A7 = round(100*(MS.node(i).effectors.position{1,2}(1) + Params.Fixed.xlimNorm)*Params.Sheet.SheetSize)/100;
    A8 = round(100*MS.node(i).effectors.position{1,2}(3))/100;
    
    MS.node(i).effectors.axes{1} = [A1 A2 A3 A4 A5 A6 A7 A8];
    [nr,~] = size(MS.node(i).effectors.action);
    
    % After each action
    for j = 1:nr
        
        % Determine effector action numbers
        [end1size,~] = size(MS.node(i).effectors.action{j,1});
        endsize = size(MS.node(i).effectors.action);
        if endsize(2) > 1
            [end2size,~] = size(MS.node(i).effectors.action{j,2});
        else
            end2size = 0;
        end
        
        % Define roll gap
        if i == 1 && j < 3
            A2 = 12;
        else
            A2 = Params.Machine.RollGap;
        end
        
        % Front effector active
        if end1size
            % Define feed rate
            if (MS.node(i).effectors.action{j,1}.enabled && strcmp(MS.node(i).effectors.action{j,1}.name,'translation')) || strcmp(MS.node(i).effectors.action{j,1}.name,'drive')
                A1 = Params.Machine.FeedRate;
            else
                A1 = 0;
            end
            
            % Axes A3 A4
            A3 = round(100*((MS.node(i).effectors.action{j,1}.position{2}(2) - Params.Fixed.WheelDistNorm)*Params.Sheet.SheetSize))/100;
            A4 = round(100*(MS.node(i).effectors.action{j,1}.position{2}(1) + Params.Fixed.xlimNorm)*Params.Sheet.SheetSize)/100;
            
            % Axes A5 A8
            if strcmp(MS.node(i).effectors.action{j,1}.name,'transfer')
                A5 = 50;
                A8 = 0;
            else
                A5 = round(100*MS.node(i).effectors.action{j,1}.position{2}(3))/100;
                A8 = MS.node(i).effectors.axes{j}(8);
            end
        %  Front effector inactive      
        else
            A3 = MS.node(i).effectors.axes{j}(3);
            A4 = MS.node(i).effectors.axes{j}(4);
        end
        
        % Back effector active
        if end2size
            % Define feed rate
            if (MS.node(i).effectors.action{j,2}.enabled && strcmp(MS.node(i).effectors.action{j,2}.name,'translation')) || strcmp(MS.node(i).effectors.action{j,2}.name,'drive')
                A1 = Params.Machine.FeedRate;
            else
                A1 = 0;
            end
            
            % Axes A6 A7
            A6 = round(100*(-MS.node(i).effectors.action{j,2}.position{2}(2) - Params.Fixed.WheelDistNorm)*Params.Sheet.SheetSize)/100;
            A7 = round(100*(MS.node(i).effectors.action{j,2}.position{2}(1) + Params.Fixed.xlimNorm)*Params.Sheet.SheetSize)/100;
            
            % Axes A5 A8
            if strcmp(MS.node(i).effectors.action{j,2}.name,'transfer')
                A5 = 0;
                A8 = 50;
            else
                A5 = MS.node(i).effectors.axes{j}(5);
                A8 = MS.node(i).effectors.action{j,2}.position{2}(3);
            end
        % Back effector inactive    
        else
            A6 = MS.node(i).effectors.axes{j}(6);
            A7 = MS.node(i).effectors.axes{j}(7);
        end      
        
        % Set value
        MS.node(i).effectors.axes{j+1} = [A1 A2 A3 A4 A5 A6 A7 A8];

    end
end