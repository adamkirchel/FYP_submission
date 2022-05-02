function[MS] = AddNodes(Params,PassNum)
% % Determines position of all nodes for the given wheel path
% 
% Inputs:
% Params - Parameters defining the wheel path (struct)
% PassNum - Number of passes (int)
%
% Outputs:
% MS - Manufacturing strategy (struct)

% Indicates initial transfer
transfer1 = 0;

% Switch strategies
switch Params.Strategy.Name
    case 'Centre to outside'
        
        % Initialise width and height of pattern
        width = ((Params.Strategy.Value.ScalingFactor-1) * ((PassNum-1)/(Params.Strategy.Value.NumPasses-1)) * Params.Strategy.Norm.MinWidth) + Params.Strategy.Norm.MinWidth;
        height = ((Params.Strategy.Value.ScalingFactor-1) * ((PassNum-1)/(Params.Strategy.Value.NumPasses-1)) * Params.Strategy.Norm.MinHeight) + Params.Strategy.Norm.MinHeight;
        centrex = Params.Strategy.Norm.Centrex;
        centrey = Params.Strategy.Norm.Centrey;

        % Calculate number of nodes
        Params.NumNodes = 2*floor(width/Params.Path.SpacingNorm);

        % Initial wheel path node
        WheelPath = zeros(Params.NumNodes + 1,2);
        WheelPath(1,1) = Params.Strategy.Norm.Centrex - width/2;
        
    case 'Diagonal'
        
        % Initialise width and height of pattern
        width = Params.Strategy.Norm.Width;
        height = Params.Strategy.Norm.Height;
        centrex = ((1-(2*Params.Strategy.Norm.Startx)) * ((PassNum-1)/(Params.Strategy.Value.NumPasses-1))) + Params.Strategy.Norm.Startx;
        centrey = ((1-(2*Params.Strategy.Norm.Starty)) * ((PassNum-1)/(Params.Strategy.Value.NumPasses-1))) + Params.Strategy.Norm.Starty;
        
        % Calculate number of nodes
        Params.NumNodes = 2*floor(Params.Strategy.Norm.Width/Params.Path.SpacingNorm);

        % Initial wheel path node
        WheelPath = zeros(Params.NumNodes + 1,2);
        WheelPath(1,1) = centrex - Params.Strategy.Norm.Width/2;
        
    case 'Vertical expand'
        
        % Initialise width and height of pattern
        width = ((Params.Strategy.Value.ScalingFactor-1) * ((PassNum-1)/(Params.Strategy.Value.NumPasses-1)) * Params.Strategy.Norm.MinWidth) + Params.Strategy.Norm.MinWidth;
        height = 1 - 2*Params.Path.WheelBoundNorm;
        centrex = Params.Strategy.Norm.Centrex;
        centrey = 0.5;

        % Calculate number of nodes
        Params.NumNodes = 2*floor(width/Params.Path.SpacingNorm);

        % Initial wheel path node
        WheelPath = zeros(Params.NumNodes + 1,2);
        WheelPath(1,1) = Params.Strategy.Norm.Centrex - width/2;
        
    case 'Horizontal expand'
        
        % Initialise width and height of pattern
        width = 1 - 2*Params.Path.WheelBoundNorm;
        height = ((Params.Strategy.Value.ScalingFactor-1) * ((PassNum-1)/(Params.Strategy.Value.NumPasses-1)) * Params.Strategy.Norm.MinHeight) + Params.Strategy.Norm.MinHeight;
        centrex = 0.5;
        centrey = Params.Strategy.Norm.Centrey;
        
        % Calculate number of nodes
        Params.NumNodes = 2*floor(width/Params.Path.SpacingNorm);

        % Initial wheel path node
        WheelPath = zeros(Params.NumNodes + 1,2);
        WheelPath(1,1) = Params.Path.WheelBoundNorm;
        
    case 'Overlayed'
        
        % Initialise width and height of pattern
        width = Params.Strategy.Norm.Width;
        height = Params.Strategy.Norm.Height;
        centrex = Params.Strategy.Norm.Centrex;
        centrey = Params.Strategy.Norm.Centrey;
        
        % Calculate number of nodes
        Params.NumNodes = 2*floor(Params.Strategy.Norm.Width/Params.Path.SpacingNorm);

        % Initial wheel path node
        WheelPath = zeros(Params.NumNodes + 1,2);
        WheelPath(1,1) = Params.Strategy.Norm.Centrex - Params.Strategy.Norm.Width/2;
        
    case 'Track horizontal'
        
        % Initialise width and height of pattern
        width = Params.Strategy.Norm.MinWidth + (Params.Strategy.Norm.MaxWidth - Params.Strategy.Norm.MinWidth)*((PassNum - 1)/(Params.Strategy.Value.NumPasses - 1));
        height = 1 - 2*Params.Path.WheelBoundNorm;
        centrey = 0.5;
        
        % Calculate number of nodes
        Params.NumNodes = 2*floor(width/Params.Path.SpacingNorm);

        % Initial wheel path node
        WheelPath = zeros(Params.NumNodes + 1,2);
        WheelPath(1,1) = Params.Path.WheelBoundNorm;
        
    case 'Track vertical'
        
        % Initialise width and height of pattern
        height = Params.Strategy.Norm.MinHeight + (Params.Strategy.Norm.MaxHeight - Params.Strategy.Norm.MinHeight)*((PassNum - 1)/(Params.Strategy.Value.NumPasses - 1));
        width = 1 - 2*Params.Path.WheelBoundNorm;
        centrex = 0.5;
        centrey = (height/2) + Params.Path.WheelBoundNorm;
        
        % Calculate number of nodes
        Params.NumNodes = 2*floor(width/Params.Path.SpacingNorm);

        % Initial wheel path node
        WheelPath = zeros(Params.NumNodes + 1,2);
        WheelPath(1,1) = Params.Path.WheelBoundNorm;
        
    case 'Converging'
        
    case 'Triangle'
        
        % Initialise width and height of pattern
        height = Params.Strategy.Norm.Height * (PassNum/Params.Strategy.Value.NumPasses);
        width = Params.Strategy.Norm.BaseWidth * (1 - ((PassNum - 1)/(Params.Strategy.Value.NumPasses)));
        centrex = Params.Strategy.Norm.BaseCentrex;
        centrey = Params.Strategy.Norm.BaseCentrey + (height/2);
        
        % Calculate number of nodes
        Params.NumNodes = 2*floor(width/Params.Path.SpacingNorm);

        % Initial wheel path node
        WheelPath = zeros(Params.NumNodes + 1,2);
        WheelPath(1,1) = centrex - (width/2);
        
end

if Params.Process.Discrete
    
    height = round((height*Params.Sheet.SheetSize)/(2*Params.Process.Resolution))*(2*Params.Process.Resolution/Params.Sheet.SheetSize);
    width = round((width*Params.Sheet.SheetSize)/(2*Params.Process.Resolution))*(2*Params.Process.Resolution/Params.Sheet.SheetSize);
    centrex = round((centrex*Params.Sheet.SheetSize)/(Params.Process.Resolution))*(Params.Process.Resolution/Params.Sheet.SheetSize);
    centrey = round((centrey*Params.Sheet.SheetSize)/(Params.Process.Resolution))*(Params.Process.Resolution/Params.Sheet.SheetSize);
    
    Params.NumNodes = 2*floor(width/Params.Path.SpacingNorm);
    startx = WheelPath(1,1);
    WheelPath = zeros(Params.NumNodes + 1,2);
    WheelPath(1,1) = round((startx*Params.Sheet.SheetSize)/(Params.Process.Resolution))*(Params.Process.Resolution/Params.Sheet.SheetSize);
end

% Starting point at the drive through distance
WheelPath(1,2) = Params.Path.DriveDistNorm;

% Check if transfer is possible for second node
if centrey + (height/2) > Params.Path.EffBoundNorm + Params.Fixed.WheelDistNorm
    WheelPath(2,1) = WheelPath(1,1);
    WheelPath(2,2) = Params.Path.EffBoundNorm + Params.Fixed.WheelDistNorm;
    transfer1 = 1;
    start_index = 3;
else
    start_index = 2;
end

% First pivot point
WheelPath(start_index,1) = WheelPath(1,1);
WheelPath(start_index,2) = centrey + height/2;

if Params.Process.Discrete
    WheelPath(start_index,2) = round((WheelPath(start_index,2)*Params.Sheet.SheetSize)/(Params.Process.Resolution))*(Params.Process.Resolution/Params.Sheet.SheetSize);
end

% Calculate pattern for subsequent nodes
for i = start_index + 1:Params.NumNodes + 1
    % Bottom node
    if mod(i,2) == 0 && start_index == 3 || mod(i,2) ~= 0 && start_index == 2
        WheelPath(i,1) = WheelPath(i-1,1) + Params.Path.SpacingNorm;
        WheelPath(i,2) = WheelPath(i-1,2) - height;

    % Top node
    else
        WheelPath(i,1) = WheelPath(i-1,1);
        WheelPath(i,2) = WheelPath(i-1,2) + height;
    end

end

if Params.Process.Discrete
    maxpt = WheelPath(start_index,1) + width;
else
    maxpt = WheelPath(end,1);
end

% Define area of wheel
MS.AreaCorners = [WheelPath(start_index,1) WheelPath(start_index,2); ...
    maxpt WheelPath(start_index,2); ...
    maxpt min(WheelPath(start_index:end,2)); ...
    WheelPath(start_index,1) min(WheelPath(start_index:end,2)); ...
    WheelPath(start_index,1) WheelPath(start_index,2)];

if Params.Process.Discrete
    NumCells = 1/Params.Process.ResolutionNorm;
    xmax = maxpt/Params.Process.ResolutionNorm;
    xmin = WheelPath(start_index,1)/Params.Process.ResolutionNorm;
    ymin = min(WheelPath(start_index:end,2))/Params.Process.ResolutionNorm;
    ymax = WheelPath(start_index,2)/Params.Process.ResolutionNorm;
else
    % Create density estimation array
    NumCells = 1/Params.Path.SpacingNorm;
    xmax = maxpt/Params.Path.SpacingNorm;
    xmin = WheelPath(start_index,1)/Params.Path.SpacingNorm;
    ymin = min(WheelPath(start_index:end,2))/Params.Path.SpacingNorm;
    ymax = WheelPath(start_index,2)/Params.Path.SpacingNorm;
end

MS.DiscretePath = [];

index = 1;

for i = 1:xmax - xmin
    for j = 1:ymax - ymin
        if Params.Process.Discrete
            MS.DiscretePath(index,:) = [(xmin+i-1)*Params.Process.ResolutionNorm,(ymin+j-1)*Params.Process.ResolutionNorm];
        else
            MS.DiscretePath(index,:) = [(xmin+i-1)*Params.Path.SpacingNorm,(ymin+j-1)*Params.Path.SpacingNorm];
        end
        index = index + 1;
    end
end


% Initial position
MS.node(1).datum.global.position = [WheelPath(1,1);WheelPath(1,2);0];
MS.node(1).datum.global.angle = 90;
MS.node(1).enabled = [1,0];

% If transfer occured on second node
if transfer1
    MS.node(2).datum.global.position = [WheelPath(2,1);WheelPath(2,2);0];
    MS.node(2).datum.global.angle = 90;
    MS.node(2).enabled = [0,1];
    start_index = 3;
    n = 3;
else
    start_index = 2;
    n = 2;
end

% Loop through all wheel path points
for i = start_index:length(WheelPath(:,1))-1
    
    % Current node position
    MS.node(n).datum.global.position = [WheelPath(i,1);WheelPath(i,2);0];
    current_angle = CalculateAbsAngle(WheelPath,i);  
    MS.node(n).datum.global.angle = current_angle;
    
    % Next angle
    next_angle = CalculateAbsAngle(WheelPath,i+1);  
    
    % Determine relative position of end effector to limit
    % Bottom node
    if mod(i,2) == 0 && start_index == 3 || mod(i,2) ~= 0 && start_index == 2
        % Front end effector active
        if MS.node(n-1).enabled(1)
            xglobal = WheelPath(i,1) - Params.Fixed.WheelDistNorm * cosd(next_angle);
            yglobal = WheelPath(i,2) - Params.Fixed.WheelDistNorm * sind(next_angle);
            xrel = xglobal - Params.Path.EffBoundNorm;
            yrel = yglobal - Params.Path.EffBoundNorm;
        % Back end effector active
        else
            xglobal = WheelPath(i,1) + Params.Fixed.WheelDistNorm * cosd(next_angle);
            yglobal = WheelPath(i,2) + Params.Fixed.WheelDistNorm * sind(next_angle);
            xrel = xglobal - Params.Path.EffBoundNorm;
            yrel = 1 - yglobal - Params.Path.EffBoundNorm;
        end
    % Top node
    else
        % Front end effector active
        if MS.node(n-1).enabled(1)
            xglobal = WheelPath(i,1) - Params.Fixed.WheelDistNorm * cosd(next_angle);
            yglobal = WheelPath(i,2) - Params.Fixed.WheelDistNorm * sind(next_angle);
            xrel = xglobal - Params.Path.EffBoundNorm;
            yrel = yglobal - Params.Path.EffBoundNorm;
        % Back end effector active
        else
            xglobal = WheelPath(i,1) + Params.Fixed.WheelDistNorm * cosd(next_angle);
            yglobal = WheelPath(i,2) + Params.Fixed.WheelDistNorm * sind(next_angle);
            xrel = xglobal - Params.Path.EffBoundNorm;
            yrel = 1 - yglobal - Params.Path.EffBoundNorm;
        end
    end 
    
    % Check if effector is within limits
    if xrel > 0 && yrel > 0
        % Complete transfer
        MS.node(n).enabled = ~MS.node(n-1).enabled;
        n = n + 1;
        continue
    else
        % No transfer
        MS.node(n).enabled = MS.node(n-1).enabled;
        n = n + 1;
    end
    
    % Work out equation of line between current and next node
    grad = (WheelPath(i,2) - WheelPath(i+1,2))/(WheelPath(i,1) - WheelPath(i+1,1));
    c = WheelPath(i,2) - grad*WheelPath(i,1);
    
    % Work out limiting position
    % Vertical line
    if abs(grad) > 1000
        % Front end effector active
        if MS.node(n-1).enabled(1)
            x_bound_1 = WheelPath(i,1);
            y_bound_1 = Params.Path.EffBoundNorm;
        else
            x_bound_1 = WheelPath(i,1);
            y_bound_1 = 1 - Params.Path.EffBoundNorm;
        end

    % Non-vertical line
    else
        % Front end effector active
        if MS.node(n-1).enabled(1)
            y_bound_1 = Params.Path.EffBoundNorm;
            x_bound_1 = (y_bound_1 - c)/grad;

            x_bound_2 = Params.Path.EffBoundNorm;
            y_bound_2 = grad*x_bound_2 + c;
        else
            y_bound_1 = 1 - Params.Path.EffBoundNorm;
            x_bound_1 = (y_bound_1 - c)/grad;

            x_bound_2 = Params.Path.EffBoundNorm;
            y_bound_2 = grad*x_bound_2 + c;
        end
    end
    
    % Determine global end effector position
    if x_bound_1 > 0
        pos_end_global = [x_bound_1; y_bound_1; 0];
    else
        pos_end_global = [x_bound_2; y_bound_2; 0];
    end
    
    % Work out global position of datum
    if MS.node(n-1).enabled(1)
        datum_pos = pos_end_global + [Params.Fixed.WheelDistNorm * cosd(next_angle); Params.Fixed.WheelDistNorm * sind(next_angle);0];
    else
        datum_pos = pos_end_global - [Params.Fixed.WheelDistNorm * cosd(next_angle); Params.Fixed.WheelDistNorm * sind(next_angle);0];
    end
    
    % Eliminate bug with value being equal to limit
    if abs(datum_pos(2) - MS.node(n-1).datum.global.position(2)) < 0.01
        datum_pos = MS.node(n-1).datum.global.position;
    end
    
    % Set transfer node if possible
    % Bottom node
    if mod(i,2) == 0 && start_index == 3 || mod(i,2) ~= 0 && start_index == 2

        if datum_pos(2) >=  MS.node(n-1).datum.global.position(2) && datum_pos(2) <= WheelPath(i+1,2)
            % Complete transfer
            MS.node(n).datum.global.angle = next_angle;
            MS.node(n).enabled = ~MS.node(n-1).enabled;
            MS.node(n).datum.global.position = datum_pos;
            n = n + 1;
        else
            continue
        end
    % Top node
    else
        
        if datum_pos(2) <=  MS.node(n-1).datum.global.position(2) && datum_pos(2) >= WheelPath(i+1,2)
            % Complete transfer
            MS.node(n).datum.global.angle = next_angle;
            MS.node(n).enabled = ~MS.node(n-1).enabled;
            MS.node(n).datum.global.position = datum_pos;
            n = n + 1;
        else
            continue
        end
    end  
end

% Final node position
MS.node(n).datum.global.position = [WheelPath(end,1);WheelPath(end,2);0];
MS.node(n).datum.global.angle = CalculateAbsAngle(WheelPath,length(WheelPath(:,1)));