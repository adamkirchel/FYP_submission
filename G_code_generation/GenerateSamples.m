function [solution_set,raw_data,area_data] = GenerateSamples(bound,ns,name,delta,sm,len,mina)
% % Generates samples using a modified latin hypercube technique
%
% Inputs:
% bound - Boundary values (nx2 array)
% ns - Number of samples (int)
% name - Name of strategy (String)
% delta - Wheeling boundary (double)
% sm - Number of points per dimension (int)
% len - Length of sheet (double)
% mina - Minimum area (double)
%
% Outputs:
% solution_set - Samples selected (nxm array)
% raw_data - Sample space after minimum area applied (nxm array)
% area_data - Sample space before minimum area applied (nxm array)

% Define parameters
range = 1/(ns+1);
remaining_points = zeros((sm)^(length(bound(:,1))),length(bound(:,1)));
minarea = ((len - 2*delta)^2) * (mina/100);

% Create initial sample space
for i = 1:length(remaining_points(:,1))
    vec = zeros(1,length(bound(:,1)));
    index = zeros(1,length(bound(:,1)));
    num = i;
    
    for j = 1:length(vec)
        modulus = mod(num,sm);
        if modulus == 0
            modulus = sm;
        end
        num = ceil(num/(sm));
        vec(end-j+1) = bound(end-j+1,1) + ((modulus-1)/(sm-1))*((bound(end-j+1,2) - bound(end-j+1,1)));
    end
    
    %index(index==0) = 1;
    
    remaining_points(i,:) = vec;
    
end

% Reduce sample space based upon restrictions
switch name
    case 'Centre to outside'
        xpts = remaining_points(:,3) - ((remaining_points(:,1)/2).*remaining_points(:,5));
        ypts = remaining_points(:,4) - ((remaining_points(:,2)/2).*remaining_points(:,5));
        remaining_points(xpts < delta | ypts < delta,:) = [];
        area_data = remaining_points;
        
        area = remaining_points(:,1).*remaining_points(:,2).*(remaining_points(:,5).^(2));
        remaining_points(area < minarea,:) = [];
    case 'Vertical expand'
        xpts = remaining_points(:,2) - ((remaining_points(:,1)/2).*remaining_points(:,3));
        remaining_points(xpts < delta,:) = [];
        area_data = remaining_points;
        
        area = remaining_points(:,1).*(len - 2*delta).*(remaining_points(:,3).^(2));
        remaining_points(area < minarea,:) = [];
    case 'Horizontal expand'
        ypts = remaining_points(:,2) - ((remaining_points(:,1)/2).*remaining_points(:,3));
        remaining_points(ypts < delta,:) = [];
        area_data = remaining_points;
        
        area = remaining_points(:,1).*(len - 2*delta).*(remaining_points(:,3).^(2));
        remaining_points(area < minarea,:) = [];
    case 'Overlayed'
        xpts = remaining_points(:,3) - (remaining_points(:,1)/2);
        ypts = remaining_points(:,4) - (remaining_points(:,2)/2);
        remaining_points(xpts < delta | ypts < delta,:) = [];
        area_data = remaining_points;
        
        area = remaining_points(:,1).*remaining_points(:,2);
        remaining_points(area < minarea,:) = [];
    case 'Triangle'
        xpts = remaining_points(:,3) - (remaining_points(:,1)/2);
        ypts = -remaining_points(:,4) - remaining_points(:,2) + len;
        remaining_points(xpts < delta | ypts < delta | remaining_points(:,4) < delta,:) = [];
        area_data = remaining_points;
        
        area = 0.5 * remaining_points(:,1).*remaining_points(:,2);
        remaining_points(area < minarea,:) = [];
end

norm_points = zeros(size(remaining_points));
raw_data = remaining_points;

% Normalise points
for i = 1:length(remaining_points(1,:))
    norm_points(:,i) = (remaining_points(:,i) - min(remaining_points(:,i)))/(max(remaining_points(:,i)) - min(remaining_points(:,i)));
end

% Restructure samples
% figure()
% scatter3(remaining_points(:,1),remaining_points(:,2),remaining_points(:,3),'.')

% Set base case
switch name
    case 'Centre to outside'
        int_pts = remaining_points(remaining_points(:,1) == max(remaining_points(:,1)) & remaining_points(:,2) == max(remaining_points(:,2)),:);
        base_ind = remaining_points(:,1) == max(remaining_points(:,1)) & remaining_points(:,2) == max(remaining_points(:,2)) & remaining_points(:,5) == max(int_pts(:,5));
    case 'Vertical expand'
        int_pts = remaining_points(remaining_points(:,1) == max(remaining_points(:,1)) & remaining_points(:,2) == len/2,:);
        base_ind = remaining_points(:,1) == max(remaining_points(:,1)) & remaining_points(:,2) == len/2 & remaining_points(:,3) == max(int_pts(:,3));
    case 'Horizontal expand'
        int_pts = remaining_points(remaining_points(:,1) == max(remaining_points(:,1)) & remaining_points(:,2) == len/2,:);
        base_ind = remaining_points(:,1) == max(remaining_points(:,1)) & remaining_points(:,2) == len/2 & remaining_points(:,3) == max(int_pts(:,3));
    case 'Overlayed'
        base_ind = remaining_points(:,1) == max(remaining_points(:,1)) & remaining_points(:,2) == max(remaining_points(:,2));
    case 'Triangle'
        base_ind = remaining_points(:,1) == max(remaining_points(:,1)) & remaining_points(:,2) == max(remaining_points(:,2));
end

solution_set = [];
% c_ind = ceil(length(remaining_points(:,1))/2);

% Set initial sample as base case
solution_set(1,:) = remaining_points(base_ind,:);
x = norm_points(base_ind,:);

% Iteratively calculate next sample
for i = 1:ns-1
    for j = 1:length(bound(:,1))
        upper = x(j) + range/2;
        lower = x(j) - range/2;
        remaining_points(norm_points(:,j) < upper & norm_points(:,j) > lower,:) = [];
        norm_points(norm_points(:,j) < upper & norm_points(:,j) > lower,:) = [];
    end
    
    index = 0;
    target = 0;
    for j = 1:length(remaining_points(:,1))
        sum_dist = 0;
        
        for k = 1:length(solution_set(:,1))
            sum_dist = sum_dist + norm(remaining_points(j,:)-solution_set(k,:));
        end
        
        if sum_dist > target
            index = j;
            target = sum_dist;
        end
    end
    solution_set(i+1,:) = remaining_points(index,:);
    x = norm_points(index,:);

end

% figure()
% scatter3(remaining_points(:,1),remaining_points(:,2),remaining_points(:,3),'.')
% hold on
% scatter3(solution_set(:,1),solution_set(:,2),solution_set(:,3))

