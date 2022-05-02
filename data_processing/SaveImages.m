%% Normalise and save data in image format

path = 'C:\Users\adsk1\Documents\FYP\Python\data\geometry\curvature\p2\';
savepath = 'C:\Users\adsk1\Documents\FYP\Python\data\geometry_no_3\curvature\p2\';
filenames = dir(append(path,'csv'));

data = {};
new_max = 0;
new_min = 0;
t = 3;
% Find max and min of all files for normalisation
for i = 3:length(filenames)
    data{i-2} = csvread(append(path,'csv\',filenames(i).name));
    std = std2(data{i-2});
    m = mean2(data{i-2});
    mx = max(data{i-2},[],'all');
%     [~,ii] = max(data{i-2});
%     out = data{i-2}(ii);
    mn = -max(-data{i-2},[],'all');
    
    if mx > new_max && m + t*std > new_max
        if mx < m + t*std
            new_max = mx;
        else
            new_max = m + t*std;
        end
    end
    
    if mn < new_min && m - t*std < new_min
        if mn > m - t*std
            new_min = mn;
        else
            new_min = m - t*std;
        end
    end
end

% Normalise images and write to file
for i = 1:length(data)
    x = split(filenames(i+2).name,'_');
    file = x{1};
    maxval = data{i};
    maxval = (maxval - new_min)./(new_max - new_min);
    
    if str2double(x{2}(1)) > 3
        new_name = append(savepath,'images\',file(1:end-1),'\',file,'_0_',x{2}(1),'.jpg');
        imwrite(maxval,new_name);

        if str2double(x{2}(1)) == 4
            continue
        else
            n = str2double(x{2}(1)) - 4;

            for j = 1:n
                name = append(file,'_',num2str(j),'.csv');
                minval = csvread(append(path,'csv\',name));
                minval = (minval - new_min)./(new_max - new_min);
                diff = maxval - minval;

                new_name = append(savepath,'images\',file(1:end-1),'\',file,'_',num2str(j),'_',x{2}(1),'.jpg');
                imwrite(diff,new_name)
            end
        end
    end
end