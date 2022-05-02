classdef PreProcess
    properties
        TCloud
        polydata
        polymodel
        Gauss
        Mean
        Pmax
        Pmin
        mag
        inc
    end
    methods
        function [obj] = PreProcess(markers,ptcloud)
            % Initiate orientated data
            obj = obj.Orientate(markers,ptcloud);
        end
            
        function [obj] = Orientate(obj,markers,ptcloud)
            % OrientateData() - orientates a point cloud to fit to standardised axes
            % 
            % Inputs:
            % markers - Marker positions (.p3 file path)
            % ptcloud - Point cloud (.asc file path)
            %
            % Outputs:
            % T_cloud - Transformed point cloud (nx3 array)
            
            % Load data %
            
            % Load markers
            fid = fopen(markers);  % open the text file
            S = textscan(fid,'%s');   % text scan the data
            fclose(fid) ;      % close the file
            S = S{1} ;
            data = cellfun(@(x)str2double(x), S);  % convert the cell array to double
            data(isnan(data))=[];

            data = data(3:end);
            markers = zeros(length(data)/6,3);

            % Format markers
            for i = 1:length(data)/6
                markers(i,1) = data((i*6)-5);
                markers(i,2) = data((i*6)-4);
                markers(i,3) = data((i*6)-3);
            end

            % Load point cloud
            fid = fopen(ptcloud);  % open the text file
            S = textscan(fid,'%s');   % text scan the data
            fclose(fid) ;      % close the file
            S = S{1} ;
            data = cellfun(@(x)str2double(x), S);  % convert the cell array to double
            data(isnan(data))=[];

            % data = data(1:end);
            ptcloud = zeros(length(data)/3,3);

            % Format point cloud
            for i = 1:length(data)/6
                ptcloud(i,1) = data((i*6)-5);
                ptcloud(i,2) = data((i*6)-4);
                ptcloud(i,3) = data((i*6)-3);
            end

            % Identify nodal points %

            % Specify maximum distance between nodes
            nodedist = 21;

            nodelist = {};
            sheetlist = [];
            lst = [];

            % Node list markers index
            n = 1;

            % Sheet list markers index
            m = 1;

            for i = 1:length(markers)

                % Search for 5 nearest nodes
                idx = knnsearch(markers,markers(i,:),'k',5);

                % Nearest node
                knnmarker = markers(idx(2),:);

                % Distance of nearest node
                knndist = sqrt(((markers(i,1) - knnmarker(1))^2) + ((markers(i,2) - knnmarker(2))^2) + ((markers(i,3) - knnmarker(3))^2));

                % Marker is on sheet
                if knndist > nodedist

                    sheetlist(m,:) = markers(i,:);
                    m = m + 1;

                % Marker is on node
                elseif knndist <= nodedist
                    % Check it isn't in a node that has been found already
                    if ismember(markers(i,:),lst)
                        continue
                    else
                        nodeinst = [];
                        % Add points for node detected
                        for j = 1:length(idx)
                            dist = sqrt(((markers(i,1) - markers(idx(j),1))^2) + ((markers(i,2) - markers(idx(j),2))^2) + ((markers(i,3) - markers(idx(j),3))^2));

                            if dist <= nodedist
                                nodeinst(j,:) = markers(idx(j),:);
                            else
                                break
                            end
                        end

                        nodelist{n} = nodeinst;
                        lst = [lst;markers(idx,:)];
                        n = n + 1;
                    end

                end

            end

            % Calculate centre points %

            centrepts = zeros(3,3);

            for i = 1:length(nodelist)
                minsum = 1000;

                for j = 1:length(nodelist{i})
                    centresum = 0;
                    % Calculate distance of each marker from surrounding markers
                    for k = 1:length(nodelist{i})
                        dist = sqrt(((nodelist{i}(j,1) - nodelist{i}(k,1))^2) + ((nodelist{i}(j,2) - nodelist{i}(k,2))^2) + ((nodelist{i}(j,3) - nodelist{i}(k,3))^2));
                        centresum = centresum + dist;
                    end

                    % Find smallest cumulative distance - central marker
                    if centresum < minsum
                        minsum = centresum;
                        centrepts(i,:) = nodelist{i}(j,:);
                    end
                end
            end

            % Locate nodes as either origin, x axis or y axis %
            nodepos = zeros(3,3);

            for i = 1:length(nodelist)
                % Origin node
                if length(nodelist{i}) > 4
                    nodepos(1,:) = centrepts(i,:);
                % y axis node
                elseif length(nodelist{i}) > 3
                    nodepos(2,:) = centrepts(i,:);
                % x axis node
                else
                    nodepos(3,:) = centrepts(i,:);
                end
            end

            % Work out transformations %

            % Translation to [0 0 0]
            translation = -nodepos(1,:);
            xaxisnode = nodepos(3,:) + translation;
            yaxisnode = nodepos(2,:) + translation;

            % sqrt((xaxisnode(1)^2) + (xaxisnode(2)^2) + (xaxisnode(3)^2))
            % sqrt((yaxisnode(1)^2) + (yaxisnode(2)^2) + (yaxisnode(3)^2))

            % x axis rotation
            theta1 = -atan(xaxisnode(2)/xaxisnode(3));
            rot1 = [1 0 0; 0 cos(theta1) -sin(theta1); 0 sin(theta1) cos(theta1)];
            a1 = rot1' * yaxisnode';
            b1 = rot1' * xaxisnode';

            % y axis rotation
            theta2 = -atan(b1(3)/b1(1));
            rot2 = [cos(theta2) 0 sin(theta2); 0 1 0; -sin(theta2) 0 cos(theta2)];
            a2 = rot2' * a1;
            b2 = rot2' * b1;

            % x axis rotation
            theta3 = pi + atan(a2(3)/a2(2));
            rot3 = [1 0 0; 0 cos(theta3) -sin(theta3); 0 sin(theta3) cos(theta3)];
            a3 = rot3' * a2;
            b3 = rot3' * b2;

            % Rotation matrix
            R = rot1 * rot2 * rot3;

            % Homogeneous transformation matrix
            T = [R -translation'; 0 0 0 1];

            % Inverse
            T_inv = CalculateTinv(T);

            % Compute marker transformation %

            T_markers = zeros(length(markers(:,1)),length(markers(1,:)));

            for i = 1:length(markers(:,1))
                 pos = T_inv * [markers(i,:)' ; 1]; 
                 T_markers(i,:) = pos(1:3)';
            end

            % Compute point cloud transformation %

            T_cloud = zeros(length(ptcloud(:,1)),length(ptcloud(1,:)));

            for i = 1:length(ptcloud(:,1))
                 pos = T_inv * [ptcloud(i,:)' ; 1]; 
                 T_cloud(i,:) = pos(1:3)';
            end

            % Post-processing %

            % Remove ground
            T_cloud = T_cloud(T_cloud(:,3) > 2,:);
            norms = zeros(length(T_cloud(:,1)),1);

            % Remove outliers (needs work)
            for i = 1:length(norms)
                norms(i) = norm(T_cloud(i,:),1);
            end

            mean_norm = mean(norms);
            T_cloud = T_cloud(norms<880,:);

            % Define corners of sheet
            close all
            fig = figure();
            title('Select top left corner then bottom right')
            plot(T_cloud(:,1),T_cloud(:,2),'.')
            [xpts,ypts] = getpts(fig);

            close all

            T_cloud = T_cloud(T_cloud(:,1) > xpts(1) & T_cloud(:,2) < ypts(1) & T_cloud(:,2) > ypts(2) & T_cloud(:,1) < xpts(2),:);
            obj.TCloud = T_cloud - [xpts(1),ypts(1),0];

        end
        
        function [] = PtCloudShow(obj)
            % Visualise current point cloud form%
            
            close all
            figure()
            %ptCloud = pointCloud(obj.TCloud);
            pcshow(obj.TCloud)
            xlabel('x (mm)')
            ylabel('y (mm)')
            zlabel('z (mm)')
            colormap(hsv)
            cbar = colorbar;
            cbar.Label.String = "Z axis (mm)";
            cbar.Color = "w";
        end
        
        function [obj] = Poly(obj,mag,inc)
            % Convert to polynomial patch
            
            obj.mag = mag;
            obj.inc = inc;
            
            obj.polymodel = polyfitn(obj.TCloud(:,1:2),obj.TCloud(:,3),mag);

            if exist('sympoly') == 2
              polyn2sympoly(obj.polymodel);
            end
            if exist('sym') == 2
              polyn2sym(obj.polymodel);
            end

            % Determine size of new grid
            xmin = ceil(min(obj.TCloud(:,1)));
            xmax = floor(max(obj.TCloud(:,1)));
            ymin = ceil(min(obj.TCloud(:,2)));
            ymax = floor(max(obj.TCloud(:,2)));

            % Create new grid
            [xn,yn] = meshgrid(xmin:inc:xmax,ymin:inc:ymax);
            zn = polyvaln(obj.polymodel,[xn(:),yn(:)]);
            zn = reshape(zn,size(xn));
            obj.polydata = {xn, yn, zn};
            % Store data
            %polydata = [reshape(xn,[numel(xn),1]) reshape(yn,[numel(yn),1]) reshape(zn,[numel(zn),1])];
        end
        
        function [] = PolyShow(obj)
            % Show
%             % 3D plot surface with point cloud
%             figure
%             subplot(2,2,[1 2])
%             surf(obj.polydata{1},obj.polydata{2},obj.polydata{3},obj.polydata{3})
%             hold on
%             daspect([1 1 1])
%             vals = 500; % ,<-- how many vals to plot from raw data
%             vals = floor(linspace(1,length(obj.TCloud(:,2)),vals));
%             scatter3(obj.TCloud(vals,1),obj.TCloud(vals,2),obj.TCloud(vals,3))
%             xlabel('x')
%             ylabel('y')
%             zlabel('z')
%             title(append('Poly',num2str(obj.mag),num2str(obj.mag),' fit Surface with sample of Raw data'))
            
            % Polynomial surface top
            figure
            surf(obj.polydata{1},obj.polydata{2}+290,obj.polydata{3},obj.polydata{3},'EdgeColor','none')
            view(2)
            xlabel('x (mm)')
            ylabel('y (mm)')
            xlim([0 300])
            ylim([0 300])
            %title(append('Poly',num2str(obj.mag),num2str(obj.mag),' fit Surface'))
            hcb = colorbar;
            hcb.Title.String = "Z axis (mm)";
            axis equal
            
            % Point cloud top
            figure()
            scatter3(obj.TCloud(:,1),obj.TCloud(:,2)+290,obj.TCloud(:,3),[],obj.TCloud(:,3))
            view(2)
            xlabel('x (mm)')
            ylabel('y (mm)')
            xlim([0 300])
            ylim([0 300])
            %title('Raw data')
            hcb = colorbar;
            hcb.Title.String = "Z axis (mm)";
            axis equal
        end
        
        function [obj] = Curvature(obj)
            % Calculate curvature
            [obj.Gauss,obj.Mean,obj.Pmax,obj.Pmin] = surfature(obj.polydata{1},obj.polydata{2},obj.polydata{3});
        end
        
        function [] = ShowCurvature(obj,type)
            % Visualise curvature
            switch type
                case 'Gauss'
                    data = obj.Gauss;
                case 'Mean'
                    data = obj.Mean;
                case 'P1'
                    data = obj.Pmax;
                case 'P2'
                    data = obj.Pmin;
            end
            
            %figure()
            surf(obj.polydata{1},obj.polydata{2}+290,obj.polydata{3},data,'EdgeColor','none')
            daspect([1 1 1])
            view(2)
            xlabel('x (mm)')
            ylabel('y (mm)')
            xlim([0 300])
            ylim([0 300])
            %title(append(type,' Curvature'))
            colorbar
            axis equal
        end
        
        function [] = SaveCurvAsMat(obj,dir,name,type)
            % Save curvature values as a matrix
            filename = append(dir,'\',name,'.csv');
            switch type
                case 'Gauss'
                    writematrix(obj.Gauss,filename{1},'Delimiter',',');
                case 'Mean'
                    writematrix(obj.Mean,filename{1},'Delimiter',',');
                case 'Pmax'
                    writematrix(obj.Pmax,filename{1},'Delimiter',',');
                case 'Pmin'
                    writematrix(obj.Pmin,filename{1},'Delimiter',',');
            end
        end
        
        function [] = SaveCurvAsImg(obj,dir,name,type)
            % Save curvature data as an image
            filename = append(dir,'\',name,'.jpg');
            switch type
                case 'Gauss'
                    imwrite(obj.Gauss,filename{1});
                case 'Mean'
                    imwrite(obj.Mean,filename{1});
                case 'Pmax'
                    imwrite(obj.Pmax,filename{1});
                case 'Pmin'
                    imwrite(obj.Pmin,filename{1});
            end
        end
        
        function [] = SavePolyAsMat(obj,dir,name)
            % Save polynomial data as matrix
            filename = append(dir,'\',name,'.csv');
            writematrix(obj.polydata,filename{1},'Delimiter',',');
        end
        
        function [] = SavePolyAsImg(obj,dir,name)
            % Save polynomial data as image
            filename = append(dir,'\',name,'.jpg');
            imwrite(obj.polydata,filename{1});
        end
        
        function [] = SaveRaw(obj,dir,name)
            % Save raw data as a matrix
            filename = append(dir,'\',name,'.csv');
            writematrix(obj.TCloud,filename{1},'Delimiter',',');
        end
        
    end
end