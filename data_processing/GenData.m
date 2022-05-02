%%% Run this to generate and plot data
% *Take all this with an extra big pinch of salt*
%
% Remember, sheet should shape more in 'x' than in 'y'
clc;clear;close all

% Store all data collected/specified in 'Example' structure
Strat = struct();
%% Toggles

% How many points on Bez Patch
global npts 
npts = 125;

% Do plots
selPlots = 1:19;     % Which example to process
doPlot = false;
doImgs = false;
doinptsplot = false;

% Write to file
doWriteToFile = false;





%% Strat 1
p=1;
Strat(p).name = 'Center to outside (All)';

% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.4-(0.1*(i-1));
    Strat(p).Pass(i).Input.p(2) = 0.4-(0.1*(i-1));
    Strat(p).Pass(i).Input.p(3) = 0.6+(0.1*(i-1));
    Strat(p).Pass(i).Input.p(4) = 0.6+(0.1*(i-1));
end


% Now define output as Bez patch
for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0 0.2 0.2 0;
        0.5 1 1 0.5;
        0.5 1 1 0.5;
        0 0.2 0.2 0].*(i^2/16);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
        2/3 0.6+(0.1*(i-1)) 0.6+(0.1*(i-1)) 2/3;
        1/3 0.4-(0.1*(i-1)) 0.4-(0.1*(i-1)) 1/3;
        0/3 0/3 0/3 0/3]';
    
    Strat(p).Pass(i).Output.Bezctr.x = [3/3 3/3 3/3 3/3;
        2/3 0.6+(0.1*(i-1)) 0.6+(0.1*(i-1)) 2/3;
        1/3 0.4-(0.1*(i-1)) 0.4-(0.1*(i-1)) 1/3;
        0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end


%% Strat 2
p = p + 1;
Strat(p).name = 'Center to outside (Corner)';

% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.3-(i*0.05);
    Strat(p).Pass(i).Input.p(2) = 0.3-(i*0.05);
    Strat(p).Pass(i).Input.p(3) = 0.3+(i*0.05);
    Strat(p).Pass(i).Input.p(4) = 0.3+(i*0.05);
end

for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0 0.5 0.5 0;
        0.2 1 1 0.2;
        0.2 1 1 0.2;
        0 0.5 0.5 0].*(i^2/32);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
        0.3+(0.05*(i-1)) 0.3+(0.05*(i-1)) 0.3+(0.05*(i-1)) 0.3+(0.05*(i-1));
        0.25-(0.05*(i-1)) 0.25-(0.05*(i-1)) 0.25-(0.05*(i-1)) 0.25-(0.05*(i-1));
        0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0/3 0.25-(0.05*(i-1)) 0.35+(0.05*(i-1)) 3/3;
        0/3 0.25-(0.05*(i-1)) 0.35+(0.05*(i-1)) 3/3;
        0/3 0.25-(0.05*(i-1)) 0.35+(0.05*(i-1)) 3/3;
        0/3 0.25-(0.05*(i-1)) 0.35+(0.05*(i-1)) 3/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end


%% Example 3
p = p + 1;
Strat(p).name = 'Center to outside (Up)';

% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.3-(i*0.05);
    Strat(p).Pass(i).Input.p(2) = 0.5-(i*0.1);
    Strat(p).Pass(i).Input.p(3) = 0.3+(i*0.05);
    Strat(p).Pass(i).Input.p(4) = 0.5+(i*0.1);
end

for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0 0.5 0.5 0;
        0.2 1 1 0.2;
        0.2 1 1 0.2;
        0 0.5 0.5 0].*(i^2/32);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
        0.5+(i*0.1) 0.5+(i*0.1) 0.5+(i*0.1) 0.5+(i*0.1);
        0.5-(i*0.1) 0.5-(i*0.1) 0.5-(i*0.1) 0.5-(i*0.1);
        0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0/3 0.3-(i*0.05) 0.3+(i*0.05) 3/3;
        0/3  0.3-(i*0.05) 0.3+(i*0.05) 3/3;
        0/3  0.3-(i*0.05) 0.3+(i*0.05) 3/3;
        0/3  0.3-(i*0.05) 0.3+(i*0.05) 3/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end

%% Example 4
p = p + 1;
Strat(p).name = 'Center to outside (Across)';

% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(2) = 0.3-(i*0.05);
    Strat(p).Pass(i).Input.p(1) = 0.5-(i*0.1);
    Strat(p).Pass(i).Input.p(4) = 0.3+(i*0.05);
    Strat(p).Pass(i).Input.p(3) = 0.5+(i*0.1);
end

for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0 0.5 0.5 0;
        0.2 1 1 0.2;
        0.2 1 1 0.2;
        0 0.5 0.5 0].*(i^2/32);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
        0.3+(i*0.05) 0.3+(i*0.05) 0.3+(i*0.05) 0.3+(i*0.05);
        0.3-(i*0.05) 0.3-(i*0.05) 0.3-(i*0.05) 0.3-(i*0.05);
        0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0/3 0.5-(i*0.1) 0.5+(i*0.1) 3/3;
        0/3  0.5-(i*0.1) 0.5+(i*0.1) 3/3;
        0/3  0.5-(i*0.1) 0.5+(i*0.1) 3/3;
        0/3  0.5-(i*0.1) 0.5+(i*0.1) 3/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end

%% Example 5
p = p + 1;
Strat(p).name = 'Center to outside (Off Centre)';

% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.4-(0.05*i);
    Strat(p).Pass(i).Input.p(2) = 0.4-(0.05*i);
    Strat(p).Pass(i).Input.p(3) = 0.5+(0.05*i);
    Strat(p).Pass(i).Input.p(4) = 0.5+(0.05*i);
end

for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0 0.5 0.5 0;
        0.2 1 1 0.2;
        0.2 1 1 0.2;
        0 0.5 0.5 0].*(i^2/26);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
        0.5+(0.05*i) 0.5+(0.05*i) 0.5+(0.05*i) 0.5+(0.05*i);
        0.4-(0.05*i) 0.4-(0.05*i) 0.4-(0.05*i) 0.4-(0.05*i);
        0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0/3 0.4-(0.05*i) 0.5+(0.05*i) 3/3;
        0/3  0.4-(0.05*i) 0.5+(0.05*i) 3/3;
        0/3  0.4-(0.05*i) 0.5+(0.05*i) 3/3;
        0/3  0.4-(0.05*i) 0.5+(0.05*i) 3/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end

%% Example 6 
p=p+1;
Strat(p).name = 'Horizontal (All)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.1;
    Strat(p).Pass(i).Input.p(2) = 0.4-(0.1*(i-1));
    Strat(p).Pass(i).Input.p(3) = 0.9;
    Strat(p).Pass(i).Input.p(4) = 0.6+(0.1*(i-1));
end

% Now define output as Bez patch
for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0   0.15 0.2  0.15 0;
        0.15 0.35 0.4  0.35 0.1;
        0.3  0.5  0.8  0.5  0.3;
        0.15 0.35 0.4  0.35 0.1;
        0    0.15 0.2  0.15 0;].*(i/6);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3 3/3;
        0.6+(0.05*(i-1)) 0.6+(0.05*(i-1)) 0.6+(0.05*(i-1)) 0.6+(0.05*(i-1)) 0.6+(0.05*(i-1));
        0.5 0.5 0.5 0.5 0.5;
        0.4-(0.05*(i-1)) 0.4-(0.05*(i-1)) 0.4-(0.05*(i-1)) 0.4-(0.05*(i-1)) 0.4-(0.05*(i-1));
        0/3 0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1];
    
    
    Strat(p).Pass(i).Output.Bezctr.order = 5;
end

%% Example 7
p=p+1;
Strat(p).name = 'Horizontal (Bottom)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.1;
    Strat(p).Pass(i).Input.p(2) = 0.3-(0.05*i);
    Strat(p).Pass(i).Input.p(3) = 0.9;
    Strat(p).Pass(i).Input.p(4) = 0.3+(0.05*i);
end


% Now define output as Bez patch
for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0   0.15 0.2  0.15 0;
        0.15 0.35 0.4  0.35 0.1;
        0.3  0.5  0.8  0.5  0.3;
        0.15 0.35 0.4  0.35 0.1;
        0    0.15 0.2  0.15 0;].*(i/6);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3 3/3;
        0.45+(0.05*(i-1)) 0.45+(0.05*(i-1)) 0.45+(0.05*(i-1)) 0.45+(0.05*(i-1)) 0.45+(0.05*(i-1));
        0.35 0.35 0.35 0.35 0.35;
        0.25-(0.05*(i-1)) 0.25-(0.05*(i-1)) 0.25-(0.05*(i-1)) 0.25-(0.05*(i-1)) 0.25-(0.05*(i-1));
        0/3 0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1];
    
    
    Strat(p).Pass(i).Output.Bezctr.order = 5;
end


%% Example 8
p=p+1;
Strat(p).name = 'Horizontal (Off Centre)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.1;
    Strat(p).Pass(i).Input.p(2) = 0.35-(0.075*(i-1));
    Strat(p).Pass(i).Input.p(3) = 0.9;
    Strat(p).Pass(i).Input.p(4) = 0.55+(0.075*(i-1));
end



% Now define output as Bez patch
for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0   0.15 0.2  0.15 0;
        0.15 0.35 0.4  0.35 0.1;
        0.3  0.5  0.8  0.5  0.3;
        0.15 0.35 0.4  0.35 0.1;
        0    0.15 0.2  0.15 0;].*(i/6);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3 3/3;
        0.5+(0.075*(i-1)) 0.5+(0.075*(i-1)) 0.5+(0.075*(i-1)) 0.5+(0.075*(i-1)) 0.5+(0.075*(i-1)) ;
        0.45 0.45 0.45 0.45 0.45;
        0.4-(0.075*(i-1)) 0.4-(0.075*(i-1)) 0.4-(0.075*(i-1)) 0.4-(0.075*(i-1)) 0.4-(0.075*(i-1)) ;
        0/3 0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1];
    
    
    Strat(p).Pass(i).Output.Bezctr.order = 5;
end



%% Example 9
p=p+1;
Strat(p).name = 'Repeated (All)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.1;
    Strat(p).Pass(i).Input.p(2) = 0.1;
    Strat(p).Pass(i).Input.p(3) = 0.9;
    Strat(p).Pass(i).Input.p(4) = 0.9;
end

for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0 0.5 0.5 0;
        0.2 1 1 0.2;
        0.2 1 1 0.2;
        0 0.5 0.5 0].*(i/6);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
         2/3 2/3 2/3 2/3;
         1/3 1/3 1/3 1/3;
         0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0/3 1/3 2/3 3/3;
         0/3 1/3 2/3 3/3;
         0/3 1/3 2/3 3/3;
         0/3 1/3 2/3 3/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end

%% Example 10
p=p+1;
Strat(p).name = 'Repeated (Corner)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.1;
    Strat(p).Pass(i).Input.p(2) = 0.1;
    Strat(p).Pass(i).Input.p(3) = 0.5;
    Strat(p).Pass(i).Input.p(4) = 0.5;
end

for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0 0.2 0.2 0;
        0.2 1 1 0.2;
        0.2 1 1 0.2;
        0 0.5 0.5 0].*(i/6);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
         0.5 0.5 0.5 0.5;
         0.1 0.1 0.1 0.1;
         0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0/3 0.1 0.5 3/3;
         0/3 0.1 0.5 3/3;
         0/3 0.1 0.5 3/3;
         0/3 0.1 0.5 3/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end

%% Example 11
p=p+1;
Strat(p).name = 'Repeated (Up)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.1;
    Strat(p).Pass(i).Input.p(2) = 0.1;
    Strat(p).Pass(i).Input.p(3) = 0.3;
    Strat(p).Pass(i).Input.p(4) = 0.9;
end

for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0   0.5 0.3 0;
         0.2 1   1   0.2;
         0.2 1   1   0.2;
         0   0.5 0.3 0].*(i/6);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
         0.7 0.7 0.7 0.7;
         0.3 0.3 0.3 0.3;
         0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0/3 0.1 0.4 3/3;
         0/3 0.1 0.4 3/3;
         0/3 0.1 0.4 3/3;
         0/3 0.1 0.4 3/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end

%% Example 12
p=p+1;
Strat(p).name = 'Repeated (Across)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.1;
    Strat(p).Pass(i).Input.p(2) = 0.1;
    Strat(p).Pass(i).Input.p(3) = 0.9;
    Strat(p).Pass(i).Input.p(4) = 0.3;
end


for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0   0.5 0.3 0;
         0.2 1   1   0.2;
         0.2 1   1   0.2;
         0   0.5 0.3 0].*(i/6);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
         0.3 0.3 0.3 0.3;
         0.3 0.1 0.1 0.1;
         0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0/3 0.3 0.7 3/3;
         0/3 0.3 0.7 3/3;
         0/3 0.3 0.7 3/3;
         0/3 0.3 0.7 3/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end


%% Example 13
p=p+1;
Strat(p).name = 'Triangle (All)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.0+(0.1*i);
    Strat(p).Pass(i).Input.p(2) = 0.1;
    Strat(p).Pass(i).Input.p(3) = 1-(0.1*i);
    Strat(p).Pass(i).Input.p(4) = 0.1+(0.2*i);
end

for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0   0.5 0.5 0;
         0.2 1   1   0.2;
         0.2 1   1   0.2;
         0   0.5 0.5 0].*(i/6);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
         Strat(p).Pass(i).Input.p(4) Strat(p).Pass(i).Input.p(4) Strat(p).Pass(i).Input.p(4) Strat(p).Pass(i).Input.p(4);
         Strat(p).Pass(i).Input.p(2) Strat(p).Pass(i).Input.p(2) Strat(p).Pass(i).Input.p(2) Strat(p).Pass(i).Input.p(2);
         0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0/3 Strat(p).Pass(i).Input.p(1) Strat(p).Pass(i).Input.p(3) 3/3;
         0/3 Strat(p).Pass(i).Input.p(1) Strat(p).Pass(i).Input.p(3) 3/3;
         0/3 Strat(p).Pass(1).Input.p(1) Strat(p).Pass(1).Input.p(3) 3/3;
         0/3 Strat(p).Pass(1).Input.p(1) Strat(p).Pass(1).Input.p(3) 3/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end

%% Example 14
p=p+1;
Strat(p).name = 'Triangle (Corner)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.05+(0.05*i);
    Strat(p).Pass(i).Input.p(2) = 0.5;
    Strat(p).Pass(i).Input.p(3) = 0.5-(0.05*i);
    Strat(p).Pass(i).Input.p(4) = 0.5+(0.1*i);
end

for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0   0.5 0.5 0;
         0.2 1   1   0.2;
         0.2 1   1   0.2;
         0   0.2 0.1 0].*(i/10);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
         Strat(p).Pass(i).Input.p(4) Strat(p).Pass(i).Input.p(4) Strat(p).Pass(i).Input.p(4) Strat(p).Pass(i).Input.p(4);
         Strat(p).Pass(i).Input.p(2) Strat(p).Pass(i).Input.p(2) Strat(p).Pass(i).Input.p(2) Strat(p).Pass(i).Input.p(2);
         0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0/3 Strat(p).Pass(i).Input.p(1) Strat(p).Pass(i).Input.p(3) 3/3;
         0/3 Strat(p).Pass(i).Input.p(1) Strat(p).Pass(i).Input.p(3) 3/3;
         0/3 Strat(p).Pass(1).Input.p(1) Strat(p).Pass(1).Input.p(3) 3/3;
         0/3 Strat(p).Pass(1).Input.p(1) Strat(p).Pass(1).Input.p(3) 3/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end


%% Example 15
p=p+1;
Strat(p).name = 'Triangle (side)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.05+(0.05*i);
    Strat(p).Pass(i).Input.p(2) = 0.1;
    Strat(p).Pass(i).Input.p(3) = 0.5-(0.05*i);
    Strat(p).Pass(i).Input.p(4) = 0.1+(0.2*i);
end

for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0   0.5 0.5 0;
         0.2 1   1   0.2;
         0.2 1   1   0.2;
         0   0.5 0.5 0].*(i/6);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
         Strat(p).Pass(i).Input.p(4) Strat(p).Pass(i).Input.p(4) Strat(p).Pass(i).Input.p(4) Strat(p).Pass(i).Input.p(4);
         Strat(p).Pass(i).Input.p(2) Strat(p).Pass(i).Input.p(2) Strat(p).Pass(i).Input.p(2) Strat(p).Pass(i).Input.p(2);
         0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0/3 Strat(p).Pass(i).Input.p(1) Strat(p).Pass(i).Input.p(3) 3/3;
         0/3 Strat(p).Pass(i).Input.p(1) Strat(p).Pass(i).Input.p(3) 3/3;
         0/3 Strat(p).Pass(1).Input.p(1) Strat(p).Pass(1).Input.p(3) 3/3;
         0/3 Strat(p).Pass(1).Input.p(1) Strat(p).Pass(1).Input.p(3) 3/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end


%% Example 16
p=p+1;
Strat(p).name = 'Triangle (middle)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.0+(0.1*i);
    Strat(p).Pass(i).Input.p(2) = 0.5;
    Strat(p).Pass(i).Input.p(3) = 1-(0.1*i);
    Strat(p).Pass(i).Input.p(4) = 0.5+(0.05*i);
end

for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = ...
        [0   0.5 0.5 0;
         0.2 1   1   0.2;
         0.2 1   1   0.2;
         0   0.2 0.1 0].*(i/10);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [3/3 3/3 3/3 3/3;
         Strat(p).Pass(i).Input.p(4) Strat(p).Pass(i).Input.p(4) Strat(p).Pass(i).Input.p(4) Strat(p).Pass(i).Input.p(4);
         Strat(p).Pass(i).Input.p(2) Strat(p).Pass(i).Input.p(2) Strat(p).Pass(i).Input.p(2) Strat(p).Pass(i).Input.p(2);
         0/3 0/3 0/3 0/3];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [0/3 Strat(p).Pass(i).Input.p(1) Strat(p).Pass(i).Input.p(3) 3/3;
         0/3 Strat(p).Pass(i).Input.p(1) Strat(p).Pass(i).Input.p(3) 3/3;
         0/3 Strat(p).Pass(1).Input.p(1) Strat(p).Pass(1).Input.p(3) 3/3;
         0/3 Strat(p).Pass(1).Input.p(1) Strat(p).Pass(1).Input.p(3) 3/3];
    
    Strat(p).Pass(i).Output.Bezctr.order = 4;
end
%% Example 17
p=p+1;
Strat(p).name = 'Vertical (All)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.4-(0.1*(i-1));
    Strat(p).Pass(i).Input.p(2) = 0.1;
    Strat(p).Pass(i).Input.p(3) = 0.6+(0.1*(i-1));
    Strat(p).Pass(i).Input.p(4) = 0.9;
end

% % Now define output as Bez patch
for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = [0 0.15 0.2 0.15 0;
        0.5 0.8 0.8 0.8 0.5;
        0.8 1 1 1 0.8;
        0.5 0.8 0.8 0.8 0.5;
        0 0.15 0.2 0.15 0;].*(i/6);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [3/3 3/3 3/3 3/3 3/3;
        0.6+(0.1*(i-1)) 0.6+(0.1*(i-1)) 0.6+(0.1*(i-1)) 0.6+(0.1*(i-1)) 0.6+(0.1*(i-1));
        0.5 0.5 0.5 0.5 0.5;
        0.4-(0.1*(i-1)) 0.4-(0.1*(i-1)) 0.4-(0.1*(i-1)) 0.4-(0.1*(i-1)) 0.4-(0.1*(i-1));
        0/3 0/3 0/3 0/3 0/3];
    
    
    Strat(p).Pass(i).Output.Bezctr.order = 5;
end

%% Example 18
p=p+1;
Strat(p).name = 'Vertical (Up)';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.3-(0.05*i);
    Strat(p).Pass(i).Input.p(2) = 0.1;
    Strat(p).Pass(i).Input.p(3) = 0.3+(0.05*i);
    Strat(p).Pass(i).Input.p(4) = 0.9;
end

% % Now define output as Bez patch
for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = [0 0.15 0.2 0.15 0;
        0.5 0.8 0.8 0.8 0.5;
        0.8 1 1 1 0.8;
        0.5 0.8 0.8 0.8 0.5;
        0 0.15 0.2 0.15 0;].*(i/4);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [3/3 3/3 3/3 3/3 3/3;
        0.4+(0.1*(i-1)) 0.4+(0.1*(i-1)) 0.4+(0.1*(i-1)) 0.4+(0.1*(i-1)) 0.4+(0.1*(i-1));
        0.35 0.35 0.35 0.35 0.35;
        0.3-(0.05*(i-1)) 0.3-(0.05*(i-1)) 0.3-(0.05*(i-1)) 0.3-(0.05*(i-1)) 0.3-(0.05*(i-1));        0/3 0/3 0/3 0/3 0/3];
    
    
    Strat(p).Pass(i).Output.Bezctr.order = 5;
    
end


%% Example 19
p=p+1;
Strat(p).name = 'XXXXXXXXXXX';


% Gen input passes
for i =1:4
    Strat(p).Pass(i).Input.p(1) = 0.35-(0.075*(i-1));
    Strat(p).Pass(i).Input.p(2) = 0.1;
    Strat(p).Pass(i).Input.p(3) = 0.55+(0.075*(i-1));
    Strat(p).Pass(i).Input.p(4) = 0.9;
end

% % Now define output as Bez patch
for i = 1:4
    Strat(p).Pass(i).Output.Bezctr.z = [0 0.15 0.2 0.15 0;
        0.5 0.8 0.8 0.8 0.5;
        0.8 1 1 1 0.8;
        0.5 0.8 0.8 0.8 0.5;
        0 0.15 0.2 0.15 0;].*(i/4);
    
    Strat(p).Pass(i).Output.Bezctr.y = ...
        [0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1;
        0 0.25 0.5 0.75 1];
    
    Strat(p).Pass(i).Output.Bezctr.x = ...
        [3/3 3/3 3/3 3/3 3/3;
        0.55+(0.075*(i-1)) 0.55+(0.075*(i-1)) 0.55+(0.075*(i-1)) 0.55+(0.075*(i-1)) 0.55+(0.075*(i-1));
        0.45 0.45 0.45 0.45 0.45;
        0.35-(0.075*(i-1)) 0.35-(0.075*(i-1)) 0.35-(0.075*(i-1)) 0.35-(0.075*(i-1)) 0.35-(0.075*(i-1));
        0/3 0/3 0/3 0/3 0/3];
    
    
    Strat(p).Pass(i).Output.Bezctr.order = 5;
end
















































% write output
patch = calcPatch(Strat(1).Pass(4).Output.Bezctr,npts);

% Grid data
l = sqrt(length(patch.z(:)));
z = reshape(patch.z(:),[l l]);
y = reshape(patch.y(:),[l l]);
x = reshape(patch.x(:),[l l]);
%             z = (z - min(z))./(max(z) - min(z));

[K,H,p1,p2] = surfature(x,y,z);
filename = append('gaussian_raw.jpg');
imwrite(K,filename);

mx = max(K,[],'all');
mn = -max(-K,[],'all');

K_new = (K - mn)./(mx - mn);
filename = append('gaussian_norm.jpg');
imwrite(K_new,filename);





%% Write to file & do plots
    
% write to file
if doWriteToFile
    writeToFile(Strat);
end


if doinptsplot
    inptsPlot(Strat);
end

%% -------------------------- %%
% Functions for plotting data

function [] = plotPasses(Strat,savePlot)
global npts

figure('units','normalized','outerposition',[0 0 1 1])
for i = 1:4
    Strat.Pass(i).Output.patch = calcPatch(Strat.Pass(i).Output.Bezctr,npts);
    for j = 1:i
        % Do input
        subplot(4,5,5*i-4)
        hold on
        % hack for colourmap
        if j == 1
            c = [0 0.4470 0.7410];
        elseif j == 2
            c = [0.8500 0.3250 0.0980];
        elseif j == 3
            c = [0.9290 0.6940 0.1250];
        elseif j == 4
            c = [0.4940 0.1840 0.5560];
        end
        %         passes = Strat.InputPasses(j).p;
        passes = Strat.Pass(j).Input.p;
        rectangle('Position',[passes(1) passes(2) passes(3)-passes(1) passes(4)-passes(2)],'EdgeColor',c)
        axis equal
        grid on
        ylabel('y')
        xlabel('x')
        xlim([0 1])
        ylim([0 1])
        title(['Pass ' num2str(i) ' input'])
    end
    % Do output
    % Hack:
    Bez = Strat.Pass(i).Output;
    subplot(4,5,5*i-1)
    C = Bez.patch.z;
    surf(Bez.patch.x,Bez.patch.y,Bez.patch.z,C)
    hold on
    points = [Bez.Bezctr.x(:),Bez.Bezctr.y(:),Bez.Bezctr.z(:)];
    plot3(points(:,1),points(:,2),points(:,3),'o','Color','b','MarkerSize',10,...
    'MarkerFaceColor','r');
    mesh(Bez.Bezctr.x,Bez.Bezctr.y,Bez.Bezctr.z,'EdgeColor','k','FaceColor','none')
    colorbar;
    caxis([0, 1]);
    xlim([0 1])
    ylim([0 1])
    zlim([0 1])
    zlabel('z')
    ylabel('y')
    xlabel('x')
    title(['Pass ' num2str(i) ' output (plan)'])
    view(2)
    axis equal
    %
    subplot(4,5,5*i-2)
    C = Bez.patch.z;
    surf(Bez.patch.x,Bez.patch.y,Bez.patch.z,C)
    colorbar;
    caxis([0, 1]);
    xlim([0 1])
    ylim([0 1])
    zlim([0 1])
    zlabel('z')
    ylabel('y')
    xlabel('x')
    title(['Pass ' num2str(i) ' output (side 2)'])
    view(90,0)
    axis equal
    %
    subplot(4,5,5*i-3)
    C = Bez.patch.z;
    surf(Bez.patch.x,Bez.patch.y,Bez.patch.z,C)
    colorbar;
    caxis([0, 1]);
    xlim([0 1])
    ylim([0 1])
    zlim([0 1])
    zlabel('z')
    ylabel('y')
    xlabel('x')
    title(['Pass ' num2str(i) ' output (side 2)'])
    view(0,0)
    axis equal
    %
    subplot(4,5,5*i)
    C = Bez.patch.z;
    surf(Bez.patch.x,Bez.patch.y,Bez.patch.z,C)
    colorbar;
    caxis([0, 1]);
    xlim([0 1])
    ylim([0 1])
    zlim([0 1])
    zlabel('z')
    ylabel('y')
    xlabel('x')
    title(['Pass ' num2str(i) ' output'])
end
sgtitle(Strat.name)


if savePlot
    saveas(gcf,[Strat.name '.png'])
end


end


function [] = inptsPlot(Strat)


figure('units','normalized','outerposition',[0 0 1 1])
for i = 1:length(Strat)
subplot(4,5,i)
for j = 1:4
    % Do input
    
    hold on
    % hack for colourmap
    if j == 1
        c = [0 0.4470 0.7410];
    elseif j == 2
        c = [0.8500 0.3250 0.0980];
    elseif j == 3
        c = [0.9290 0.6940 0.1250];
    elseif j == 4
        c = [0.4940 0.1840 0.5560];
    end
    %         passes = Strat.InputPasses(j).p;
    passes = Strat(i).Pass(j).Input.p;
    rectangle('Position',[passes(1) passes(2) passes(3)-passes(1) passes(4)-passes(2)],'EdgeColor',c)
    axis equal
    grid on
    ylabel('y')
    xlabel('x')
    xlim([0 1])
    ylim([0 1])
    str = ['ID: ', num2str(i)];
    title([string(Strat(i).name), str])
end

end
end

%% -------------------------- %%
% Functions for Bez Patch


function [patch] = calcPatch(Bez,npts)
% Create bez surface
% use conventions as defined on https://en.wikipedia.org/wiki/B%C3%A9zier_surface

% Define
uVals = linspace(0,1,npts);
vVals = linspace(0,1,npts);

% Initialize
patch.z = zeros(npts);
patch.x = zeros(npts);
patch.y = zeros(npts);

% Extract
n = Bez.order - 1;
m = Bez.order - 1;

for q = 1:npts
    u = uVals(q);
    for k = 1:npts
        v = vVals(k);
        for i = 0:n
            for j = 0:m
                patch.z(q,k) = patch.z(q,k) + (Bcalc(n,u,i) * Bcalc(m,v,j) * Bez.z(i+1,j+1));
                patch.x(q,k) = patch.x(q,k) + (Bcalc(n,u,i) * Bcalc(m,v,j) * Bez.x(i+1,j+1));
                patch.y(q,k) = patch.y(q,k) + (Bcalc(n,u,i) * Bcalc(m,v,j) * Bez.y(i+1,j+1));
            end
        end
    end
end
end



function v = Bcalc(n,u,i)
v = BernPol(n,i) * (u^i) * (power((1-u),(n-i)));
end


function v = BernPol(n,i)
v = factorial(n) / (factorial(i) * factorial(n-i));
end

function [] = plotBez(Bez,npts)

c = Bez.ctr
hold on

% Data is non-uniform, so use griddata function
z = reshape(Bez.patch.z,[],1);
x = reshape(Bez.patch.x,[],1);
y = reshape(Bez.patch.y,[],1);

[xq,yq] = meshgrid(linspace(0,1,npts),linspace(0,1,npts));
zq = griddata(x,y,z,xq,yq);

% plot
mesh(xq,yq,zq)

for i = 1:Bez.ctr.order
    x1 = c.x(i,:);
    y1 = c.y(i,:);
    z1 = c.z(i,:);
    
    x2 = c.x(:,i);
    y2 = c.y(:,i);
    z2 = c.z(:,i);
    
    plot3(x1,y1,z1,'k')
    plot3(x2,y2,z2,'k')
    
end

grid on
view(45,45)


end



function [] = writeToFile(Strat)
global npts
    strats = {'c2o','c2o','c2o','c2o','c2o','hor','hor','hor','over','over','over','over','tri','tri','tri','tri','ver','ver','ver'};
    path = 'C:\Users\adsk1\Documents\FYP\Python\data\geometry';
    for s = 1:length(Strat)
        
        if s > 1
            if strcmp(strats{s},strats{s-1})
                t = t + 1;
            else
                t = 1;
            end
        else
            t = 1;
        end
        name = append(strats{s},num2str(t));
        
        for p = 1:4
            
            % write output
            patch = calcPatch(Strat(s).Pass(p).Output.Bezctr,npts);
            
            % Grid data
            l = sqrt(length(patch.z(:)));
            z = reshape(patch.z(:),[l l]);
            y = reshape(patch.y(:),[l l]);
            x = reshape(patch.x(:),[l l]);
%             z = (z - min(z))./(max(z) - min(z));
            
            filename = append(path,'\raw\csv\',name,'_',num2str(p),'.csv');
            writematrix(z,filename,'Delimiter',',');
            filename = append(path,'\raw\images\',strats{s},'\',name,'_',num2str(p),'.jpg');
            imwrite(z,filename);
            
            [K,H,p1,p2] = surfature(x,y,z);
            filename = append(path,'\curvature\gaussian\csv\',name,'_',num2str(p),'.csv');
            writematrix(K,filename,'Delimiter',',');
            filename = append(path,'\curvature\gaussian\images\',strats{s},'\',name,'_',num2str(p),'.jpg');
            imwrite(K,filename);
            
            filename = append(path,'\curvature\mean\csv\',name,'_',num2str(p),'.csv');
            writematrix(H,filename,'Delimiter',',');
            filename = append(path,'\curvature\mean\images\',strats{s},'\',name,'_',num2str(p),'.jpg');
            imwrite(H,filename);
            
            filename = append(path,'\curvature\p1\csv\',name,'_',num2str(p),'.csv');
            writematrix(p1,filename,'Delimiter',',');
            filename = append(path,'\curvature\p1\images\',strats{s},'\',name,'_',num2str(p),'.jpg');
            imwrite(p1,filename);
            
            filename = append(path,'\curvature\p2\csv\',name,'_',num2str(p),'.csv');
            writematrix(p2,filename,'Delimiter',',');
            filename = append(path,'\curvature\p2\images\',strats{s},'\',name,'_',num2str(p),'.jpg');
            imwrite(p2,filename);
        end
        
        s
    end
end
