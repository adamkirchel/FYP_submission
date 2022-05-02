function[MS] = CalculateKinematics(MS,params)
% % Calculates kinematics for all DoF for the EW machine given 
%   user-defined inputs for the wheeling path.
%
% Inputs:
% MS - Manufacturing strategy (struct)
% params - Parameters (struct)
%
% Outputs:
% MS - Manufacturing strategy (struct)

% Wheel transformations relative to sheet
MS = CreateWheelActions(MS);

% End effector transformations relative to wheel
MS = EndEffectorKinematics(MS,params);

% Positions of machine axis
MS = DefineAxes(MS,params);

% G-code generation
MS = GenerateGCode(MS,params);

