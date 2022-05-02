function[T_inv] = CalculateTinv(T_matrix)
% % Calculates the inverse of the homogeneous transfromation matrices
% 
% Inputs:
% T_matrix - Homogeneous transformation matrix
% 
% Outputs:
% T_inv - Inverse transformation matrix

R_inv = T_matrix(1:3,1:3)';
O_inv = -R_inv * T_matrix(1:3,4);
T_inv = [R_inv O_inv ; [0 0 0 1]];