function [matHzUpdate, oracleGradParameter] = oracleHessianVec(obj, dir)
% 
% The function solveStepDirection calculates the primal-variable update on a
%   certain
% the tree. 
% 
% Syntax :
%   [matHzUpdate, oracleGradParameter] = oracleHessianVec(obj, dir)
% 
% Input : 
%   dir : the direction of the dual variable 
% 
% Output :
%   matHzUpdate : hessian-vector muliplication result
%   oracleGradParameter : parameters involved during the orcale of the dual 
%     gradient update. It contains the funFvarUpdate - the change in the function
%     f variables in the specified direction.  
%

system = obj.system;
constraint = system.constraint;
terminalConstraint = system.terminalConstraint;
tree = system.tree;
nx = size(system.dynamics.matA{1}, 1);
nu = size(system.dynamics.matB{1}, 2);
ny = size(constraint.matF{1}, 1);
numScen = length(tree.leaves);
numNonLeaf = length(tree.children);

[funFvarUpdate, oracleGradParameter] = oracleDualGradientUpdate(obj, dir);

matHzUpdate.y = zeros(ny, numNonLeaf);
matHzUpdate.yt = cell(numScen, 1 );
for i = 1:numNonLeaf
    matHzUpdate.y(:, i) = constraint.matF{i}*funFvarUpdate.stateX(:, i) + constraint.matG{i}*funFvarUpdate.inputU(:, i);
end

for i = 1:numScen
    matHzUpdate.yt{i} = terminalConstraint.matFt{i}*funFvarUpdate.stateX(:, numNonLeaf + i);
end 
oracleGradParameter.funFvarUpdate = funFvarUpdate;
end


