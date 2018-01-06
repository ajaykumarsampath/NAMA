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
tree = obj.tree;
nx = size(system.dynamics.matA, 1);
nu = size(system.dynamics.matB, 2);
numScen = length(tree.leaves);
numNonLeaf = length(tree.children);

[funFvarUpdate, oracleGradParameter] = oracleDualGradientUpdate(obj, dir);

matHzUpdate.y = zeros(2*(nx + nu), numNonLeaf);
matHzUpdate.yt = cell(numScen, 1 );
for iPred = 1:numNonLeaf
    matHzUpdate.y(:,iPred) = constraint.matF{iPred}*funFvarUpdate.stateX(:,iPred) + constraint.matG{iPred}*funFvarUpdate.inputU(:,iPred);
end

for iPred=1:numScen
    matHzUpdate.yt{iPred,1} = terminalConstraint.matFt{iPred,1}*funFvarUpdate.stateX(:, numNonLeaf + iPred);
end 

oracleGradParameter.funFvarUpdate = funFvarUpdate;
end


