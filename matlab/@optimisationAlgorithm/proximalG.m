function [funGvar, proximalParameter] = proximalG(obj, funFvar, dualVar)
%
%  The function proximalG calculate the proximal with respect to the non-smooth function. 
%    This proximal calculation point is equal to the minimising the argumented
%    lagrangian. Now the default non-smooth function is the indicator set of
%    the constraints 
%
%  Syntax :
%    [funGvar, proximalParameter] = proximalG(obj, funFvar, dualVariable)
%
%  Input :
%    dualGradient : dual gradient/gradient of the conjugate function. 
%      In the current context this is the state and input over the scenario tree  
%    dualVariable :  current dual variables
%
%  Output :            
%    matT : argmin of the argumented Lagrangian. This is also the proximal
%      step with respect to the function g
%    proximalDetails : structure that containt the Hx and Hx-z (primal 
%      infeasibiliy vector) 
%

system = obj.system;
tree = system.tree;
constraint = system.constraint;
terminalConstraint = system.terminalConstraint;
numNonLeaf = length(tree.children);
numScen = length(tree.leaves);

lambda = obj.algorithmParameter.lambda;
for i = 1:numNonLeaf
    proximalParameter.matHz.y(:, i) = constraint.matF{i}*funFvar.stateX(:,i) + constraint.matG{i}*funFvar.inputU(:,i);
    funGvar.y(:, i) = 1/lambda  * dualVar.y(:, i) + proximalParameter.matHz.y(:, i);
end
for i = 1:numScen
    proximalParameter.matHz.yt{i,1} = terminalConstraint.matFt{i}*funFvar.stateX(:, tree.leaves(i));
    funGvar.yt{i} = 1/lambda * dualVar.yt{i} + proximalParameter.matHz.yt{i};
end
% proximal with indicator function that represent the hard constraints
for i = numNonLeaf
    funGvar.y(:, i) = min(funGvar.y(:, i), constraint.g{i});
    fixedPointResidual.y(:, i) = funGvar.y(:, i) - proximalParameter.matHz.y(:, i);
end
for i = 1:numScen
    funGvar.yt{i} = min(funGvar.yt{i}, terminalConstraint.gt{i});
    fixedPointResidual.yt{i} = funGvar.yt{i} - proximalParameter.matHz.yt{i};
end
proximalParameter.fixedPointResidual = fixedPointResidual;
end

