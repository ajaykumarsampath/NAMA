function [value, valueParameter] = valueAugmentedLagrangian(obj, funFvar, funGvar, dualVar, fixedPointResidual)
%
% function vlaueAugmentedLagrangian calculate the value of the augmented
%   Lagrangian at the given (x, z, y)
%
% Syntax : 
%   value = valueAugmentedLagrangian(obj, funFvar, funGvar, dualVar, fixedPointResidual)
%
% Input : 
%   funFvar : x 
%   funGvar : z
%   dualVar : y 
%
% Output : 
%  value : the value of the augumented lagrangian 

system = obj.system;
stageCost = system.stageCost;
terminalCost = system.terminalCost;
lambda = obj.algorithmParameter.lambda;

tree = system.tree;
numNode = length(tree.stage);
numScen = length(tree.leaves);
numNonLeaf = numNode - numScen;
prob = tree.prob;
value = 0;
primalValue = 0;
dualGapValue = 0;
for iNode = 1:numNonLeaf
    primalValue = primalValue + prob(iNode)*(funFvar.stateX(:, iNode)'*stageCost.matQ*...
        funFvar.stateX(:, iNode));
    primalValue = primalValue + prob(iNode)*(funFvar.inputU(:, iNode)'*stageCost.matR*...
        funFvar.inputU(:, iNode));
    dualGapValue = dualGapValue - dualVar.y(:, iNode)'*fixedPointResidual.y(:, iNode);
    value = value + 0.5*lambda*norm(fixedPointResidual.y(:, iNode))^2;
end 
for iScen = 1:numScen
    iLeave = numNonLeaf + iScen;
    primalValue = primalValue + prob(iLeave)*(funFvar.stateX(:, iLeave)'*terminalCost.matVf{iScen}*...
        funFvar.stateX(:, iLeave));
    dualGapValue = dualGapValue - dualVar.yt{iScen}'*fixedPointResidual.yt{iScen};
    value = value + 0.5*lambda*norm(fixedPointResidual.yt{iScen})^2;
end 

primalValue = primalValue + obj.calculatefunGvalue(funGvar);
valueParameter.normFixedResidual = value;
value = value + primalValue + dualGapValue;
valueParameter.primalValue = primalValue;
valueParameter.dualGapValue = dualGapValue;
end 