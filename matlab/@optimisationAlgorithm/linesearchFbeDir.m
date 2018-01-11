function [updateDualVar, fbeParameter ] = linesearchFbeDir(obj, primalVar, dualVar, fixedPointResidual, dirEnvelop)
%
% Function linesearchFbeDir find the step-size to update the dual variable 
%   in the given direction. This direction is calculated such that it
%   solves the fixed point residual. The default direction is proved from 
%   L-BFGS update for solving the fixed point residual. The step-size should 
%   ensure a decrease in the augumented lagrangian. 
%
% Syntax : 
%   [ updatePrimalVar, updateDualVar ] = linesearchFbeDir(obj, primalVar, dualVar, fixedResidual, dirEnvelop)
%
% Input : 
%   primalVar : primal variables which include for the function f and
%     function g
%   dualVar : dual variables correspionding to the constraints that connect
%     the variable of function f and function g  
%   fixedResidual : fixed residual that corresponds to the differnce of
%     funGvar - linearOperator*funFvar 
%   dirEnvelop : the direction to solve the fixed-point equation based on
%     the L-BFGS method
%
% Output : 
%   updatePrimalVar : the updated primal variable in the new direction 
%   updateDualVar : the updated dual variable in the new direction 
%

system = obj.system;
stageCost = system.stageCost;
terminalCost = system.terminalCost;

tree = system.tree;
prob = tree.prob;
numNode = length(tree.stage);
numScen = length(tree.leaves);
numNonLeaf = numNode - numScen; 

funFvar = primalVar.funFvar;
funGvar = primalVar.funGvar;

lambda = obj.algorithmParameter.lambda;
funFvarDirUpdate = obj.oracleDualGradientUpdate(dirEnvelop);
% independent Augmented Lagrangian 
independentAugLagran = 0;
for i = 1:numNonLeaf
    independentAugLagran = independentAugLagran + dualVar.y(:, i)'*fixedPointResidual.y(:, i);
    independentAugLagran = independentAugLagran - 0.5*lambda*norm(fixedPointResidual.y(:, i))^2;
end 
for i = 1:numScen
    independentAugLagran = independentAugLagran + dualVar.yt{i}'*fixedPointResidual.yt{i};
    independentAugLagran = independentAugLagran - 0.5*lambda*norm(fixedPointResidual.y(:, i))^2;
end
independentAugLagran = independentAugLagran - obj.calculatefunGvalue(funGvar);
% tau dependent Augmented Lagrangian 
dependentLinearAugLagran = 0;
dependentQuadraticAugLagran = 0;
for i = 1:numNonLeaf
    dependentLinearAugLagran = dependentLinearAugLagran + 2*prob(i)*funFvarDirUpdate.stateX(:, i)'*stageCost.matQ*...
        funFvar.stateX(:, i);
    dependentLinearAugLagran = dependentLinearAugLagran + 2*prob(i)*funFvarDirUpdate.inputU(:, i)'*stageCost.matR*...
        funFvar.inputU(:, i);
    dependentQuadraticAugLagran = dependentQuadraticAugLagran + prob(i)*funFvarDirUpdate.stateX(:, i)'*stageCost.matQ*...
        funFvarDirUpdate.stateX(:, i);
    dependentQuadraticAugLagran = dependentQuadraticAugLagran + prob(i)*funFvarDirUpdate.inputU(:, i)'*stageCost.matR*...
        funFvarDirUpdate.inputU(:, i);
end
for i = 1:numScen
    iLeave = numNonLeaf + i;
    dependentLinearAugLagran = dependentLinearAugLagran + 2*prob(iLeave)*funFvarDirUpdate.stateX(:, iLeave)'*terminalCost.matVf{i}*...
        funFvar.stateX(:, iLeave);
    dependentQuadraticAugLagran = dependentQuadraticAugLagran + prob(iLeave)*funFvarDirUpdate.stateX(:, iLeave)'*terminalCost.matVf{i}*...
        funFvarDirUpdate.stateX(:, iLeave);
end

tau = 0.95;
while(1)
    updateDualVar.y = dualVar.y + tau*dirEnvelop.y;
    for i = 1:numScen
        updateDualVar.yt{i} = dualVar.yt{i} + tau*dirEnvelop.yt{i};
    end
    
    updateFunFvar.stateX = funFvar.stateX + tau*funFvarDirUpdate.stateX;
    updateFunFvar.inputU = funFvar.inputU + tau*funFvarDirUpdate.inputU;
    
    [updateFunGvar, proximalParameter] = proximalG(obj, updateFunFvar, updateDualVar);
    % every iteration Augmented Lagrangian
    everyIterAugLagran = 0;
    for i = 1:numNonLeaf
        everyIterAugLagran =  everyIterAugLagran - updateDualVar.y(:, i)'*(proximalParameter.fixedPointResidual.y(:, i)) +...
        0.5*lambda*norm(proximalParameter.fixedPointResidual.y(:, i))^2;
    end
    for i = 1:numScen
        everyIterAugLagran = everyIterAugLagran - updateDualVar.yt{i}'*proximalParameter.fixedPointResidual.yt{i} + ...
            0.5*lambda*norm(proximalParameter.fixedPointResidual.yt{i})^2;
    end 
    everyIterAugLagran = everyIterAugLagran + obj.calculatefunGvalue(updateFunGvar);
    
    deltaAugLagran = independentAugLagran + tau*dependentLinearAugLagran + tau^2*dependentQuadraticAugLagran +...
        everyIterAugLagran;
    if(deltaAugLagran >= 0)
        break
    else 
        tau = 0.5*tau;
    end 
end 
fbeParameter.funFvar = updateFunFvar;
fbeParameter.funGvar = updateFunGvar;
fbeParameter.fixedPointResidual = proximalParameter.fixedPointResidual;
fbeParameter.deltaAugLagran = deltaAugLagran;
end

