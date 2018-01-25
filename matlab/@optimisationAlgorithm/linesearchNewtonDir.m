function [updateDualVar, namaParameter ] = linesearchNewtonDir(obj, primalVar, dualVar, fixedPointResidual, dirEnvelop)
%
% Function linesearchNewtonDir find the step-size to update the dual variable 
%   in the given direction. This direction is calculated such that it
%   solves the fixed point residual. The default direction is proved from 
%   L-BFGS update for solving the fixed point residual. The step-size should 
%   ensure a decrease in the augumented lagrangian. 
%
% Syntax : 
%   [ updatePrimalVar, updateDualVar ] = linesearchNewtonDir(obj, primalVar, dualVar, fixedResidual, dirEnvelop)
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
firstDir.y = -fixedPointResidual.y;
secondDir.y = dirEnvelop.y + lambda*fixedPointResidual.y;
for i = 1:numScen
    firstDir.yt{i} = -fixedPointResidual.yt{i};
    secondDir.yt{i} = dirEnvelop.yt{i} + lambda*fixedPointResidual.yt{i};
end
funFvarUpdateFirstDir = obj.oracleDualGradientUpdate(firstDir);
funFvarUpdateSecondDir = obj.oracleDualGradientUpdate(secondDir);

independentAugLagran = 0; 
for i = 1:numNonLeaf
    independentAugLagran = independentAugLagran + prob(i)*(2*lambda*funFvar.stateX(:, i) + ...
        lambda^2*funFvarUpdateFirstDir.stateX(:, i))' * stageCost.matQ * funFvarUpdateFirstDir.stateX(:, i);
    independentAugLagran = independentAugLagran + prob(i)*(2*lambda*funFvar.inputU(:,i) + ...
        lambda^2*funFvarUpdateFirstDir.inputU(:, i))' * stageCost.matR * funFvarUpdateFirstDir.inputU(:, i);
    independentAugLagran = independentAugLagran - 0.5*lambda*norm(fixedPointResidual.y(:, i))^2;
    independentAugLagran = independentAugLagran + dualVar.y(:, i)'*fixedPointResidual.y(:, i);
end
for i = 1:numScen
    iLeave = numNonLeaf + i;
    independentAugLagran = independentAugLagran + prob(iLeave)*(2*lambda*funFvar.stateX(:, iLeave) + ...
        lambda^2*funFvarUpdateFirstDir.stateX(:, iLeave))' * terminalCost.matVf{i} * funFvarUpdateFirstDir.stateX(:, iLeave);
    independentAugLagran = independentAugLagran - 0.5*lambda*norm(fixedPointResidual.yt{i})^2;
    independentAugLagran = independentAugLagran + dualVar.yt{i}'*fixedPointResidual.yt{i};
end
independentAugLagran = independentAugLagran - obj.calculatefunGvalue(funGvar);

dependentLinearAugLagran = 0;
dependentQuadraticAugLagran = 0;
for i = 1:numNonLeaf
    dependentLinearAugLagran = dependentLinearAugLagran + 2*prob(i)*(funFvar.stateX(:, i) + ...
        lambda*funFvarUpdateFirstDir.stateX(:, i))'*stageCost.matQ * funFvarUpdateSecondDir.stateX(:, i);
    dependentLinearAugLagran = dependentLinearAugLagran + 2*prob(i)*(funFvar.inputU(:, i) +...
        lambda*funFvarUpdateFirstDir.inputU(:, i))' *stageCost.matR * funFvarUpdateSecondDir.inputU(:, i);
    dependentQuadraticAugLagran = dependentQuadraticAugLagran + prob(i)*funFvarUpdateSecondDir.stateX(:, i)' *...
        stageCost.matQ * funFvarUpdateSecondDir.stateX(:, i);
    dependentQuadraticAugLagran = dependentQuadraticAugLagran + prob(i)*funFvarUpdateSecondDir.inputU(:, i)' *...
        stageCost.matR * funFvarUpdateSecondDir.inputU(:,i);
end
for i = 1:numScen
    iLeave = numNonLeaf + i;
    dependentLinearAugLagran = dependentLinearAugLagran + 2*prob(iLeave)*(funFvar.stateX(:, iLeave) + ...
        lambda*funFvarUpdateFirstDir.stateX(:, iLeave))'*terminalCost.matVf{i} * funFvarUpdateSecondDir.stateX(:, iLeave);
    dependentQuadraticAugLagran = dependentQuadraticAugLagran + prob(iLeave)*funFvarUpdateSecondDir.stateX(:, iLeave)' *...
        terminalCost.matVf{i} * funFvarUpdateSecondDir.stateX(:, iLeave);
end

tau = 0.95;
while(1)
    updateDualVar.y = dualVar.y + lambda*firstDir.y + tau*secondDir.y;
    for i = 1:numScen
        updateDualVar.yt{i} = dualVar.yt{i} + lambda*firstDir.yt{i} + tau*secondDir.yt{i};
    end
    
    updateFunFvar.stateX = funFvar.stateX + lambda*funFvarUpdateFirstDir.stateX + tau*funFvarUpdateSecondDir.stateX;
    updateFunFvar.inputU = funFvar.inputU + lambda*funFvarUpdateFirstDir.inputU + tau*funFvarUpdateSecondDir.inputU;
    
    [updateFunGvar, proximalParameter] = proximalG(obj, updateFunFvar, updateDualVar);
    
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

%{
valueAugLag = obj.valueAugmentedLagrangian(funFvar, funGvar , dualVar, fixedPointResidual);
valueUpdatedAugLag = obj.valueAugmentedLagrangian(updateFunFvar, updateFunGvar, updateDualVar,...
    proximalParameter.fixedPointResidual);
deltaAugLag = valueUpdatedAugLag - valueAugLag;
%}
namaParameter.funFvar = updateFunFvar;
namaParameter.funGvar = updateFunGvar;
namaParameter.fixedPointResidual = proximalParameter.fixedPointResidual;
namaParameter.deltaAugLagran = deltaAugLagran;
namaParameter.tau = tau;
end

