function [funFvar, fbeParameter] = dualGlobalFbeAlgorithm(obj)
%
% This function dualGloblaFbeAlgorithm solve optimisation problem
%   associatated with the scenario-MPC problem. This solves FBE
%   for the dual formulation of the scenario-MPC optimisation problem
%
% Syntax :
%   [funFvar, fbeParameter] = dualGlobalFbeAlgorithm(obj)
%
% Input :
%  obj  : Optimisation object
%
% Output :
%   funFvar : decision variable of the smooth function
%   febParameter : structure with parameters about the fbe-algorithm
%     which include th dual cost, primal cost, dualgap, iteration etc.
%

system = obj.system;
constraint = system.constraint;
terminalConstraint = system.terminalConstraint;

tree = obj.system.tree;
numScen = length(tree.leaves);
numNode = length(tree.stage);

% initialise the dual variables
dualVar.y = zeros( size(constraint.matF{1}, 1), numNode - numScen);
numDualVarLeave = 0;
for i = 1:numScen
    dualVar.yt{i} = zeros( size(terminalConstraint.matFt{i}, 1), 1);
    numDualVarLeave = numDualVarLeave + size(terminalConstraint.matFt{i}, 1);
end
numDualVarNode =  size(constraint.matF{1}, 1) * (numNode - numScen);

iStep = 1;
tic
while(iStep < obj.algorithmParameter.stepEnvelop )
    % step 1: gradient of the congujate of the f (smooth function)
    [funFvar, solveStepDetails] = obj.solveStep(dualVar);
    if(iStep > 1)
        % step 2: proximal with respect to g or argmin of the argumented
        %   Lagrangian with respect to dual varaible
        [~, proximalParameter] = obj.dualVariableUpdate(funFvar, dualVar);
        % step 3: gradient of the dual FBE
        [gradientEnv, ~] = obj.gradientDualEnvelop(proximalParameter.fixedPointResidual);
        % step 4: find the direction - calculated through L-BFGS method
        [dirEnvelop, directionParamter] = obj.directionLbfgs(gradientEnv, oldGradientEnv, dualVar, oldDualVar);
        % step 5: lineSearch in the direction of the dual FBE
        primalVar.funFvar = funFvar;
        primalVar.funGvar = proximalParameter.funGvar;
        fixedPointResidual = proximalParameter.fixedPointResidual;
        obj.algorithmParameter.lambda = proximalParameter.lambda;
        [fbeUpdateDualVar, fbeLsParameter] = linesearchFbeDir(obj, primalVar, dualVar, fixedPointResidual, dirEnvelop);
        
        oldDualVar = dualVar;
        oldGradientEnv = gradientEnv;
        % step 5 adapted dual variable
        dualVar.y = fbeUpdateDualVar.y - obj.algorithmParameter.lambda*fbeLsParameter.fixedPointResidual.y;
        for i = 1:numScen
            dualVar.yt{i} = fbeUpdateDualVar.yt{i} - obj.algorithmParameter.lambda*fbeLsParameter.fixedPointResidual.yt{i};
        end
        
        [valueFbeUpdatedDualVar, valueParameter] = valueAugmentedLagrangian(obj, fbeLsParameter.funFvar,...
            fbeLsParameter.funGvar, fbeUpdateDualVar, fbeLsParameter.fixedPointResidual);
        fixedPointResidualVec = zeros(numDualVarNode + numDualVarLeave, 1);
        fixedPointResidualVec(1:numDualVarNode, 1) = reshape(fbeLsParameter.fixedPointResidual.y, numDualVarNode, 1);
        fixedPointResidualVec(numDualVarNode + 1: numDualVarNode + numDualVarLeave, 1) = reshape(cell2mat(...
            fbeLsParameter.fixedPointResidual.yt), numDualVarLeave, 1);
    else
        % dual gradient update
        oldDualVar = dualVar;
        [dualVar, proximalParameter] = obj.dualVariableUpdate(funFvar, oldDualVar);
        obj.algorithmParameter.lambda = proximalParameter.lambda;
        % gradient of the dual FBE
        [gradientEnv, ~] = gradientDualEnvelop( obj, proximalParameter.fixedPointResidual);
        oldGradientEnv = gradientEnv;
        
        [valueFbeUpdatedDualVar, valueParameter] = valueAugmentedLagrangian(obj, funFvar,...
            proximalParameter.funGvar, oldDualVar, proximalParameter.fixedPointResidual);
        fixedPointResidualVec = zeros(numDualVarNode + numDualVarLeave, 1);
        fixedPointResidualVec(1:numDualVarNode, 1) = reshape(proximalParameter.fixedPointResidual.y, numDualVarNode, 1);
        fixedPointResidualVec(numDualVarNode + 1: numDualVarNode + numDualVarLeave, 1) = reshape(cell2mat(...
            proximalParameter.fixedPointResidual.yt), numDualVarLeave, 1);
    end
    % termination condition
    fbeParameter.lambda(iStep) = obj.algorithmParameter.lambda;
    fbeParameter.primalCost(iStep) = valueParameter.primalValue;% primal cost;
    fbeParameter.dualCost(iStep) = valueParameter.primalValue + valueParameter.dualGapValue;% dual cost;
    fbeParameter.dualGap(iStep) = -valueParameter.dualGapValue;
    fbeParameter.valueArgLagran(iStep) = valueFbeUpdatedDualVar;
    fbeParameter.normFixedPointResidual(iStep) = norm(fixedPointResidualVec);
    if( iStep > 1)
        fbeParameter.descentValue(iStep - 1) = directionParamter.descentValue;
        fbeParameter.vecYSk(iStep - 1) = directionParamter.vecYSk;
    end
    if(norm(fixedPointResidualVec) < obj.algorithmParameter.normFixedPointResidual)
        fbeParameter.iterate = iStep;
        break
    else
        iStep = iStep + 1;
    end
end
fbeParameter.timeSolve = toc;
fbeParameter.solveInvokCount = solveStepDetails.invokCount;



%{
defaultProxLineSearch = obj.algorithmParameter.proxLineSearch;
initialLambda = obj.algorithmParameter.lambda;
iStep = 1;
tic
% step 1: gradient of the congujate of the f (smooth function)
[funFvar, solveStepDetails] = obj.solveStep(dualVar);
while(iStep < obj.algorithmParameter.stepEnvelop )
    if(iStep > 1)
        % step 1: update the gradient of the conjuage using previous
        %   iterate caluation
        funFvar = updatedFunFvar;
        while(1)
            % step 2: proximal with respect to g or argmin of the argumented
            %   Lagrangian with respect to dual varaible
            lambda = obj.algorithmParameter.lambda;
            obj.algorithmParameter.proxLineSearch = 'no';
            [~, proximalParameter] = obj.dualVariableUpdate(funFvar, dualVar);
            % step 3: gradient of the dual FBE
            [gradientEnv, ~] = obj.gradientDualEnvelop(proximalParameter.fixedPointResidual);
            % step 4: find the direction - calculated through L-BFGS method
            [dirEnvelop, directionParamter] = obj.directionLbfgs(gradientEnv, oldGradientEnv, dualVar, oldDualVar);
            % step 5: lineSearch in the direction of the dual FBE
            primalVar.funFvar = funFvar;
            primalVar.funGvar = proximalParameter.funGvar;
            fixedPointResidual = proximalParameter.fixedPointResidual;
            obj.algorithmParameter.lambda = proximalParameter.lambda;
            [fbeUpdateDualVar, fbeLsParameter] = linesearchFbeDir(obj, primalVar, dualVar, fixedPointResidual, dirEnvelop);
            [valueFbeUpdatedDualVar, valueParameter] = valueAugmentedLagrangian(obj, fbeLsParameter.funFvar,...
                fbeLsParameter.funGvar, fbeUpdateDualVar, fbeLsParameter.fixedPointResidual);
            % step 5 adapted dual variable
            nextIterateDualVar.y = fbeUpdateDualVar.y - lambda*fbeLsParameter.fixedPointResidual.y;
            for i = 1:numScen
                nextIterateDualVar.yt{i} = fbeUpdateDualVar.yt{i} - lambda*fbeLsParameter.fixedPointResidual.yt{i};
            end
            updatedFunFvar = obj.solveStep(nextIterateDualVar);
            [~, updatedProximalParameter] = obj.dualVariableUpdate(updatedFunFvar, nextIterateDualVar);
            valueAugmentedLagranUpdatedDualVar = valueAugmentedLagrangian(obj, updatedFunFvar,...
                updatedProximalParameter.funGvar, nextIterateDualVar, updatedProximalParameter.fixedPointResidual);
            if(valueAugmentedLagranUpdatedDualVar - valueFbeUpdatedDualVar > 0)
                break;
            else
                obj.algorithmParameter.lambda = 0.5*lambda;
            end
        end
        oldDualVar = dualVar;
        oldGradientEnv = gradientEnv;
        dualVar = nextIterateDualVar;
        
        fixedPointResidualVec = zeros(numDualVarNode + numDualVarLeave, 1);
        fixedPointResidualVec(1:numDualVarNode, 1) = reshape(fbeLsParameter.fixedPointResidual.y, numDualVarNode, 1);
        fixedPointResidualVec(numDualVarNode + 1: numDualVarNode + numDualVarLeave, 1) = reshape(cell2mat(...
            fbeLsParameter.fixedPointResidual.yt), numDualVarLeave, 1);
    else
        % dual gradient update
        oldDualVar = dualVar;
        [dualVar, proximalParameter] = obj.dualVariableUpdate(funFvar, oldDualVar);
        obj.algorithmParameter.lambda = proximalParameter.lambda;
        % gradient of the dual FBE
        [gradientEnv, ~] = gradientDualEnvelop( obj, proximalParameter.fixedPointResidual);
        oldGradientEnv = gradientEnv;
        % gradient of the conjugate function for the next step
        updatedFunFvar = obj.solveStep(dualVar);
        
        [valueFbeUpdatedDualVar, valueParameter] = valueAugmentedLagrangian(obj, funFvar,...
            proximalParameter.funGvar, oldDualVar, proximalParameter.fixedPointResidual);
        fixedPointResidualVec = zeros(numDualVarNode + numDualVarLeave, 1);
        fixedPointResidualVec(1:numDualVarNode, 1) = reshape(proximalParameter.fixedPointResidual.y, numDualVarNode, 1);
        fixedPointResidualVec(numDualVarNode + 1: numDualVarNode + numDualVarLeave, 1) = reshape(cell2mat(...
            proximalParameter.fixedPointResidual.yt), numDualVarLeave, 1);
    end
    % termination condition
    obj.algorithmParameter.lambda = initialLambda;
    fbeParameter.lambda(iStep) = obj.algorithmParameter.lambda;
    fbeParameter.primalCost(iStep) = valueParameter.primalValue;% primal cost;
    fbeParameter.dualCost(iStep) = valueParameter.primalValue + valueParameter.dualGapValue;% dual cost;
    fbeParameter.dualGap(iStep) = -valueParameter.dualGapValue;
    fbeParameter.valueArgLagran(iStep) = valueFbeUpdatedDualVar;
    fbeParameter.normFixedPointResidual(iStep) = norm(fixedPointResidualVec);
    if( iStep > 1)
        fbeParameter.descentValue(iStep - 1) = directionParamter.descentValue;
        fbeParameter.vecYSk(iStep - 1) = directionParamter.vecYSk;
    end
    if(norm(fixedPointResidualVec) < obj.algorithmParameter.normFixedPointResidual)
        fbeParameter.iterate = iStep;
        break
    else
        iStep = iStep + 1;
    end
end
fbeParameter.timeSolve = toc;
fbeParameter.solveInvokCount = solveStepDetails.invokCount;
%}

end

