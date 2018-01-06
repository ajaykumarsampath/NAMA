function [funFvar, apgParameter] = dualApgAlgorithm(obj)
% 
% The function dualApgAlgorithm solve the scenario-based mpc using
%   accelerated proximal-gradient method on its dual formulation. 
%
% Syntax :
%  [primalVariable, apgParameter] = dualApgAlgorithm(obj)
% 
% Input :
%
% Output:
%   primalVariable : primal variable 
%   apgParameter : structure containing details like time of
%     computation, number of itrataions, primalCost, dualCost, dualGap, 
%     dualVariable and acceleratedDualVariable
%
%

system = obj.system; 
stageCost = system.stageCost;
terminalCost = system.terminalCost;
constraint = system.constraint;
terminalConstraint = system.terminalConstraint;

tree = system.tree;
numScen = length(tree.leaves);
numNode = length(tree.stage);

% initialise the dual variables 
prevDualVar.y = zeros( size(constraint.matF{1}, 1), numNode - numScen); 
currentDualVar.y = zeros( size(constraint.matF{1}, 1), numNode - numScen);
accelerateDualVar.y = zeros( size(constraint.matF{1}, 1), numNode - numScen);
numDualVarNonLeaf = size(constraint.matF{1}, 1)*(numNode - numScen);
numDualVarLeave = 0;
for i = 1:numScen
    prevDualVar.yt{i} = zeros( size(terminalConstraint.matFt{i}, 1), 1);
    currentDualVar.yt{i} = zeros( size(terminalConstraint.matFt{i}, 1), 1);
    accelerateDualVar.yt{i} = zeros( size(terminalConstraint.matFt{i}, 1), 1);
    numDualVarLeave = numDualVarLeave + size(terminalConstraint.matFt{i}, 1);
end 

theta = [1, 1];
iStep = 1;
tic
while(iStep < obj.algorithmParameter.stepApg )
    % step 1: extrapolation step for the dual variable
    accelerateDualVar.y = currentDualVar.y + theta(2)*(1/theta(1) - 1)*(currentDualVar.y -...
        prevDualVar.y);
    for i = 1:numScen
        accelerateDualVar.yt{i} = currentDualVar.yt{i} + theta(2)*(1/theta(1)-1)*(currentDualVar.yt{i} -...
            prevDualVar.yt{i});
    end
    % step 2: dual gradient calculation
    [funFvar, solveStepDetails] = obj.solveStep(accelerateDualVar);
    % step 3: proximal step on g conjugate.
    prevDualVar.y = currentDualVar.y;
    prevDualVar.yt = currentDualVar.yt;
    [currentDualVar, proximalParameter] = obj.dualVariableUpdate(funFvar, accelerateDualVar);
    
    apgParameter.lambda(iStep) = proximalParameter.lambda;
    apgParameter.primalCost(iStep) = 0;% primal cost;
    apgParameter.dualCost(iStep) = 0;% dual cost;
    fixedPointResidual = proximalParameter.fixedPointResidual;
    fixedPointResidualVec = zeros(numDualVarNonLeaf + numDualVarLeave, 1);
    fixedPointResidualVec(1:numDualVarNonLeaf, 1) = reshape(fixedPointResidual.y, numDualVarNonLeaf, 1);
    fixedPointResidualVec(numDualVarNonLeaf + 1: numDualVarNonLeaf + numDualVarLeave, 1) = reshape(cell2mat(...
        fixedPointResidual.yt), numDualVarLeave, 1);
    for i = 1:numNode - numScen
        apgParameter.primalCost(iStep) = apgParameter.primalCost(iStep) + tree.prob(i)*(funFvar.stateX(:,i)' *...
            stageCost.matQ*funFvar.stateX(:, i) + funFvar.inputU(:,i)'*stageCost.matR*funFvar.inputU(:,i));
    end
    dualVariableVec = [reshape(currentDualVar.y, numDualVarNonLeaf, 1); reshape(cell2mat(currentDualVar.yt), numDualVarLeave, 1)];
    for i = 1:numScen
        apgParameter.primalCost(iStep) = apgParameter.primalCost(iStep)+tree.prob(tree.leaves(i))*(funFvar.stateX(:,tree.leaves(i))'*...
            terminalCost.matVf{i} *funFvar.stateX(:,tree.leaves(i)));
    end
    apgParameter.dualCost(iStep) = apgParameter.primalCost(iStep) + dualVariableVec'*fixedPointResidualVec;
    apgParameter.dualGap(iStep) = dualVariableVec'*fixedPointResidualVec;
    apgParameter.normFixedPointResidual(iStep) = norm(fixedPointResidualVec);
    
    if(norm(fixedPointResidualVec) < obj.algorithmParameter.normFixedPointResidual)
        apgParameter.iterate = iStep;
        break
    else 
        theta(1) = theta(2);
        theta(2) = (sqrt(theta(1)^4 + 4*theta(1)^2) - theta(1)^2)/2;
        iStep = iStep + 1;
    end
    %{
    [currentDualVar, proximalParameter] = obj.proximalGconjugate(funFvar, acceleratedDualVariable);
    apgParameter.lambda(iStep) = proximalParameter.lambda;
    obj.algorithmParameter.lambda = proximalParameter.lambda;
    
    primalConstraint = proximalParameter.primalConstraint;
    primalTerminalConstraint = proximalParameter.primalTerminalConstraint;
    apgParameter.primalConstraint{iStep} = primalConstraint;
    
    apgParameter.primalCost(iStep) = 0;% primal cost;
    apgParameter.dualCost(iStep) = 0;% dual cost;
    epsilonPrimalConstraint = max( max( max(primalConstraint, 0) ) );
    epsilonPrimalConstraint = max(max(cell2mat(primalTerminalConstraint), epsilonPrimalConstraint));
    
    % primal infeasibility 
    primalInfs.y = currentDualVar.y - prevDualVar.y;
    for i = 1:numScen
        primalInfs.yt{i} = currentDualVar.yt{i} - prevDualVar.yt{i};
    end
    Glambda = [vec(primalInfs.y);vec(cell2mat(primalInfs.yt))];
    apgParameter.Glambda(iStep) = norm(Glambda)/proximalParameter.lambda;
    
    if(norm(primalInfs.y) > proximalParameter.lambda*algorithmParameter.primalInfeasibility)
        % step 4: theta update
        theta(1) = theta(2);
        theta(2) = (sqrt(theta(1)^4 + 4*theta(1)^2) - theta(1)^2)/2;
        iStep = iStep + 1;
    else
        apgParameter.iterate = iStep;
        break
    end
    
    apgParameter.epsilonPrimalConstraint(iStep) = epsilonPrimalConstraint;
    for i = 1:numNode - numScen
        apgParameter.primalCost(iStep) = apgParameter.primalCost(iStep) + tree.prob(i,1)*(funFvar.stateX(:,i)' *...
            stageCost.matQ*funFvar.stateX(:,i) + funFvar.inputU(:,i)'*stageCost.matR*funFvar.inputU(:,i));
        apgParameter.dualCost(iStep) = apgParameter.dualCost(iStep) + currentDualVar.y(:,i)'*(primalConstraint(:,i));
    end
    for i = 1:numScen
        apgParameter.primalCost(iStep) = apgParameter.prm_cst(iStep)+tree.prob(tree.leaves(i))*(funFvar.stateX(:,tree.leaves(i))'*...
            terminalCost.matVf{i,1} *funFvar.stateX(:,tree.leaves(i)));
        apgParameter.dualCost(iStep) = apgParameter.dualCost(iStep) + currentDualVar.yt{i,1}'*(primalTerminalConstraint{i,1});
    end
    apgParameter.dualCost(iStep) = apgParameter.primalCost(iStep) + apgParameter.dualCost(iStep);   
    %}
end
apgParameter.timeSolve = toc;
apgParameter.acceleratedDualVariable = accelerateDualVar;
apgParameter.dualVariable = currentDualVar;
apgParameter.gradInvokCount = solveStepDetails.invokCount;
end