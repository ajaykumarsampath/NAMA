function [funFvar, namaParameter] = newtonAmeAlgorithm(obj)
%
% Function newtonAmeAlgorithm solves the scenario-based mpc using
%   netwon-based alternating minimisation envelope algorithm. This algorithm
%   belong to the class of quasi-newton method for constrainted optimisation
%   problems. This algorithm combines the limited-memory BFGS along 
%   with the dual gradient such that the augumented lagrangian ( dual 
%   forward-backward envelop) is decreasing at each iterate.
%
% Syntax :
%  [funFvar, namaParameter] = newtonAmeAlgorithm(obj)
%
% Input :
%   obj : optimisationAlgorithm object
% 
% Output :
%  primalVariable : state and the input over the scenario-tree 
%  namaParameter : structure for the NAMA algorithm which contain time of 
%    computation, iterations, dualVaraible, residual.
%

system = obj.system; 
stageCost = system.stageCost;
terminalCost = system.terminalCost;
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
        [~, proximalParameter] = dualVariableUpdate(obj, funFvar, dualVar);
        %[funGvar, proximalParameter] = proximalG(obj, funFvar, dualVar);
        % step 3: find the direction - calculated through L-BFGS method  
        funGvar = proximalParameter.funGvar;
        fixedPointResidual = proximalParameter.fixedPointResidual;
        obj.algorithmParameter.lambda = proximalParameter.lambda;
        [ obj, dirEnvelop ] = obj.directionLbfgs(fixedPointResidual, oldFixedPointResidual,...
            dualVar, oldDualVar);
        oldDualVar = dualVar;
        oldFixedPointResidual = fixedPointResidual;
        % step 4 line search on the Lagrangian
        primalVar.funFvar = funFvar;
        primalVar.funGvar = funGvar;
        [newtonUpdateDualVar, newtonLsParameter] = linesearchNewtonDir(obj, primalVar, dualVar, fixedPointResidual, dirEnvelop);
        % step 5 update the dual variable
        dualVar.y = newtonUpdateDualVar.y + newtonLsParameter.fixedPointResidual.y;
        for i = 1:numScen
            dualVar.yt{i} = newtonUpdateDualVar.yt{i} + newtonLsParameter.fixedPointResidual.yt{i};
        end
        fixedPointResidualVec = zeros(numDualVarNode + numDualVarLeave, 1);
        fixedPointResidualVec(1:numDualVarNode, 1) = reshape(newtonLsParameter.fixedPointResidual.y, numDualVarNode, 1);
        fixedPointResidualVec(numDualVarNode + 1: numDualVarNode + numDualVarLeave, 1) = reshape(cell2mat(...
            newtonLsParameter.fixedPointResidual.yt), numDualVarLeave, 1);
    else
        % dual gradient update
        oldDualVar = dualVar; 
        [dualVar, proximalParameter] = obj.dualVariableUpdate(funFvar, oldDualVar);
        oldFixedPointResidual = proximalParameter.fixedPointResidual;
        fixedPointResidualVec = zeros(numDualVarNode + numDualVarLeave, 1);
        fixedPointResidualVec(1:numDualVarNode, 1) = reshape(proximalParameter.fixedPointResidual.y, numDualVarNode, 1);
        fixedPointResidualVec(numDualVarNode + 1: numDualVarNode + numDualVarLeave, 1) = reshape(cell2mat(...
            proximalParameter.fixedPointResidual.yt), numDualVarLeave, 1);
    end
    % termination condition 
    namaParameter.lambda(iStep) = proximalParameter.lambda;
    namaParameter.primalCost(iStep) = 0;% primal cost;
    namaParameter.dualCost(iStep) = 0;% dual cost;
    
    for i = 1:numNode - numScen
        namaParameter.primalCost(iStep) = namaParameter.primalCost(iStep) + tree.prob(i)*(funFvar.stateX(:,i)' *...
            stageCost.matQ*funFvar.stateX(:, i) + funFvar.inputU(:,i)'*stageCost.matR*funFvar.inputU(:,i));
    end
    dualVariableVec = [reshape(dualVar.y, numDualVarNode, 1); reshape(cell2mat(dualVar.yt), numDualVarLeave, 1)];
    for i = 1:numScen
        namaParameter.primalCost(iStep) = namaParameter.primalCost(iStep)+tree.prob(tree.leaves(i))*(funFvar.stateX(:,tree.leaves(i))'*...
            terminalCost.matVf{i} *funFvar.stateX(:,tree.leaves(i)));
    end
    namaParameter.dualCost(iStep) = namaParameter.primalCost(iStep) + dualVariableVec'*fixedPointResidualVec;
    namaParameter.dualGap(iStep) = dualVariableVec'*fixedPointResidualVec;
    namaParameter.normFixedPointResidual(iStep) = norm(fixedPointResidualVec);
    if(norm(fixedPointResidualVec) < obj.algorithmParameter.normFixedPointResidual)
        namaParameter.iterate = iStep;
        break
    else 
        iStep = iStep + 1;
    end
end
namaParameter.timeSolve = toc;
namaParameter.solveInvokCount = solveStepDetails.invokCount;
end 