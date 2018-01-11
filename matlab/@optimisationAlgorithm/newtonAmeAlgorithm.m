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
% initialise the lbfgs memory
obj = createDirectionParameter(obj);

iStep = 1;
tic
while(iStep < obj.algorithmParameter.stepEnvelop )
    % step 1: gradient of the congujate of the f (smooth function) 
    [funFvar, solveStepDetails] = obj.solveStep(dualVar);
    if(iStep > 1)
        % step 2: proximal with respect to g or argmin of the argumented
        %   Lagrangian with respect to dual varaible
        [~, proximalParameter] = dualVariableUpdate(obj, funFvar, dualVar);
        % step 3: find the direction - calculated through L-BFGS method  
        funGvar = proximalParameter.funGvar;
        fixedPointResidual = proximalParameter.fixedPointResidual;
        obj.algorithmParameter.lambda = proximalParameter.lambda;
        [ dirEnvelop, directionParamter ] = obj.directionLbfgs(fixedPointResidual, oldFixedPointResidual,...
            dualVar, oldDualVar);
        obj.algorithmParameter.lbfgsParameter = directionParamter.lbfgsParameter;
        oldDualVar = dualVar;
        oldFixedPointResidual = fixedPointResidual;
        % step 4 line search on the Lagrangian
        primalVar.funFvar = funFvar;
        primalVar.funGvar = funGvar;
        [newtonUpdateDualVar, newtonLsParameter] = linesearchNewtonDir(obj, primalVar, dualVar, fixedPointResidual, dirEnvelop);
        [valueNamaUpdatedDualVar, valueParameter] = valueAugmentedLagrangian(obj, newtonLsParameter.funFvar,...
            newtonLsParameter.funGvar, newtonUpdateDualVar, newtonLsParameter.fixedPointResidual);
        % step 5 update the dual variable 
        dualVar.y = newtonUpdateDualVar.y - obj.algorithmParameter.lambda*newtonLsParameter.fixedPointResidual.y;
        for i = 1:numScen
            dualVar.yt{i} = newtonUpdateDualVar.yt{i} - obj.algorithmParameter.lambda*...
                newtonLsParameter.fixedPointResidual.yt{i};
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
        
        [valueNamaUpdatedDualVar, valueParameter] = valueAugmentedLagrangian(obj, funFvar,...
            proximalParameter.funGvar, oldDualVar, proximalParameter.fixedPointResidual);
        fixedPointResidualVec = zeros(numDualVarNode + numDualVarLeave, 1);
        fixedPointResidualVec(1:numDualVarNode, 1) = reshape(proximalParameter.fixedPointResidual.y, numDualVarNode, 1);
        fixedPointResidualVec(numDualVarNode + 1: numDualVarNode + numDualVarLeave, 1) = reshape(cell2mat(...
            proximalParameter.fixedPointResidual.yt), numDualVarLeave, 1);
    end
    % termination condition 
    namaParameter.lambda(iStep) = proximalParameter.lambda;
    namaParameter.primalCost(iStep) = valueParameter.primalValue;% primal cost;
    namaParameter.dualCost(iStep) = valueParameter.primalValue + valueParameter.dualGapValue;% dual cost;
    namaParameter.dualGap(iStep) = -valueParameter.dualGapValue;
    namaParameter.valueArgLagran(iStep) = valueNamaUpdatedDualVar;
    namaParameter.normFixedPointResidual(iStep) = norm(fixedPointResidualVec);
    if( iStep > 1)
        namaParameter.descentValue(iStep - 1) = directionParamter.descentValue;
        namaParameter.vecYSk(iStep - 1) = directionParamter.vecYSk;
        namaParameter.valueArgLagran(iStep - 1) = obj.valueAugmentedLagrangian(newtonLsParameter.funFvar,...
            newtonLsParameter.funGvar, dualVar, newtonLsParameter.fixedPointResidual);
    end
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