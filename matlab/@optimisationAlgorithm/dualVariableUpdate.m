function [updateDualVariable, proximalParameter] = dualVariableUpdate(obj, funFvar, dualVar)
%
%  The function dualVaraibleUpdate update the dual variable by computing
%  the proximal with respect to the conjugate of g. This is broken into two
%  steps - proximal with g and then update the dual variable. This separating 
%  is based on the property the connect the proximal and its proximal conjugate. 
%  The proximal with g is calulated at a point such that this is equivalent to
%  computing the argmin of the agumented lagrangian with respect to z. 
%
%  Syntax :
%    [updateDualVariable, proximalParameter] = dualVariableUpdate(obj, primalVariable, dualVariable)
%
%  Input :
%    dualGradient : dual gradient/ gradient of the conjugate function. 
%      In the current context this is the state and input over the scenario tree  
%    dualVariable :  current dual variables
%
%  Output :            
%    dualVariable : updated dual variable
%    proximalDetails : structure that containt the Hx and Hx-z (primal 
%      infeasibiliy vector) 
%

system = obj.system;
tree = system.tree;
constraint = system.constraint;
terminalConstraint = system.terminalConstraint;
algorithmParameter = obj.algorithmParameter;
numNonLeaf = length(tree.children);
numScen = length(tree.leaves);
nx = size(system.dynamics.matA{1}, 1);
ny = size(constraint.matF{1});

if(strcmp(algorithmParameter.proxLineSearch, 'yes'))
    beta = 0.5;
    alpha = 0.5;
    while(1)
        [funGvar, proximalParameter] = proximalG(obj, funFvar, dualVar);
        proximalParameter.funGvar = funGvar;
        lambda = obj.algorithmParameter.lambda;
        fixedPointResidual = proximalParameter.fixedPointResidual;
        updateDualVariable.y = dualVar.y - lambda*(fixedPointResidual.y);
        for i = 1:numScen
            updateDualVariable.yt{i, 1} = dualVar.yt{i, 1} - lambda*(fixedPointResidual.yt{i, 1});
        end
        nextPrimalVariable = obj.solveStep(obj, updateDualVariable);
        for i = 1:numNonLeaf
            matHzNextIterate.y(:,i) = constraint.matF{i}*nextPrimalVariable.stateX(:,i) + constraint.matG{i}*nextPrimalVariable.inputU(:,i);
        end
        for i=1:numScen
            matHzNextIterate.yt{i,1} = terminalConstraint.matFt{i,1}*nextPrimalVariable.stateX(:, tree.leaves(i));
        end
        
        deltaDualGrad(1:ny*numNonLeaf, 1) = reshape(matHzNextIterate.y - proximalParameter.matHz, ny*numNonLeaf, 1);
        deltaDualGrad(ny*numNonLeaf + 1:ny*numNonLeaf + 2*numScen*nx, 1) = reshape(cell2mat(matHzNextIterate.yt) -...
            cell2mat(proximalParameter.yt), 2*numScen*nx, 1);
        
        deltaDualIterate(1:ny*numNonLeaf, 1) = reshape(updateDualVariable.y - dualVaraible.y, ny*numNonLeaf, 1);
        deltaDualIterate(ny*numNonLeaf + 1:ny*numNonLeaf + 2*numScen*nx, 1) = reshape(cell2mat(updateDualVariable.yt) -...
            cell2mat(dualVaraible.yt), 2*numScen*nx, 1);
        if(lambda*norm(deltaDualGrad) > alpha*norm(deltaDualIterate))
            obj.algorithmParameter.lambda = beta*lambda;
        else
            break
        end
    end
    proximalParameter.lambda = lambda;
else
    [funGvar, proximalParameter] = proximalG(obj, funFvar, dualVar);
    proximalParameter.funGvar = funGvar;
    lambda = obj.algorithmParameter.lambda;
    fixedPointResidual = proximalParameter.fixedPointResidual;
    updateDualVariable.y = dualVar.y - lambda*(fixedPointResidual.y);
    for i = 1:numScen
        updateDualVariable.yt{i} = dualVar.yt{i} - lambda*(fixedPointResidual.yt{i});
    end
    proximalParameter.lambda = lambda;
end 

end

