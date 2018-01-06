function [updateDualVariable, proximalParameter] = proximalGconjugate(obj, funFvar, dualVariable)
%
%  The function proximalGconjugate calculate the proximal for the function conjugate 
%    of function g. This function is the indicator function of the
%    constraints of the system. 
%
%  Syntax :
%    [Y,details_prox] = proximalGconj(dualGradient, dualVaraible)
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

tree = obj.tree;
system = obj.system;
constraint = system.constraint;
terminalConstraint = system.terminalConstraint;
numNonLeaf = length(tree.children);
numScen = length(tree.leaves);

if(strcmp(algorithmParameter.proxLineSearch, 'yes'))
    previousLambda = obj.algorithmParameter.lambda;
    
    for i = 1:numNonLeaf
        proximalParameter.matHz(:, i) = constraint.matF{i}*funFvar.stateX(:,i) + constraint.G{i}*funFvar.inputU(:,i);
        proximalParameter.primalConstraint(:, i) = proximalParameter.matHz(:,i) - constraint.g{i};
    end
    for i = 1:numScen
        proximalParameter.matHzTerminal{i,1} = terminalConstraint.matFt{i,1}*funFvar.stateX(:, tree.leaves(i));
        proximalParameter.primalTerminalConstraint{i,1} = proximalParameter.matHzTerminal{i,1} - terminalConstraint.gt{i};
    end
    
    backtrackingParameter.currentDualVaraible = dualVariable;
    backtrackingParameter.matHz = proximalParameter.matHz;
    backtrackingParameter.matHzTerminal = proximalParameter.matHzTerminal;
    backtrackingParameter.primalConstraint = proximalParameter.primalConstraint;
    backtrackingParameter.primalTerminalConstraint = proximalParameter.primalTerminalConstraint;

    lambda = obj.backtackingStepsize( previousLambda, backtrackingParameter);
    for i = 1:numNonLeaf
        updateDualVariable.y(:,i) = max(0, dualVariable.y(:,i) + lambda*(proximalParameter.primalConstraint(:,i)));
    end
    for i = 1:numScen
        updateDualVariable.yt{i,1} = max(0, dualVariable.yt{i,:} + lambda*(proximalParameter.primalTerminalConstraint{i,1}));
    end
    proximalParameter.lambda = lambda;
else
    lambda=obj.algorithmParameter.lambda;
    proximalParameter.lambda = lambda;
    for i = 1:numNonLeaf
        proximalParameter.matHz(:, i) = constraint.matF{i}*funFvar.stateX(:,i) + constraint.G{i}*funFvar.inputU(:,i);
        proximalParameter.primalConstraint(:, i) = proximalParameter.matHz(:,i) - constraint.g{i};
        updateDualVariable.y(:,i) = max(0, dualVariable.y(:,i) + lambda*(proximalParameter.primalConstraint(:, i)));
    end
    for i = 1:numScen
        proximalParameter.matHzTerminal{i,1} = terminalConstraint.matFt{i,1}*funFvar.stateX(:, tree.leaves(i));
        proximalParameter.primalTerminalConstraint{i,1} = proximalParameter.matHzTerminal{i,1}- terminalConstraint.gt{i};
        updateDualVariable.yt{i,1} = max(0, dualVariable.yt{i,:} + lambda*(proximalParameter.primalTerminalConstraint{i,1}));
    end
end 

end

