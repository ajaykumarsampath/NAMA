function [ lambda ] = backtackingStepsize( previousLambda, backtrackingParameter)
%
% The function backtrackingProximal calculates the step-size for the 
%   proximal-gradient algorithm. This step-size is calcualted using
%   backtracking algorithm that decrease the step-size until it satisfies 
%   the condition of the costs.  
%
% Syntax : 
%  lambda = backtackingStepsize( previousLambda, backtrackingParameter )
%
% Input  : 
%  previousLambda  :  step-size before back tracking 
%  backtrackingParameter :  parameters for the backtracking algorithm.
%
% Output : 
%  lambda       :  Current step-size
%
%

system = obj.system;
constraint = system.constraint;
terminalConstraint = system.terminalConstraint;

tree = obj.tree;
numNode = length(tree.stage);
numScen = length(tree.leaves);
numNonLeaf = numNode - numScen;
nx = size(system.dynamics.matA{1}, 1);
ny = size(constraint.matF{1});

dualVaraible = backtrackingParameter.currentDualVaraible;
currentIterate.matHz = backtrackingParameter.matHz;
currentIterate.matHzTerminal = backtrackingParameter.matHzTerminal;
currentIterate.primalConstraint = backtrackingParameter.primalConstraint;
currentIterate.primalTerminalConstraint = backtrackingParameter.primalTerminalConstraint;

nextIterate.matHz = zeros(size(currentIterate.matHz));
nextIterate.matHzTerminal = cell(numScen, 1);
deltaDualIterate = zeros(ny*numNonLeaf + 2*numScen*nx, 1);
deltaDualGrad = zeros(ny*numNonLeaf + 2*numScen*nx, 1);

lambda = previousLambda;
beta = 0.5;
alpha = 0.5;

while(1)
    nextDualIterate.y = max(0, dualVaraible.y + lambda*currentIterate.primalConstraint);
    for i = 1:numScen
        nextDualIterate.yt{i,1} = max(0, dualVariable.yt{i,1} + lambda*(currentIterate.primalTerminalConstraint{i,1}));
    end
    nextDualGrad = obj.solveStep(obj, nextDualIterate);
    
    for i = 1:numNonLeaf
        nextIterate.matHz(:,i) = constraint.matF{i}*nextDualGrad.stateX(:,i) + constraint.matG{i}*nextDualGrad.inputU(:,i);
    end
    for i=1:numScen
        nextIterate.matHzTerminal{i,1} = terminalConstraint.matFt{i,1}*nextDualGrad.stateX(:, tree.leaves(i));
    end
    
    deltaDualGrad(1:ny*numNonLeaf, 1) = vec(nextIterate.matHz - currentIterate.matHz);
    deltaDualGrad(ny*numNonLeaf + 1:ny*numNonLeaf + 2*numScen*nx, 1) = vec(cell2mat(nextIterate.matHzTerminal) -...
        cell2mat(currentIterate.matHzTerminal));
    
    deltaDualIterate(1:ny*numNonLeaf, 1) = vec(nextDualIterate.y - dualVaraible.y);
    deltaDualIterate(ny*numNonLeaf + 1:ny*numNonLeaf + 2*numScen*nx, 1) = vec(cell2mat(nextDualIterate.yt) - cell2mat(dualVaraible.yt));
    
    if(lambda*norm(deltaDualGrad) > alpha*norm(deltaDualIterate))
        lambda = beta*lambda;
    else
        break
    end
end

end

