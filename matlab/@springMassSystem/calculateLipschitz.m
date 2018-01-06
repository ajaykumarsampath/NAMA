function lipschitzConstant = calculateLipschitz( obj )
%
% This function calculates the lipschitz constant of the scenario-tree based 
% system dynamics. This value is used as the step size in the gradient
% algorithm for the dual proximal gradient method. 
%
% Syntax : 
%   lipschitzConstant = calculateLipschitz( obj )
% 

numNode = length( obj.tree.stage);
numScenario = length( obj.tree.leaves);

lipschitzConstant = norm( obj.constraint.matG{1}*(obj.stageCost.matR\obj.constraint.matG{1}'), 2);
for iNode = 2:numNode - numScenario 
    value = 1/obj.tree.prob(iNode) * norm( obj.constraint.matF{iNode}*(obj.stageCost.matQ\obj.constraint.matF{iNode}')...
        + obj.constraint.matG{iNode}*(obj.stageCost.matR\obj.constraint.matG{iNode}'), 2 );
    lipschitzConstant = max(lipschitzConstant, value);
end
for iSec = 1:numScenario
    value = 1/obj.tree.prob(numNode - numScenario + iSec) * norm(obj.terminalConstraint.matFt{iSec}*...
        (obj.terminalCost.matVf{iSec}\obj.terminalConstraint.matFt{iSec}'));
    lipschitzConstant = max(lipschitzConstant, value);
end

end

