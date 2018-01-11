function obj  = scaleConstraintSystem( obj )
%
% The function scaleConstrintSystem normalises the constraints of the system
%
% Syntax : 
%  obj = scaleConstraintSystem(obj)
%

numNode = length(obj.tree.stage);
numScenario = length(obj.tree.leaves);
numConstraint = size(obj.constraint.matF{1}, 1);

for iNode = 1:numNode - numScenario
    for iConstraint = 1:numConstraint
        if(abs(obj.constraint.g{iNode}(iConstraint)) > 0)
            obj.constraint.matF{iNode}(iConstraint, :) = obj.constraint.matF{iNode}(iConstraint, :)/...
                abs(obj.constraint.g{iNode}(iConstraint));
            obj.constraint.matG{iNode}(iConstraint, :) = obj.constraint.matG{iNode}(iConstraint, :)/...
                abs(obj.constraint.g{iNode}(iConstraint));
            obj.constraint.g{iNode}(iConstraint) = obj.constraint.g{iNode}(iConstraint)/...
                abs(obj.constraint.g{iNode}(iConstraint));
        end
    end
end

numConstraint = size(obj.terminalConstraint.matFt{1},1);
for iSec = 1:numScenario
    for iConstraint = 1:numConstraint
        if(abs(obj.terminalConstraint.gt{iSec}(iConstraint)) > 0)
            obj.terminalConstraint.matFt{iSec}(iConstraint,:) = obj.terminalConstraint.matFt{iSec}(iConstraint, :)/...
                obj.terminalConstraint.gt{iSec}(iConstraint);
            obj.terminalConstraint.gt{iSec}(iConstraint) = obj.terminalConstraint.gt{iSec}(iConstraint)/...
                abs(obj.terminalConstraint.gt{iSec}(iConstraint));
        end
    end
end

end

