function [funFvarUpdate, oracleGradParameter] = oracleDualGradientUpdate(obj, dir)
% 
% The function oracleDualGradientUpdate computes the dual gradient change
%   in the corresponding direction. The dual vector changes as y + \tau d and the 
%   dual gradient changes as x + \tau x(d). This function compute x(d).
% 
% Syntax :
%   [funFvarUpdate, oracleGradParameter] = oracleDualGradientUpdate(obj, dir)
% 
% Input : 
%   dir : the direction of the dual variable 
% 
% Output :
%   funFvarUpdate : the update of the dual gradient
%   oracleGradParameter : parameters of the oracleGradParameter 
%

persistent invokCount; 
if(obj.algorithmParameter.hessPersistance == 0)
    invokCount = 1;
else 
    invokCount = invokCount + 1;
end

system = obj.system;
dynamics = system.dynamics;
factorStepstruct = obj.factorStepStruct;

tree = system.tree;
nx = size(system.dynamics.matA{1}, 1);
nu = size(system.dynamics.matB{1}, 2);
numNode = length(tree.stage);
numScen = length(tree.leaves);
numNonLeaf = length(tree.children);

funFvarUpdate.stateX = zeros(nx, numNode);
funFvarUpdate.inputU = zeros(nu, numNonLeaf);
matS = zeros(nu, numNonLeaf);

q = zeros(nx, numNonLeaf);
qt = dir.yt;

for iPred = tree.predictionHorizon:-1:1
    nodesStage = find(tree.stage == iPred - 1);
    numNodeStage = length(nodesStage);
    for iNode = 1:numNodeStage
        numChild = length(tree.children{nodesStage(iNode)});
        if(numChild > 1)
            sumU = zeros(nu, 1);
            sumQ = zeros(nx, 1);
            for k = 1:numChild
                currentChildNode = tree.children{nodesStage(iNode)}(k);
                sumU = sumU + factorStepstruct.matTheta{currentChildNode - 1}*q(:, currentChildNode);
                sumQ = sumQ + factorStepstruct.matLambda{currentChildNode - 1}'*q(:,currentChildNode);
            end
            oracleGradParameter.sumU{nodesStage(iNode)} = sumU;
            matS(:, nodesStage(iNode)) = factorStepstruct.matPhi{nodesStage(iNode)}*dir.y(:,nodesStage(iNode)) + sumU;
            q(:, nodesStage(iNode)) = factorStepstruct.matD{nodesStage(iNode)}'*dir.y(:,nodesStage(iNode)) + sumQ;
        else
            if(iPred == tree.predictionHorizon)
                currentChildNode = tree.children{nodesStage(iNode)};
                matS(:,nodesStage(iNode)) = factorStepstruct.matPhi{nodesStage(iNode)}*dir.y(:,nodesStage(iNode))...
                    + factorStepstruct.matTheta{currentChildNode - 1}*qt{iNode};
                q(:,nodesStage(iNode)) = factorStepstruct.matD{nodesStage(iNode)}'*dir.y(:,nodesStage(iNode))...
                    + factorStepstruct.matLambda{currentChildNode - 1}'*qt{iNode};
            else
                sumQ = q(:,tree.children{nodesStage(iNode)});
                currentChildNode = tree.children{nodesStage(iNode)};
                matS(:,nodesStage(iNode)) = factorStepstruct.matPhi{nodesStage(iNode)}*dir.y(:,nodesStage(iNode))...
                    + factorStepstruct.matTheta{currentChildNode - 1}*sumQ;
                q(:,nodesStage(iNode)) = factorStepstruct.matD{nodesStage(iNode)}'*dir.y(:,nodesStage(iNode))...
                    + factorStepstruct.matLambda{currentChildNode - 1}'*sumQ;
            end
        end
    end
end

funFvarUpdate.stateX(:,1) = zeros(nx, 1);
for iPred = 1:numNonLeaf
    funFvarUpdate.inputU(:,iPred) = factorStepstruct.matK{iPred}*funFvarUpdate.stateX(:,iPred) + matS(:,iPred);
    for iNode = 1:length(tree.children{iPred})
        currentChildNode = tree.children{iPred}(iNode);
        funFvarUpdate.stateX(:,currentChildNode) = dynamics.matA{currentChildNode}*funFvarUpdate.stateX(:, iPred)+...
            dynamics.matB{currentChildNode}*funFvarUpdate.inputU(:,iPred);
    end
end

oracleGradParameter.S = matS;
oracleGradParameter.q = q;
oracleGradParameter.qt = qt;
oracleGradParameter.invokCount = invokCount; 

end


