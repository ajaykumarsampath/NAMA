function [funFvar, solveStepDetails] = solveStep(obj, dualVariable)
% 
% The function solveStep compute the dual gradient using the dynamics programming.  
% All the constant matrics are computed off-line. 
% 
% Syntax : 
%  [ sructDualGrad , Q] = solveStep(obj, dualVariable);
% 
% Input :     
%   dualVariable : dual variable at which the gradient of the conjugate function 
%     is to be calculated 
% 
% Output :
%  primalVariable : The gradient of the conjugate function, in the current
%   context the state and the input vectors along the scenario tree
%  solveStepDetails : contains the temporary details of the algorithm
% 
%

persistent invokCount; 
if( obj.algorithmParameter.gradPersistance == 0)
    invokCount = 1;
    obj.algorithmParameter.gradPersistance = 1;
else
    invokCount = invokCount + 1;
end 

system = obj.system;
tree = obj.system.tree;
factorStepStruct = obj.factorStepStruct;
dynamics = system.dynamics;

nx = size(dynamics.matA{1}, 1);
nu = size(dynamics.matB{1}, 2);
numNodes = length(tree.stage);
numNonLeaf = length(tree.children);
numScen = length(tree.leaves);

funFvar.stateX = zeros(nx, numNodes);
funFvar.inputU = zeros(nu, numNonLeaf);
matS = zeros(nu, numNonLeaf);
q = zeros(nx, numNonLeaf);
qt = cell(1, numScen);

for iScen = 1:numScen
    qt{1, iScen} = dualVariable.yt{iScen};
end

% backward substitution 
for iPred = tree.predictionHorizon:-1:1
    nodesStage = find(tree.stage == iPred-1);
    numNodeStage = length(nodesStage);
    for jNode = 1:numNodeStage
        numChildNode = length(tree.children{nodesStage(jNode)});
        if(numChildNode > 1)
            sumU = zeros(nu, 1);
            sumQ = zeros(nx, 1);
            for k = 1:numChildNode
                currentChildNode = tree.children{nodesStage(jNode)}(k);
                sumU = sumU + factorStepStruct.matTheta{currentChildNode - 1} * q(:, currentChildNode);
                sumQ = sumQ + factorStepStruct.matLambda{currentChildNode - 1}' * q(:, currentChildNode);
            end
            solveStepDetails.sumU{nodesStage(jNode)} = sumU;
            matS(:,nodesStage(jNode)) = factorStepStruct.matPhi{nodesStage(jNode)} * dualVariable.y(:,nodesStage(jNode))...
                + sumU + factorStepStruct.vecSigma{nodesStage(jNode)};
            q(:,nodesStage(jNode)) = factorStepStruct.matD{nodesStage(jNode)}'*dualVariable.y(:,nodesStage(jNode)) + sumQ...
                + factorStepStruct.vecC{nodesStage(jNode)} ;
        else
            if(iPred == tree.predictionHorizon)
                matS(:,nodesStage(jNode)) = factorStepStruct.matPhi{nodesStage(jNode)}*dualVariable.y(:,nodesStage(jNode))...
                    + factorStepStruct.matTheta{tree.children{nodesStage(jNode)} - 1}*qt{1,jNode} + factorStepStruct.vecSigma{nodesStage(jNode)};
                q(:,nodesStage(jNode)) = factorStepStruct.vecC{nodesStage(jNode)} + factorStepStruct.matD{nodesStage(jNode)}'*dualVariable.y(:,nodesStage(jNode)) ...
                    + factorStepStruct.matLambda{tree.children{nodesStage(jNode)}-1}'*qt{1,jNode};
            else
                sumQ = q(:,tree.children{nodesStage(jNode)});
                matS(:,nodesStage(jNode)) = factorStepStruct.matPhi{nodesStage(jNode)}*dualVariable.y(:,nodesStage(jNode))...
                    + factorStepStruct.matTheta{tree.children{nodesStage(jNode)}-1}*sumQ + factorStepStruct.vecSigma{nodesStage(jNode)};
                q(:,nodesStage(jNode)) = factorStepStruct.vecC{nodesStage(jNode)} + factorStepStruct.matD{nodesStage(jNode)}'*dualVariable.y(:,nodesStage(jNode)) ...
                    + factorStepStruct.matLambda{tree.children{nodesStage(jNode)}-1}'*sumQ;
            end
        end
    end
end

% forward substitution 
funFvar.stateX(:,1) = system.initialState;
for iPred = 1:numNonLeaf
    funFvar.inputU(:,iPred) = factorStepStruct.matK{iPred}*funFvar.stateX(:,iPred) + matS(:,iPred);
    for jNode = 1:length(tree.children{iPred})
        childNode = tree.children{iPred}(jNode);
        funFvar.stateX(:, childNode) = dynamics.matA{childNode} * funFvar.stateX(:,iPred) +...
            dynamics.matB{childNode}*funFvar.inputU(:,iPred) + tree.value(childNode, :)';
    end
end
solveStepDetails.matS = matS;
solveStepDetails.q = q;
solveStepDetails.qt = qt;
solveStepDetails.invokCount = invokCount; 

end


