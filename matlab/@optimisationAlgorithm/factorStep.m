function obj = factorStep(obj)
%
% factorStep computes the factor step for calculating the dual gradient
% of the scenario-tree based mpc problem. These matricies are calcualted
% in factorStepParameter. These matrices have the structure of the
% scenario-tree. The fields in the factorStepStruct are same as the
% varaibles in the Algorithm in the paper. These parameters are constraint
% for a given scenario-tree and system matrices.
%
% Syntax :
%  obj = factorStep(obj);
%

tree = obj.system.tree;
dynamics = obj.system.dynamics;
stageCost = obj.system.stageCost;
constraint = obj.system.constraint;
terminalCost = obj.system.terminalCost;
terminalConstraint = obj.system.terminalConstraint;

nx = size(dynamics.matA{1}, 1);
nu = size(dynamics.matB{1}, 2);
nc = size(constraint.matF{1}, 1);

factorStepStruct = struct('matP', cell(1,1), 'vecC', cell(1,1), 'matD', cell(1,1),...
    'matLambda', cell(1,1), 'matPhi',cell(1,1),'matTheta',cell(1,1),'vecSigma',cell(1,1));

for iSec = 1:length(tree.leaves)
    factorStepStruct.matP{1, tree.leaves(iSec)} = tree.prob(tree.leaves(iSec))*terminalCost.matVf{iSec};
end

for iPred = tree.predictionHorizon :-1:1
    nodeStage = find(tree.stage == iPred - 1);
    numNodes = length(nodeStage);
    for jNodes = 1:numNodes
        %childrenStage = tree.prob(tree.children{nodeStage(jNodes)})/tree.prob(nodeStage(jNodes));
        %childrenStage = length(tree.children{nodeStage(jNodes)});
        numChild = length(tree.children{nodeStage(jNodes)});
        matPbar = zeros(nu);
        vecNetaU = zeros(nu,1);
        vecNetaX = zeros(nx,1);
        matKbar = zeros(nu, nx);
        for kChild = 1:numChild
            currentChildNode = tree.children{nodeStage(jNodes)}(kChild);
            matPbar = matPbar + dynamics.matB{currentChildNode}' * factorStepStruct.matP{currentChildNode} *...
                dynamics.matB{currentChildNode}; %\bar{P}
            vecNetaU = vecNetaU + dynamics.matB{currentChildNode}' * factorStepStruct.matP{currentChildNode} *...
                tree.value(currentChildNode, :)'; %phi_{k-1}^{(i)}
            vecNetaX = vecNetaX + dynamics.matA{currentChildNode}' * factorStepStruct.matP{currentChildNode} *...
                tree.value(currentChildNode,:)'; %phi_{k-1}^{(i)}
            matKbar = matKbar + dynamics.matB{currentChildNode}' * factorStepStruct.matP{currentChildNode} *...
                dynamics.matA{currentChildNode};
        end
        
        % terms in the control u_{k-1}^{\star (i)}
        matRbar = 2*(tree.prob(nodeStage(jNodes)) * stageCost.matR + matPbar);
        matInvRbar = matRbar\eye(nu);
        factorStepStruct.vecSigma{1, nodeStage(jNodes)} = -2*matInvRbar*vecNetaU; %sigma_{k-1}^{(i)}
        factorStepStruct.matK{1, nodeStage(jNodes)} = -2*matInvRbar*matKbar; %K_{k-1}^{(i)}
        factorStepStruct.matPhi{1, nodeStage(jNodes)} = -matInvRbar*constraint.matG{nodeStage(jNodes)}';%\Phi_{k-1}^{(i)}
        
        if(iPred == tree.predictionHorizon )
            for kChild = 1:numChild
                currentChildNode = tree.children{nodeStage(jNodes)}(kChild);
                factorStepStruct.matTheta{1, currentChildNode - 1} = -matInvRbar * dynamics.matB{currentChildNode}' * ...
                    terminalConstraint.matFt{jNodes}'; %\Theta_{k-1}^{(i)}
            end
        else
            for kChild=1:numChild
                currentChildNode = tree.children{nodeStage(jNodes)}(kChild);
                factorStepStruct.matTheta{1, currentChildNode - 1} = -matInvRbar * dynamics.matB{currentChildNode}'; %\Theta_{k-1}^{(i)}
            end
        end       
        % terms in the linear cost
        factorStepStruct.vecC{1, nodeStage(jNodes)} = 2*(vecNetaX + factorStepStruct.matK{1, nodeStage(jNodes)}' * vecNetaU);
        factorStepStruct.matD{1, nodeStage(jNodes)} = constraint.matF{nodeStage(jNodes)} + constraint.matG{nodeStage(jNodes)} *...
            factorStepStruct.matK{nodeStage(jNodes)}; %d_{k-1}^{(i)}
        if( iPred == tree.predictionHorizon )
            for kChild = 1:numChild
                currentChildNode = tree.children{nodeStage(jNodes)}(kChild);
                factorStepStruct.matLambda{1, currentChildNode - 1} = terminalConstraint.matFt{jNodes} * (dynamics.matA{currentChildNode} + ...
                    dynamics.matB{currentChildNode} * factorStepStruct.matK{nodeStage(jNodes)}); %f_{k-1}^{(i)}
            end
        else
            for kChild = 1:numChild
                currentChildNode = tree.children{nodeStage(jNodes)}(kChild);
                factorStepStruct.matLambda{1, currentChildNode - 1} = (dynamics.matA{currentChildNode} + dynamics.matB{currentChildNode} *...
                    factorStepStruct.matK{nodeStage(jNodes)});%f_{k-1}^{(i)}
            end
        end
        %Quadratic cost
        if(iPred == tree.predictionHorizon)
            factorStepStruct.matP{nodeStage(jNodes)} = tree.prob(nodeStage(jNodes))*(stageCost.matQ + factorStepStruct.matK{nodeStage(jNodes)}' *...
                stageCost.matR * factorStepStruct.matK{nodeStage(jNodes)});
            for kChild = 1:numChild
                factorStepStruct.matP{nodeStage(jNodes)} = factorStepStruct.matP{nodeStage(jNodes)} + (dynamics.matA{tree.children{nodeStage(jNodes)}(kChild)} +...
                    dynamics.matB{tree.children{nodeStage(jNodes)}(kChild)} * factorStepStruct.matK{nodeStage(jNodes)})' * ...
                    factorStepStruct.matP{tree.children{nodeStage(jNodes)}(kChild)} * (dynamics.matA{tree.children{nodeStage(jNodes)}(kChild)} +...
                    dynamics.matB{tree.children{nodeStage(jNodes)}(kChild)}*factorStepStruct.matK{nodeStage(jNodes)});
            end
        else
            factorStepStruct.matP{nodeStage(jNodes)} = tree.prob(nodeStage(jNodes))*(stageCost.matQ + factorStepStruct.matK{nodeStage(jNodes)}' *...
                stageCost.matR * factorStepStruct.matK{nodeStage(jNodes)});
            for kChild = 1:numChild
                currentChildNode = tree.children{nodeStage(jNodes)}(kChild);
                factorStepStruct.matP{nodeStage(jNodes)} = factorStepStruct.matP{nodeStage(jNodes)} + factorStepStruct.matLambda{currentChildNode - 1}' * ...
                    factorStepStruct.matP{currentChildNode} * factorStepStruct.matLambda{currentChildNode - 1};
            end
        end
    end
end

obj.factorStepStruct = factorStepStruct;

end

