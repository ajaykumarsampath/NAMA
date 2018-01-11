function [obj] = preconditionSystem( obj, preconditionType )
%
% Function preconditionSystem precondition the scenario-mpc system to
%   improve the condition number of dual proximal-gradient methods.
%   Exact preconditioning in such cases is hard to compute as the involves
%   evaluating inversion of the large-scale problem that results with the
%   scenario-mpc. We identified the main reason for this ill-conditioning
%   are the low probabilities of the nodes. So the dual cost is multiplied
%   with the square-root of probabiliy of the node. Another varient is use
%   the Jacobi information of single scenario and multiple the dual cost.
%
% Syntax :
%   obj = preconditionSystem(obj, preconditionType);
%
% Input :
%   preconditionType : it can take variables 'Jacobi' or 'Simple'
%

if( nargin == 1)
    preconditionType = 'Jacobi';
end
obj.systemParameter.preconditionType = preconditionType;
if(strcmp(preconditionType, 'Jacobi'))   
    numStages = length(obj.tree.stage);
    numScenarios = length(obj.tree.leaves);
    nx = size(obj.dynamics.matA{1}, 1);
    nu = size(obj.dynamics.matB{1}, 2);
    nz = nx + nu;
    numConstraint = size(obj.constraint.matF{1},1);
    numTerminalConstraint = size(obj.terminalConstraint.matFt{1},1);
    scenarioList = obj.tree.getScenarioList();
    predHorizon = obj.tree.predictionHorizon;
    rowKktMat = predHorizon*nz + nx + (predHorizon + 1)*nx;
    offsetKktMat = predHorizon*nz + nx;
    matKkt = zeros(rowKktMat, rowKktMat);
    matKkt(1:predHorizon*nz, 1:nz*predHorizon) = kron(eye(predHorizon),...
        blkdiag(2*obj.stageCost.matQ, 2*obj.stageCost.matR));
    matKkt(predHorizon*nz+1:predHorizon*nz+nx, predHorizon*nz+1:predHorizon*nz+nx) = 2*obj.terminalCost.matVf{1};
    matKktEquality = zeros((predHorizon + 1)*nx, predHorizon*nz + nx); 
    
    matSys = [obj.dynamics.matA{1} obj.dynamics.matB{1} -eye(nx)];
    for iNode = 1:predHorizon
        matKktEquality((iNode-1)*nx+1:iNode*nx,(iNode-1)*nz+1:iNode*nz+nx) = matSys;
    end
    matKktEquality(predHorizon*nx+1:(predHorizon+1)*nx, 1:nx) = eye(nx);
    matKkt(offsetKktMat+1:end, 1:predHorizon*nz+nx) = matKktEquality;
    matKkt(1:predHorizon*nz+nx, predHorizon*nz+nx+1:predHorizon*nz+nx+(predHorizon+1)*nx) = matKktEquality';
    
    matInvKkt = matKkt\eye(rowKktMat);
    matInvKkt11 = matInvKkt(1:offsetKktMat, 1:offsetKktMat);
    
    matLinearOperatorH = zeros(numConstraint*predHorizon+numTerminalConstraint, predHorizon*nz+nx);
    for iPred = 1:predHorizon
        iNode = scenarioList(iPred, 1);
        matSys = [obj.constraint.matF{iNode} obj.constraint.matG{iNode}];
        matLinearOperatorH((iPred-1)*numConstraint+1:iPred*numConstraint, (iPred-1)*nz+1:iPred*nz) = matSys;
    end
    matLinearOperatorH(predHorizon*numConstraint+1:predHorizon*numConstraint + numTerminalConstraint,...
        predHorizon*nz+1:predHorizon*nz+nx) = obj.terminalConstraint.matFt{1};
    
    matDualHessian = matLinearOperatorH*matInvKkt11*matLinearOperatorH';
    obj.systemParameter.lipschitzConstant = 1/norm(matDualHessian, 2);
    matDiagDualHessian = diag(matDualHessian);
    matDiagDualHessian(1:2*nx) = 0;
    matInvSqrtDiagDualHessian = 1./sqrt(matDiagDualHessian);
    matInvSqrtDiagDualHessian(1:2*nx) = 0;
    for iNode = 1:numStages - numScenarios
        iPred = obj.tree.stage(iNode) + 1;
        obj.constraint.matF{iNode} = sqrt(obj.tree.prob(iNode)) * diag(matInvSqrtDiagDualHessian((iPred-1)*numConstraint +...
            1:iPred*numConstraint)) * obj.constraint.matF{iNode};
        obj.constraint.matG{iNode} = sqrt(obj.tree.prob(iNode)) * diag(matInvSqrtDiagDualHessian((iPred-1)*numConstraint +...
            1:iPred*numConstraint)) * obj.constraint.matG{iNode};
        obj.constraint.g{iNode} = sqrt(obj.tree.prob(iNode)) * diag(matInvSqrtDiagDualHessian((iPred-1)*numConstraint +...
            1:iPred*numConstraint)) * obj.constraint.g{iNode};
    end
    for iScen = 1:numScenarios
        obj.terminalConstraint.matFt{iScen} = sqrt(obj.tree.prob(obj.tree.leaves(iScen))) * diag(matInvSqrtDiagDualHessian(...
            predHorizon*numConstraint + 1:end)) * obj.terminalConstraint.matFt{iScen};
        obj.terminalConstraint.gt{iScen} = sqrt(obj.tree.prob(obj.tree.leaves(iScen))) * diag(matInvSqrtDiagDualHessian(...
            predHorizon*numConstraint + 1:end)) * obj.terminalConstraint.gt{iScen};
    end
elseif( strcmp(preconditionType, 'Simple') )
    
    numStages = length(obj.tree.stage);
    numScenarios = length(obj.tree.leaves);
    for iNode = 1:numStages - numScenarios
        obj.constraint.matF{iNode} = sqrt(obj.tree.prob(iNode))*(obj.constraint.matF{iNode});
        obj.constraint.matG{iNode} = sqrt(obj.tree.prob(iNode))*(obj.constraint.matG{iNode});
        obj.constraint.g{iNode} = sqrt(obj.tree.prob(iNode))*(obj.constraint.g{iNode});
    end
    for iScen = 1:numScenarios
        obj.terminalConstraint.matFt{iScen} = sqrt(tree.prob(tree.leaves(iScen)))*obj.terminalConstraint.matFt{iScen};
        obj.terminalConstraint.gt{iScen} = sqrt(tree.prob(tree.leaves(iScen)))*obj.terminalConstraint.gt{iScen};
    end
    
else
    error('springMassSystem:preconditionSystem', 'unspecified precondition type');
end

end

