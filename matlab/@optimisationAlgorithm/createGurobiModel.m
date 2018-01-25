function [ model] = createGurobiModel(obj)
%
% This functions solves the optimisation of the scenario-tree based mpc
%   with gurobi solver 
% 
% Syntax 
%  model = gurobiSolve(obj)
%
% Input
%   obj : optimisation object
% 
% Output
%   model : optimisation model for gurobi solver  
%

system = obj.system;
dynamics = system.dynamics;
tree = system.tree;
numNode = length(tree.stage);
numScen = length(tree.leaves);
numNonLeaf = numNode - numScen;
nx = size(system.dynamics.matA{1}, 1);
nu = size(system.dynamics.matB{1}, 2);
ny = size(system.constraint.matF{1}, 1);
nz = nx + nu;
numDualVarTerm = 0;
for i = 1:numScen
    numDualVarTerm = numDualVarTerm + size(system.terminalConstraint.matFt{i}, 1);
end
numDualVar = ny*numNonLeaf + numDualVarTerm;
model.Q = sparse(nz*numNonLeaf + numScen*nx, nz*numNonLeaf + numScen*nx);
model.A = sparse(ny*numNonLeaf + numDualVarTerm + nx*numNode, nz*numNonLeaf + nx*numScen);
model.rhs = zeros(ny*numNonLeaf + numDualVarTerm + nx*numNode, 1);
sparseMatQ = sparse(system.stageCost.matQ);
sparseMatR = sparse(system.stageCost.matR);
sparseLinearMatH = sparse([system.constraint.matF{1} system.constraint.matG{1}]);

currentNodePosition = 0;
for i = 1:numNonLeaf
    model.Q((i-1)*nz+1:(i-1)*nz+nx, (i-1)*nz+1:(i-1)*nz+nx) = tree.prob(i)*sparseMatQ;
    model.Q((i-1)*nz+nx+1:i*nz, (i-1)*nz+nx+1:i*nz) = tree.prob(i)*sparseMatR;
    numChild = length(tree.children{i});
    model.A((i-1)*ny+1:i*ny, (i-1)*nz+1:i*nz) = sparseLinearMatH;
    %
    for j = 1:numChild
        model.A(numDualVar + (currentNodePosition+j-1)*nx + 1:numDualVar + (currentNodePosition+j)*nx,...
            (i-1)*nz+1:i*nz) = sparse([dynamics.matA{currentNodePosition+j} dynamics.matB{currentNodePosition+j}]);
        model.rhs(numDualVar + (currentNodePosition+j-1)*nx + 1:numDualVar + (currentNodePosition+j)*nx,...
            1) = 1*-tree.value(currentNodePosition+j+1, :)';
        if(i <= numNonLeaf - numScen)
            model.A(numDualVar + (currentNodePosition+j-1)*nx + 1:numDualVar + (currentNodePosition+j)*nx,...
                (currentNodePosition+j)*nz+1:(currentNodePosition+j)*nz+nx) = -speye(nx);
        else
            model.A(numDualVar + (currentNodePosition+j-1)*nx+1: numDualVar + (currentNodePosition+j)*nx,...
                numNonLeaf*nz + (currentNodePosition - numNonLeaf+j)*nx + 1:...
                numNonLeaf*nz + (currentNodePosition - numNonLeaf+j+1)*nx) = -speye(nx);
        end
    end
    currentNodePosition = currentNodePosition + numChild;
    %}
end
model.rhs(1:ny*numNonLeaf) = kron(ones(numNonLeaf,1), system.constraint.g{1});
model.A(end - nx+1:end, 1:nx) = speye(nx);
for i = 1:numScen
    nyTerm = size(system.terminalConstraint.matFt{i}, 1);
    model.Q(nz*numNonLeaf + (i-1)*nx + 1:nz*numNonLeaf + i*nx, nz*numNonLeaf + (i-1)*nx + 1:...
        nz*numNonLeaf + i*nx) = tree.prob(tree.leaves(i))*sparse(system.terminalCost.matVf{i});
    model.A(numNonLeaf*ny + (i-1)*nyTerm + 1:numNonLeaf*ny + i*nyTerm, nz*numNonLeaf + (i-1)*nx+1:...
        nz*numNonLeaf + i*nx) = sparse(system.terminalConstraint.matFt{i});
    model.rhs(ny*numNonLeaf + (i-1)*nyTerm + 1:ny*numNonLeaf + i*nyTerm) = system.terminalConstraint.gt{i};
end
model.sense = [repmat('<',ny*numNonLeaf + nyTerm*numScen, 1);repmat('=', numNode*nx,1)];
model.obj = zeros(numNonLeaf*nz + numScen*nx, 1);
model.lb = -inf*ones(numNonLeaf*nz + numScen*nx, 1);


end

