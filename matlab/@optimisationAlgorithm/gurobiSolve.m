function [ output, gurobiParameter] = gurobiSolve(obj, model)
%
% This functions solves the optimisation of the scenario-tree based mpc
%   with gurobi solver
%
% Syntax
%  [ output, gurobiParameter] = gurobiSolve(obj, model)
%
% Input
%   model : optimisation model for gurobi solver
%
% Output
%   output : structure with the optimisation variables
%   gurobiParameter : structure with time
%

system = obj.system;
nx = size(system.dynamics.matA{1}, 1);
nu = size(system.dynamics.matB{1}, 2);
numNode = length(system.tree.stage);
numScen = length(system.tree.leaves);
numNonLeaf = numNode - numScen;
%system.initialState
model.rhs(end - nx+1:end) = system.initialState;
params.outputflag = 0;

tic
results = gurobi(model, params);
gurobiParameter.time = toc;

output.stateX = zeros(nx, numNode);
output.inputU = zeros(nu, numNonLeaf);
if(strcmp(results.status,'OPTIMAL'))
    %disp('OK');
    gurobiParameter.statusFlag = 1;
    nz = nx + nu;
    for j=1:numNonLeaf
        output.stateX(:,j) = results.x((j-1)*nz + 1:(j-1)*nz + nx);
        output.inputU(:,j) = results.x((j-1)*nz+nx+1:j*nz);
    end
    for j=1:numScen
        output.stateX(:, numNonLeaf+j) = results.x(numNonLeaf*nz + (j-1)*nx + 1:...
            numNonLeaf*nz + j*nx);
    end
else
    gurobiParameter.statusFlag = 0;
    %disp('Error');
end
end

