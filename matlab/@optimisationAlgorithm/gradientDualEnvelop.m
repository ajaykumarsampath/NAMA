function [ gradientEnvelop, fbeGradParameter] = gradientDualEnvelop( obj, fixedPointResidual)
%
% This function graidentDualEnvelop calcualtes the gradient of the dual 
%   FBE envelop of the optimisation problem resulting from scenario-mpc
%   problem.
% 
% Syntax :
%   [ gradientEnv, envelopParameter] = gradientDualEnvelop( obj, dualVar)
%
% Input :
%   fixedPointResidual :  Dual variable 
% 
% Output :
%   gradientEnvelop :  gradient of the dual FBE envelope 
%   fbeGradEnvelop  :  structure of the gradient of the FBE gradient
%

system = obj.system;
tree = system.tree;
numScen = length(tree.leaves);
lambda = obj.algorithmParameter.lambda;
[matHzUpdate, fbeGradParameter] = obj.oracleHessianVec(fixedPointResidual);
gradientEnvelop.y = fixedPointResidual.y + lambda*matHzUpdate.y;

for i = 1:numScen
    gradientEnvelop.yt{i} = fixedPointResidual.yt{i} + lambda*matHzUpdate.yt{i};
end

end

