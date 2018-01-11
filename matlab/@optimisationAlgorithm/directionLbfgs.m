function [ dirEnvelop , directionParamter] = directionLbfgs( obj, gradientEnv, oldGradientEnv, dualVarY, oldDualVarY)
%
% This function calculate the direction using - limited memory BFGS method,
% quasi-newton based direction update
%
% Syntax : 
%   [ dirEnvelop , directionParamter] = directionLbfgs( obj, gradientEnv, oldGradientEnv, dualVarY, oldDualVarY)
%
% Input  :    obj          :   algorithm object 
%             graientdEnv  :   gradient of the envelope
%             oldGradientEnv : past gradient gradient envelope 
%             dualVarY     :   current dual variable  
%             Yold         :   past dual variable 
% 
% Output :    
%  obj          :   algorithm object 
%  dirEnvelop   :   direction calculated with LBFGS method
%
%

system = obj.system;
constraint = system.constraint;
lbfgsParameter = obj.algorithmParameter.lbfgsParameter;
memory = lbfgsParameter.memory;
alphaC = lbfgsParameter.alphaC;

ny = size(constraint.matF{1}, 1);
tree = system.tree;
numNonLeaf = length(tree.children);
numScen = length(tree.leaves);
numDualVarNonLeaf = obj.algorithmParameter.numDualVarNonLeaf;
numDualVarLeave = obj.algorithmParameter.numDualVarLeave;
numDualVar = numDualVarNonLeaf + numDualVarLeave;

sIter = zeros(numDualVar, 1);
gradIter = zeros(numDualVar, 1);

sIter(1:numDualVarNonLeaf, 1) = reshape(dualVarY.y - oldDualVarY.y, numDualVarNonLeaf, 1);
sIter(numDualVarNonLeaf+1:numDualVar, 1) = reshape(cell2mat(dualVarY.yt) - cell2mat(oldDualVarY.yt),...
    numDualVarLeave, 1);
gradIter(1:numDualVarNonLeaf, 1) = reshape(gradientEnv.y, numDualVarNonLeaf, 1);
gradIter(numDualVarNonLeaf+1:numDualVar, 1) = reshape(cell2mat(gradientEnv.yt), numDualVarLeave, 1);
gradIterOld(1:numDualVarNonLeaf, 1) = reshape(oldGradientEnv.y, numDualVarNonLeaf, 1);
gradIterOld(numDualVarNonLeaf+1:numDualVar, 1) = reshape(cell2mat(oldGradientEnv.yt), numDualVarLeave, 1);
deltaGrad  = gradIter - gradIterOld;
vecYSk = deltaGrad'*sIter;
if norm(gradIter) < 1,alphaC = 3;end
if vecYSk/(sIter'*sIter) > 1e-6*norm(gradIter) ^alphaC
    lbfgsParameter.colLbfgs = 1 + mod(lbfgsParameter.colLbfgs, memory);
    lbfgsParameter.memLbfgs = min(lbfgsParameter.memLbfgs + 1, memory);
    lbfgsParameter.matS(:, lbfgsParameter.colLbfgs) = sIter;
    lbfgsParameter.matY(:, lbfgsParameter.colLbfgs) = deltaGrad;
    lbfgsParameter.vecYS(lbfgsParameter.colLbfgs)  = vecYSk;
else
    lbfgsParameter.skipCount = lbfgsParameter.skipCount + 1;
end
matHessian = vecYSk/(deltaGrad'*deltaGrad);
if(matHessian < 0 || abs(matHessian - lbfgsParameter.matHessian) == 0)
    lbfgsParameter.matHessian = 1;
else
    lbfgsParameter.matHessian = matHessian;
end
lbfgsParameter.numeratorHessian = vecYSk;
lbfgsParameter.denominatorHessian = (deltaGrad'*deltaGrad);
lbfgsParameter.rho = lbfgsParameter.vecYS(lbfgsParameter.colLbfgs); 
vecDirEnvelop = LBFGS(lbfgsParameter.matS, lbfgsParameter.matY, lbfgsParameter.vecYS, lbfgsParameter.matHessian,...
    -gradIter, int32(lbfgsParameter.colLbfgs), int32(lbfgsParameter.memLbfgs));
dirEnvelop.y = reshape(vecDirEnvelop(1:numDualVarNonLeaf,1), ny, numNonLeaf);
currentDualVarLeave = 0;
for i=1:numScen
    iDualVar = size(obj.system.terminalConstraint.matFt{i}, 1);
    dirEnvelop.yt{i} = vecDirEnvelop(numDualVarNonLeaf + currentDualVarLeave+1:numDualVarNonLeaf +....
        currentDualVarLeave + iDualVar, 1);
    currentDualVarLeave = currentDualVarLeave + iDualVar;
end 
directionParamter.descentValue = gradIter'*vecDirEnvelop;
directionParamter.lbfgsParameter = lbfgsParameter;
%directionParamter.descentValue
if( directionParamter.descentValue > 0)
    error('wrongLbfgsDirection', 'lbfg direction is not a decent direction');
end
directionParamter.vecYSk = vecYSk;

end

