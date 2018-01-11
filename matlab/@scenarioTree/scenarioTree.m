classdef scenarioTree
    
    properties( GetAccess = public, SetAccess = private)
        stage;
        value;
        prob;
        leaves;
        ancestor;
        children;
        branchingFactor;
        predictionHorizon;
    end 

    methods( Access = public )
        function obj = scenarioTree( scenarioTreeParameter )
            %
            % The scenarioTree class is used to define the propagation of
            % uncertanity. This is created by passing the structure - 
            % branching factor. The uncertainty that account at the each
            % node of the scenario-tree is created from the data that 
            % represent the uncertainty. When uncertanity is absent, 
            % random data from uniformly sampled distribution is used.
            %
            % Syntax:
            %   tree = scenarioTree(scenarioTreeParameter);
            %
            % INPUT:
            %   scenarioTreeParameter : scenrio tree parameters 
            %     Fields of scenarioTreeParameter
            %       nx : length of the uncertainty
            %       N  : prediction horizon of the scenario tree 
            %       branchingFactor : branchingFactor of the scenario tree
            %       uncertainData : uncertanity data that will used to
            %         generate the scenario tree 
            % 
            % TODO
            %   Implement functions to generate the scenario tree from the
            %     uncertainData
            %
            
            if(isfield(scenarioTreeParameter, 'uncertainData'))
                
            else
                nx = scenarioTreeParameter.nx;
                branchingFactorStage = length(scenarioTreeParameter.branchingFactor);
                obj.branchingFactor = ones(scenarioTreeParameter.N, 1);
                obj.branchingFactor(1:branchingFactorStage) = scenarioTreeParameter.branchingFactor;
                obj.predictionHorizon = scenarioTreeParameter.N;
                obj.stage = 0;
                obj.prob = 1;
                obj.leaves = 1;
                obj.ancestor = 0;
                obj.value(1, :) = zeros(1, nx);
                Ns = 1;
                for iPred = 1:scenarioTreeParameter.N
                    Ns = Ns*obj.branchingFactor(iPred);
                    currentNodeVec = obj.leaves(end) + 1:obj.leaves(end) + Ns;
                    obj.ancestor(currentNodeVec) = kron(obj.leaves, ones(1, obj.branchingFactor(iPred)));
                    obj.stage(currentNodeVec) = iPred*ones(Ns, 1);
                    obj.value(currentNodeVec, :) = 0.1*rand(Ns, nx);
                    if(iPred <= branchingFactorStage)
                        pd = rand(1, Ns);
                        if(iPred == 1)
                            obj.prob(currentNodeVec) = pd/sum(pd);
                            obj.children{obj.leaves} = currentNodeVec;
                        else
                            for i = 1:obj.branchingFactor(iPred - 1)
                               currentbranch = obj.branchingFactor(iPred);
                               pd((i-1)*currentbranch + 1: i*currentbranch) = pd((i-1)*currentbranch + 1:...
                                   i*currentbranch)/sum(pd((i-1)*currentbranch + 1: i*currentbranch));
                               obj.children{obj.leaves(i)} = currentNodeVec((i-1)*currentbranch + 1: i*currentbranch);
                            end
                            obj.prob(currentNodeVec) = pd;
                        end
                    else
                        obj.prob(currentNodeVec) = pd;
                        for i = 1:length(obj.leaves)
                            obj.children{obj.leaves(i)} = currentNodeVec(i);
                        end
                    end
                    obj.leaves = currentNodeVec;
                end
            end
        end
        
        function scenarioList = getScenarioList(obj)
            numScenario = length(obj.leaves);
            scenarioList = zeros(obj.predictionHorizon + 1, numScenario);
            scenarioList(obj.predictionHorizon + 1, :) = obj.leaves;
            for iPred = obj.predictionHorizon:-1:1
                scenarioList(iPred, :) = obj.ancestor(scenarioList(iPred + 1, :));
            end
        end
    end 
end
