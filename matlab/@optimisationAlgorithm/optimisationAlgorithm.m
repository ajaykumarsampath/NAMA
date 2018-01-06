classdef optimisationAlgorithm
    
    properties( GetAccess = public, SetAccess = private)
        system;
        factorStepStruct;
        dualVariable;
        stateVariable;
        controlVariable;
    end
    
     properties( Access = public)
        algorithmParameter;
    end
    
    methods( Access = public )
        function obj = optimisationAlgorithm( system, algorithmParameter)
            %
            % Class optimisationAlgorithm contain the optimisation
            %   algorithms to solve the scenario-tree based MPC control. 
            %   The algoruithms in this classes are 
            %     - dual acceleration proximal gradient method (dualApg)
            %     - forward-backward envelop methods with quasiNetwon update (LbfgsFbe)
            %     - newton alternate minimisation envelope (LbfgsNama)
            % Syntax: 
            %   obj = optimisationAlgorithm( system, algorithmParameter)
            % 
            % Input:
            %   system : contain the scenario-tree representation of the
            %     dynamic system 
            %   algorithmParameter : contain the parameters for the
            %     algorithm like the maximum number of iterates, primal and
            %     dual infeasibility conditions for the termination
            %     conditions, lipschitz constant.
            %
              
            if (nargin == 1)
                algorithmParameter.primalInfeasibility = 1e-2;
                algorithmParameter.dualInfeasibility = 1e-2;
                algorithmParameter.normFixedPointResidual = 1e-2; 
                algorithmParameter.stepApg = 1000;
                algorithmParameter.proxLineSearch = 'no';
                algorithmParameter.algorithmUsed = 'dual-APG';
                algorithmParameter.direction = 'L-BFGS';
                algorithmParameter.lineSearchDirection = 'Wolfe';
                algorithmParameter.stepEnvelop = 1000;
                algorithmParameter.backtrackingBeta = 0.5;
                
                algorithmParameter.lbfgsParameter.colLbfgs = 1;
                algorithmParameter.lbfgsParameter.memLbfgs = 0;
                algorithmParameter.lbfgsParameter.skipCount = 0;
                algorithmParameter.lbfgsParameter.alphaC = 1;
                algorithmParameter.lbfgsParameter.matHessian = 1;
                algorithmParameter.lbfgsParameter.memory = 5;
            end
            
            obj.system = system;
            obj.algorithmParameter = algorithmParameter;
            obj = createDirectionParameter(obj);
            if( ~isfield(algorithmParameter, 'lambda') )
                obj.algorithmParameter.lambda = 1/system.systemParameter.lipschitzConstant;
            end
            
            %{
            nx = size(system.dynamics.matA{1}, 1);
            nu = size(system.dynamics.matB{1}, 2);
            numStage = length(system.tree.stage);
            numScenario = length(system.tree.leaves);
            numNonLeaf = numStage - numScenario; 
            dimDual = 2*(numNonLeaf*(nx + nu) + numScenario*nx);
            
            memory = obj.algorithmParameter.memoryLbfgs;
            obj.algorithmParameter.parameterLbfgs.S = zeros(dimDual, memory); % dual variable
            obj.algorithmParameter.ops_FBE.Lbfgs.Y  = zeros(dimDual,memory); % dual gradient
            obj.algorithmParameter.ops_FBE.Lbfgs.YS = zeros(memory, 1);
              
            ops_APG.prox_LS = 'no';
            ops_FBE = ops_APG;
            ops_FBE.steps = 200;
            ops_APG.type = 'yes';
            
            ops_FBE.memory = 5;
            ops_FBE.LS = 'WOLFE';
            ops_FBE.direction = 'LBFGS';
            obj.algorithmParameter.ops_FBE.monotonicity = 'yes';
             
            default_options=struct('algorithm', 'APG', 'verbose', 0, ...
                'ops_APG', ops_APG, 'ops_FBE', ops_FBE);
            flds = fieldnames(default_options);
            for i=1:numel(flds)
                if (~isfield(obj.algorithmParameter, flds(i)) &&...
                        ~isstruct(default_options.(flds{i})))
                    obj.algorithmParameter.(flds{i}) = default_options.(flds{i});  
                else
                    if ~isfield(obj.algorithmParameter, flds(i))
                        obj.algorithmParameter.(flds{i}) = default_options.(flds{i});
                    else
                        if(isstruct(default_options.(flds{i})))
                            sub_flds = fieldnames(default_options.(flds{i}));
                            for j=1:numel(sub_flds)
                                if ~isfield(obj.algorithmParameter.(flds{i}), sub_flds(j))
                                    obj.algorithmParameter.(flds{i}).(sub_flds{j}) = ...
                                        default_options.(flds{i}).(sub_flds{j});
                                end
                            end
                        end
                    end
                end
            end
            
            if(strcmp(obj.algo_details.ops_FBE.direction, 'LBFGS'))
                % Create the buffer when the direction is calculted
                % using LBFGS algorithm
                obj.algorithmParameter.ops_FBE.Lbfgs.LBFGS_col = 1;
                obj.algorithmParameter.ops_FBE.Lbfgs.LBFGS_mem = 0;
                obj.algorithmParameter.ops_FBE.Lbfgs.skipCount = 0;
                obj.algorithmParameter.ops_FBE.alphaC=1;
                obj.algorithmParameter.ops_FBE.Lbfgs.H = 1;
                
                memory = obj.algorithmParameter.ops_FBE.memory;
                obj.algorithmParameter.ops_FBE.Lbfgs.S=zeros(dim_dual, memory); % dual variable 
                obj.algorithmParameter.ops_FBE.Lbfgs.Y  = zeros(dim_dual,memory); % dual gradient 
                obj.algorithmParameter.ops_FBE.Lbfgs.YS = zeros(memory, 1); 
            
            else
                % Present stage we used the CG method to calculate the
                % direction. The direction is updated using the last
                % gradient. 
                
                obj.algorithmParameter.ops_FBE.ConjGrad.prev_grad_norm=zeros(1,...
                    obj.algorithmParameter.ops_FBE.steps); % norm of the gradient
                obj.algorithmParameter.ops_FBE.ConjGrad.prev_dir=zeros(dim_dual,1); % previous direction 
                obj.algorithmParameter.ops_FBE.ConjGrad.prev_Grad=zeros(dim_dual,1);% previous gradient
                
            end
            % Monotonicity option
            obj.algorithmParameter.ops_FBE.monotonicity = 'no';
            %}
            obj.algorithmParameter.gradPersistance = 0;
            obj.algorithmParameter.hessPersistance = 0;
        end
        
        function obj = createDirectionParameter(obj, algorithmParameter)
            if(nargin == 1)
                algorithmParameter.memory = obj.algorithmParameter.lbfgsParameter.memory;
                algorithmParameter.direction = 'L-BFGS';
            end 
            nx = size(obj.system.dynamics.matA{1}, 1);
            %nu = size(obj.system.dynamics.matB{1}, 2);
            ny = size(obj.system.constraint.matF{1}, 1);
            numStage = length(obj.system.tree.stage);
            numScenario = length(obj.system.tree.leaves);
            numNonLeaf = numStage - numScenario;
            numDualVarNonLeaf = ny*numNonLeaf;
            numDualVarLeave = 0;
            for i = 1:numScenario
                numDualVarLeave = numDualVarLeave + size(obj.system.terminalConstraint.matFt{i}, 1);
            end 
            dimDual = numDualVarNonLeaf + numDualVarLeave;
            
            if( strcmp(algorithmParameter.direction, 'L-BFGS'))
                memory = algorithmParameter.memory;
                obj.algorithmParameter.lbfgsParameter.matS = zeros(dimDual, memory); % dual variable
                obj.algorithmParameter.lbfgsParameter.matY = zeros(dimDual,memory); % dual gradient
                obj.algorithmParameter.lbfgsParameter.vecYS = zeros(memory, 1);
            else
                obj.algorithmParameter.conjGradParameter.prevGradNorm = zeros(1,...
                    obj.algorithmParameter.stepEnvelop); % norm of the gradient
                obj.algorithmParameter.conjGradParameter.prevDir = zeros(dim_dual,1); % previous direction
                obj.algorithmParameter.conjGradParameter.prevGrad = zeros(dim_dual,1);% previous gradient
            end
            obj.algorithmParameter.numDualVarNonLeaf = numDualVarNonLeaf;
            obj.algorithmParameter.numDualVarLeave = numDualVarLeave;
        end
        
        
        obj = factorStep( obj );
        
        [strcutDualGrad, solveStepDetails] = solveStep(obj, dualVariable);
        
        [funGvar, proximalParameter] = proximalG(obj, funFvar, dualVariable);
        
        [updatedDualGradient, proximalDetails] = dualVariableUpdate(obj, primalVariable, dualVariable);
         
        [updatedDualGradient, proximalDetails] = proximalGconjugate(obj, primalVariable, dualVariable);
        
        [ lambda ] = backtackingStepsize( previousLambda, backtrackingParameter);
        
        [primalVariable, apgParameter] = dualApgAlgorithm(obj)
        
        [funFvarUpdate, oracleGradParameter] = oracleDualGradientUpdate(obj, dir)
        
        [matHzUpdate, oracleGradParameter] = oracleHessianVec(obj, dir);
                
        [ obj, dirEnvelop ] = directionLbfgs( obj, gradientEnv, oldGradientEnv, dualVarY, oldDualVarY);
        
        [ updatePrimalVar, updateDualVar ] = linesearchNewtonDir(obj, primalVar, dualVar, fixedResidual, dirEnvelop);
        
        [funFvar, namaParameter] = newtonAmeAlgorithm(obj);
        %{
        [ Hd ] = dual_hessian_free( obj,Y,d,Z)
        % This function approximate the dual hessian update function 
        
        [ Grad_env,Z,details] = grad_dual_envelop( obj,Y,x0)
        % This function calcualte the gradient of the envelope of the 
        % dual function 
        
        [ dir_env,beta ] = CG_direction( obj,Grad_env)
        % This function calculates the direction using conjugate-gradient 
        % method (in particulate Flecture-Reeves method)
        
        [ Z,Y1,details ] = Dual_FBE(obj,x0)
        % This function implements the L-BFGS method for the Forward-Backward
        % Envelope on the dual function 
        
        [Z,Y1,details] = Dual_FBE_extGrad(obj,x0)
        % This function implements the L-BFGS method for the Forward-Backward
        % Envelope on the dual function. The L-BFGS update is calculated after 
        % the buffer is filled 
        
        [ alpha,term_LS ] = wolf_linesearch( obj,Grad,Z,Y,d,ops)
        % This function calculates the step size for the direction calculated
        % from L-BFGS method. This stepsize should satisfy the 
        % strong wolfe condition. Algorithm 3.2 Nocedal and Wright
        
        [ alpha,term_LS ]= zoom_sectioning(obj,Y,d,aLo,aHo,ops)
        % This function calculates the step size for the direction calculated
        % from L-BFGS method. This stepsize should satisfy the 
        % strong wolfe condition. Algorithm 3.2 Nocedal and Wright
        
        [ alpha,details_LS ] = Goldstein_conditions( obj,Grad,Z,Y,d,ops)
        % This funciton calculates the step size for the direction
        % calculated from L-BFGS method. This step-size should satisfy the 
        % GOldstein conditions
        
        [Z,Y,details]=Dual_GlobalFBE(obj,x0)
        % This function is the implement with an intermediate L-BFGS step that 
        % decrease the cost on the envelope and later apply the proximal
        % gradient method. 
        
        [Z,Y,details]=Dual_FBEAdaptive(obj,x0)
        % This function implements the FBE algorithm with 
        % line search. This  algorithm is listed as Algortihm 1 
        % (without step 6; final prox step) in 
        % the paper "FORWARD-BACKWARD QUASI-NEWTON METHODS FOR NONSMOOTH 
        % OPTIMIZATION PROBLEMS". The direction is given either by CG
        % method or LBFGS method.
        
        [Z,Y,details]=Dual_GlobalFBEAdaptive(obj,x0)
        % This function implements the GlobalFBE algorithm with 
        % line search. This  algorithm is listed as Algortihm 1 in 
        % the paper "FORWARD-BACKWARD QUASI-NEWTON METHODS FOR NONSMOOTH 
        % OPTIMIZATION PROBLEMS". The direction is given either by CG
        % method or LBFGS method.
        
        [Y,details_prox]=GobalFBS_proximal_gcong(obj,Z,W)
        % This function is the implementation of the proximal on the
        % conjugate of the dual in the FBE function. 
        
        [ alpha,details_LS ] = LS_backtracking(obj,Grad,Z,Y,d,ops)
        % This function implements backtracking line search method for
        % the decrease of cost on the envelope. 
        
        [Z,Y,details]=Dual_GlobalFBE_version2(obj,x0)
        % This function implements the LBFGS
        
        [ alpha,details_LS ] = LS_backtrackingVersion2(obj,Grad,Z,Y,d,ops)
        % This function
        
        [ Lbfgs,dir_env ] = LBFGS_direction_version2( obj,Grad_env,Grad_envOld,Y,Yold)
        % This funciton 
        %
        [Z,Y0,details]=Dual_AccelGlobFBE(obj,x0)
        % This function is the implementation of the L-BFGS step with
        % accelerated step.
        
        [Z,Y0,details]=Dual_AccelGlobFBE_version2(obj,x0)
        % This function is the implementation of the L-BFGS step with
        % accelerated step--Version 2
        
        [env_dir,Lbfgs]=CUDA_LBFGS_direction(obj);
        % This function generate the data for testing CUDA LBFGS.  
        
        [Z,Y,details]=Dual_FBEAdaptiveCG(obj,x0);
        % This fucnction implements the FBE algorithm with direction given by 
        % CG methods. The line search is made robust my adding gradient
        % direction whenever the strong wolfe condition is not satisfied. 
        
        [ alpha,details_LS ] = WolfCndCGVersion( obj,Grad,Z,Y,d,ops);
        % This function implements the line search for the direction given
        % by CG method. 
        %}
    end
    
    methods( Access = private )
        alpha = calculateLipschitz(obj);
    end
end