classdef springMassSystem < matlab.mixin.Copyable

    properties(GetAccess = public, SetAccess = private)
        dynamics;
        constraint;
        stageCost;
        terminalCost;
        terminalConstraint;
        tree;
        initialState;
        verbose = 1; 
        systemParameter;
    end 

    methods( Access = public )
        function obj = springMassSystem( sampleTree, systemParameter )
            %
            % springMassSystem constructs the spring-mass system and associated 
            % scenario-tree that represent the propagration of the
            % uncertainty and the associated cost.
            %
            % SYNTAX
            %   springMass = springMassSystem( sampleTree, systemParameter );
            %
            % INPUT
            %   sampleTree : scenario tree that represent the uncertainty 
            %     structure 
            %   systemParameter : structure that contains the system parameters 
            %     masses = number of masses 
            %     samplingTime = sampling time of the discretisation 
            %     b = drag/friction/damping constant  
            %     k = spring constants 
            %     xMin = minimum state 
            %     xMax = maximum state 
            %     uMin = minmum input 
            %     uMax = maximum input 
            %     
            % TODO 
            %   Include parametric uncertainty for the system. 
            %
            
            if nargin < 2
                N = size(sampleTree.value, 2)/2;
                systemParameter = struct('masses', 1*ones(N ,1), 'b', 0.1*ones(N+1,1), ...
                    'k', 1*ones(N+1,1),'xMin', -5*ones(2*N,1), 'xMax', 5*ones(2*N,1), 'uMin', ...
                    -2*ones(N-1,1), 'uMax', 2*ones(N-1,1), 'samplingTime', 0.1,...
                    'parametric', false);
            end
            
            if(systemParameter.parametric)
                % TODO parametric uncertainty 
            else
                obj.tree = sampleTree;
                numMasses = length(systemParameter.masses);
                nx = 2*numMasses;
                nu = numMasses - 1;
                masses = systemParameter.masses;
                b = systemParameter.b;
                k = systemParameter.k;
                Ag = zeros(nx, nx);
                Bg = zeros(nx, nu);
                for iSec = 1:numMasses
                    h = 2*iSec - 1;
                    Ag(h, h+1) = 1; 
                    Ag(h+1, h+1) = -(b(iSec) + b(iSec+1))/masses(iSec);
                    Ag(h+1, h) = -(k(iSec) + k(iSec+1))/masses(iSec);
                    if iSec > 1,
                        Ag(h+1, h-2) = k(iSec)/masses(iSec);
                        Ag(h+1, h-1) = b(iSec)/masses(iSec);
                    end
                    if iSec < numMasses,
                        Ag(h+1, h+2) = k(iSec+1)/masses(iSec);
                        Ag(h+1, h+1) = b(iSec+1)/masses(iSec);
                    end
                    if iSec > 1
                        Bg(h+1, iSec-1) = -1/masses(iSec);
                    end
                    if iSec < N
                        Bg(h+1, iSec) = 1/masses(iSec);
                    end
                end
                %{
                sysContinous = ss(Ag, Bg, eye(nx), zeros(nx,nu));
                sysDiscrete = c2d(sysContinous, systemParameter.samplingTime );
                tol = 1e-6;
                
                matA = (1 - (abs(sysDiscrete.a) < tol)).*sysDiscrete.a;
                matB = (1 - (abs(sysDiscrete.b) < tol)).*sysDiscrete.b;
                %}
                discreteTime = systemParameter.samplingTime/5;
                matAd = eye(nx) + discreteTime*Ag;
                matBd = discreteTime*Bg;
                matA = matAd^5;
                matB = matBd;
                for i = 1:4
                    matB = matB + matAd^i*matBd;
                end 
                matF = [eye(nx); -eye(nx); zeros(2*nu, nx)];
                matG = [zeros(2*nx, nu); eye(nu); -eye(nu)];
                g = [systemParameter.xMax; -systemParameter.xMin;...
                    systemParameter.uMax; -systemParameter.uMin];
                
                nodes = length(sampleTree.stage);
                ns = length(obj.tree.leaves);
                for iNodes = 1:nodes
                    obj.dynamics.matA{iNodes} = matA;
                    obj.dynamics.matB{iNodes} = matB;
                    if(iNodes <= nodes - ns )
                        obj.constraint.matF{iNodes} = matF;
                        obj.constraint.matG{iNodes} = matG;
                        obj.constraint.g{iNodes} = g;
                    end
                end
                % stage cost function
                obj.stageCost.matQ = 3*eye(nx);
                obj.stageCost.matR = 2*eye(nu);
                % terminal stage
                for iSec = 1:ns
                    obj.terminalConstraint.matFt{iSec} = [eye(nx); -eye(nx)];
                    obj.terminalConstraint.gt{iSec} = (3 + 0.1*rand(1))*ones(2*nx,1);
                end
                matVf = dare(obj.dynamics.matA{1}, obj.dynamics.matB{1}, obj.stageCost.matQ, obj.stageCost.matR);
                for iSec = 1:ns
                    obj.terminalCost.matVf{iSec} = matVf;
                end
            end
            obj.systemParameter.lipschitzConstant = calculateLipschitz(obj);
        end
        
        function updateInitialState(obj, initialState)
            obj.initialState = initialState;
        end
        
        obj = scaleConstraintSystem( obj );
        
        obj = preconditionSystem( obj );
        
        %[sys,V,tree] = system_generation(obj, ops_sys_masses, ops_tree);
        
        %[value,primal_epsilon ] = cost_function(obj, Z);
    end
    
    methods(Access = private)
        lipschitzConstant = calculateLipschitz(obj);
    end 
    
    methods(Access = protected)
        function cpObj = copyElement(obj)
            cpObj = copyElement@matlab.mixin.Copyable(obj);
        end
    end 
end 
