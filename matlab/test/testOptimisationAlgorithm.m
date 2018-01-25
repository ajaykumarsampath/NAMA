classdef testOptimisationAlgorithm < matlab.unittest.TestCase

    properties
        testPath;
        testSystem;
        testScenarioTree;
        testApgAlgorithm;
        testNamaAlgorithm;
        testGlobalFbeAlgorithm;
        testGurobiModel;
    end
    
    methods(TestClassSetup)
        function addpathToTestClass(testCase)
            p = path;
            testCase.addTeardown(@path, p);
            testCase.testPath = fileparts(mfilename('fullpath'));
            addpath(fullfile(testCase.testPath, '..'));
            addpath(fullfile(testCase.testPath, '../../../../../../software/gurobi702/linux64/matlab/'));
            addpath(fullfile(testCase.testPath, '../../../../../../software/YALMIP-master/'));
        end
        function createOptimisationAlgorithm(testCase)
            testCase.testApgAlgorithm = optimisationAlgorithm( testCase.testSystem );
            testCase.testApgAlgorithm = testCase.testApgAlgorithm.factorStep();
            testCase.testNamaAlgorithm = optimisationAlgorithm( testCase.testSystem );
            testCase.testNamaAlgorithm = testCase.testNamaAlgorithm.createDirectionParameter();
            testCase.testNamaAlgorithm = testCase.testNamaAlgorithm.factorStep();
            testCase.testGlobalFbeAlgorithm = optimisationAlgorithm( testCase.testSystem );
            testCase.testGlobalFbeAlgorithm = testCase.testGlobalFbeAlgorithm.createDirectionParameter();
            testCase.testGlobalFbeAlgorithm = testCase.testGlobalFbeAlgorithm.factorStep();
            testCase.testGurobiModel = testCase.testApgAlgorithm.createGurobiModel();
        end 
    end
    
    methods
        function createScenarioTree(testCase)
            masses = 5;
            scenarioTreeParameter.nx = 2*masses;
            scenarioTreeParameter.branchingFactor = 2*ones(1, 5);
            scenarioTreeParameter.N = 10;
            testCase.testScenarioTree = scenarioTree(scenarioTreeParameter);
        end
        function createSpringMassSystem(testCase)
            testCase.testSystem = springMassSystem(testCase.testScenarioTree);
        end
    end
    
    properties( TestParameter)
        testInitialState = {0.8*[0.4; 0.7; 0.8; -0.5; 2.5; 1.7; -2.9; -3.6; 0.9; 1.9],...
            1.3*[0.4; 0.7; 0.8; -0.5; 2.5; 1.7; -2.9; -3.6; 0.9; 1.9]};
        %testInitialState = {rand(10, 1)};
    end 
    
    methods(Test)
        function testImplementationSolveStep(testCase)
            system = testCase.testSystem;
            constraint = system.constraint;
            terminalConstraint = system.terminalConstraint;
            tree = system.tree;
            numScen = length(tree.leaves);
            numNode = length(tree.stage);
            
            % initialise the dual variables
            dualVar.y = zeros( size(constraint.matF{1}, 1), numNode - numScen);
            dualVar.yt = cell( numScen, 1);
            for i = 1:numScen
                dualVar.yt{i} = zeros( size(terminalConstraint.matFt{i}, 1), 1);
            end
            initialState = ones(10, 1);
            testCase.testNamaAlgorithm.system.updateInitialState(initialState);
            [funFvar, solveStepParameter] = testCase.testNamaAlgorithm.solveStep(dualVar);
        end
        function testImplementAlgorithm(testCase, testInitialState)
            testCase.testNamaAlgorithm.system.updateInitialState( testInitialState);
            [funFvar, namaParameter] = testCase.testNamaAlgorithm.newtonAmeAlgorithm();
            figure
            subplot(3, 1, 1)
            plot(namaParameter.primalCost)
            hold all;
            title('primal cost NAMA algorithm')
            subplot(3, 1, 2)
            plot(namaParameter.dualCost)
            hold all;
            title('dual cost NAMA algorithm')
            subplot(3, 1, 3)
            plot(namaParameter.valueArgLagran)
            hold all;
            title('argumented Lagrangian NAMA algorithm')
            if(namaParameter.iterate > 1)
                minDescentDirection = max(namaParameter.descentValue);
            end
            assert( minDescentDirection <= 0, 'lbfgs direction in NAMA is not decent');
            %
            testCase.testGlobalFbeAlgorithm.system.updateInitialState(testInitialState);
            [funFvarFbeAlgo, globalFbeParameter] = testCase.testGlobalFbeAlgorithm.dualGlobalFbeAlgorithm();
            %figure
            subplot(3, 1, 1)
            plot(globalFbeParameter.primalCost)
            %title('primal cost FBE algorithm')
            subplot(3, 1, 2)
            plot(globalFbeParameter.dualCost)
            %title('dual cost FBE algorithm')
            subplot(3, 1, 3)
            plot(globalFbeParameter.valueArgLagran)
            legend('NAMA', 'globalFbe')
            title('argumented Lagrangian FBE algorithm')
            if(globalFbeParameter.iterate > 1)
                minDescentDirection = max(globalFbeParameter.descentValue);
            end
            assert( minDescentDirection <= 0, 'lbfgs direction in FBE is not decent');
            %}
            testCase.testApgAlgorithm.system.updateInitialState(testInitialState);
            [funFvarApgAlgo, apgParameter] = testCase.testApgAlgorithm.dualApgAlgorithm();
            %figure
            subplot(3, 1, 1)
            plot(apgParameter.primalCost)
            title('primal cost')
            legend('NAMA', 'globalFbe', 'dualAPG')
            subplot(3, 1, 2)
            plot(apgParameter.dualCost)
            title('dual cost')
            legend('NAMA', 'globalFbe', 'dualAPG')
            if(norm(funFvar.stateX - funFvarApgAlgo.stateX) > 0.05)
                display(norm(funFvar.stateX - funFvarApgAlgo.stateX));
                warning('mismatchApgVsNama', 'variation in the norm of the state in accelarated and NAMA algorithm')
            end
            if(norm(funFvar.stateX - funFvarFbeAlgo.stateX) > 0.05)
                display(norm(funFvar.stateX - funFvarFbeAlgo.stateX));
                warning('mismatchglobalFbeVsNama', 'variation in the norm of the state in globalFBE and NAMA algorithm')
            end
        end
        function testOracleInNewtonAmeAlgorithm(testCase)
            system = testCase.testSystem;
            constraint = system.constraint;
            terminalConstraint = system.terminalConstraint;
            tree = system.tree;
            numScen = length(tree.leaves);
            numNode = length(tree.stage);
            
            lambda = 0.5;
            dualVar.y = zeros( size(constraint.matF{1}, 1), numNode - numScen);
            dirEnvelop.y = rand( size(constraint.matF{1}, 1), numNode - numScen);
            updateDualVar.y = dualVar.y + lambda*dirEnvelop.y;
            dualVar.yt = cell( numScen, 1);
            dirEnvelop.yt = cell(numScen, 1);
            for i = 1:numScen
                dualVar.yt{i} = zeros( size(terminalConstraint.matFt{i}, 1), 1);
                dirEnvelop.yt{i} = rand( size(terminalConstraint.matFt{i}, 1), 1);
                updateDualVar.yt{i} = dualVar.yt{i} + lambda*dirEnvelop.yt{i};
            end
            initialState = ones(10, 1);
            testCase.testNamaAlgorithm.system.updateInitialState(initialState);
            funFvar = testCase.testNamaAlgorithm.solveStep(dualVar);
            funFvarDirection = testCase.testNamaAlgorithm.oracleDualGradientUpdate(dirEnvelop);
            expectedFunFvar = testCase.testNamaAlgorithm.solveStep(updateDualVar);
            errorFunFvarState = norm(funFvar.stateX + lambda*funFvarDirection.stateX - expectedFunFvar.stateX);
            errorFunFvarInput = norm(funFvar.inputU + lambda*funFvarDirection.inputU - expectedFunFvar.inputU);
            assert( max(errorFunFvarState, errorFunFvarInput) <= 1e-6, ['Oracle update or the dual gradient]'...
                'mismatches the dual gradient']);
        end
        function testPreconditionApgAlgorithm(testCase, testInitialState)
            preconditionedSystem = copy(testCase.testSystem);
            preconditionedSystem.scaleConstraintSystem();
            preconditionedSystem.preconditionSystem();
            preconditionApgAlgorithm = optimisationAlgorithm( preconditionedSystem );
            preconditionApgAlgorithm = preconditionApgAlgorithm.factorStep();
            preconditionApgAlgorithm.system.updateInitialState(testInitialState);
            [funFvarPrecndApgAlgo, apgPrecondParameter] = preconditionApgAlgorithm.dualApgAlgorithm();
            preconditionFbeAlgorithm = optimisationAlgorithm( preconditionedSystem );
            preconditionFbeAlgorithm = preconditionFbeAlgorithm.factorStep();
            preconditionFbeAlgorithm.system.updateInitialState(testInitialState);
            [funFvarPrecndFbeAlgo, fbePrecondParameter] = preconditionFbeAlgorithm.dualGlobalFbeAlgorithm();
            testCase.testApgAlgorithm.system.updateInitialState(testInitialState);
            [funFvarApgAlgo, apgParameter] = testCase.testApgAlgorithm.dualApgAlgorithm();
            figure
            subplot(2, 1, 1)
            plot(apgParameter.primalCost);
            hold all;
            plot(apgPrecondParameter.primalCost);
            plot(fbePrecondParameter.primalCost);
            title('primal cost')
            subplot(2, 1, 2)
            plot(apgParameter.dualCost);
            hold all;
            plot(apgPrecondParameter.dualCost);
            plot(fbePrecondParameter.dualCost);
            title('dual cost')
            if(norm(funFvarPrecndApgAlgo.stateX - funFvarApgAlgo.stateX) > 0.05)
               warning('mismatchPreconditionSys', 'variation in the norm of the state of actual and preconditioned algorithm')
            end
            if(norm(funFvarPrecndApgAlgo.stateX - funFvarPrecndFbeAlgo.stateX) > 0.05)
               warning('mismatchNamaVsApg', 'variation in the norm of the state of actual and preconditioned algorithm')
            end
        end
        function testPreconditionAlgorithm(testCase, testInitialState)
            testCase.testApgAlgorithm.system.updateInitialState(testInitialState);
            [ funFvarGurobi, gurobiParameter] = testCase.testApgAlgorithm.gurobiSolve(testCase.testGurobiModel);
            if(gurobiParameter.statusFlag)
                preconditionedSystem = copy(testCase.testSystem);
                preconditionedSystem.scaleConstraintSystem();
                preconditionedSystem.preconditionSystem();
                preconditionAlgorithm = optimisationAlgorithm( preconditionedSystem );
                preconditionAlgorithm = preconditionAlgorithm.factorStep();
                preconditionAlgorithm.system.updateInitialState( testInitialState);
                [funFvar, namaParameter] = preconditionAlgorithm.newtonAmeAlgorithm();
                figure
                subplot(3, 1, 1)
                plot(namaParameter.primalCost)
                title('primal cost')
                hold all;
                subplot(3, 1, 2)
                plot(namaParameter.dualCost)
                title('dual cost')
                hold all;
                subplot(3, 1, 3)
                plot(namaParameter.valueArgLagran)
                title('argumented Lagrangian NAMA algorithm')
                hold all;
                if(namaParameter.iterate > 1)
                    minDescentDirection = max(namaParameter.descentValue);
                end
                assert( minDescentDirection <= 0, 'lbfgs direction in NAMA is not decent');
                preconditionAlgorithm.system.updateInitialState(testInitialState);
                [funFvarFbeAlgo, globalFbeParameter] = preconditionAlgorithm.dualGlobalFbeAlgorithm();
                subplot(3, 1, 1)
                plot(globalFbeParameter.primalCost)
                subplot(3, 1, 2)
                plot(globalFbeParameter.dualCost)
                subplot(3, 1, 3)
                plot(globalFbeParameter.valueArgLagran)
                if(globalFbeParameter.iterate > 1)
                    minDescentDirection = max(globalFbeParameter.descentValue);
                end
                assert( minDescentDirection <= 0, 'lbfgs direction in FBE is not decent');
                preconditionAlgorithm.system.updateInitialState(testInitialState);
                [funFvarApgAlgo, apgParameter] = preconditionAlgorithm.dualApgAlgorithm();
                subplot(3, 1, 1)
                plot(apgParameter.primalCost)
                legend('NAMA', 'globalFbe', 'dualApg');
                subplot(3, 1, 2)
                plot(apgParameter.dualCost)
                legend('NAMA', 'globalFbe', 'dualApg');
                figure 
                semilogy(namaParameter.normFixedPointResidual);
                hold all;
                semilogy(globalFbeParameter.normFixedPointResidual);
                semilogy(apgParameter.normFixedPointResidual);
                %funFvarYalmip = testCase.yalmipImplementation();
                %yalmipCost = testCase.testSystem.calculateScenarioMpc(funFvarYalmip);
                gurobiCost = testCase.testSystem.calculateScenarioMpc(funFvarGurobi);
                namaCost = testCase.testSystem.calculateScenarioMpc(funFvar);
                apgCost = testCase.testSystem.calculateScenarioMpc(funFvarApgAlgo);
                display([namaCost - gurobiCost apgCost - gurobiCost namaCost - apgCost]);
                if(norm(funFvar.stateX - funFvarApgAlgo.stateX) > 0.05)
                    display(norm(funFvar.stateX - funFvarApgAlgo.stateX));
                    warning('mismatchApgVsNama', 'variation in the norm of the state in accelarated and NAMA algorithm')
                end
                if(norm(funFvar.stateX - funFvarGurobi.stateX) > 0.05)
                    display(norm(funFvar.stateX - funFvarGurobi.stateX));
                    warning('mismatchGurobiVsNama', 'variation in the norm of the state in accelarated and NAMA algorithm')
                end
                if(norm(funFvar.stateX - funFvarFbeAlgo.stateX) > 0.05)
                    display(norm(funFvar.stateX - funFvarFbeAlgo.stateX) );
                    warning('mismatchglobalFbeVsNama', 'variation in the norm of the state in globalFBE and NAMA algorithm')
                end
            else
                 disp('non-feasible');
            end
        end
        function testImplementationFbeAlgorithm(testCase, testInitialState)
            preconditionedSystem = copy(testCase.testSystem);
            preconditionedSystem.scaleConstraintSystem();
            preconditionedSystem.preconditionSystem();
            preconditionFbeAlgorithm = optimisationAlgorithm( preconditionedSystem );
            preconditionFbeAlgorithm = preconditionFbeAlgorithm.factorStep();
            preconditionFbeAlgorithm.system.updateInitialState(testInitialState);
            [funFvarPrecndFbeAlgo, fbePrecondParameter] = preconditionFbeAlgorithm.dualGlobalFbeAlgorithm();
            testCase.testGlobalFbeAlgorithm.system.updateInitialState(testInitialState);
            [funFvarFbeAlgo, globalFbeParameter] = testCase.testGlobalFbeAlgorithm.dualGlobalFbeAlgorithm();
            figure
            subplot(3, 1, 1)
            plot(globalFbeParameter.primalCost)
            hold all;
            plot(fbePrecondParameter.primalCost)
            title('primal cost FBE algorithm')
            subplot(3, 1, 2)
            plot(globalFbeParameter.dualCost)
            hold all;
            plot(fbePrecondParameter.dualCost)
            title('dual cost FBE algorithm')
            subplot(3, 1, 3)
            plot(globalFbeParameter.valueArgLagran)
            hold all;
            plot(fbePrecondParameter.valueArgLagran)
            title('argumented Lagrangian FBE algorithm')
            if(globalFbeParameter.iterate > 1)
                minDescentDirection = max(globalFbeParameter.descentValue);
            end
            assert( minDescentDirection <= 0, 'lbfgs direction in FBE is not decent');
        end
        function funFvar = yalmipImplementation(testCase)
            system = testCase.testSystem;
            dynamics = system.dynamics;
            tree = system.tree;
            nx = size(dynamics.matA{1}, 1);
            nu = size(dynamics.matB{1}, 2);
            numNode = length(tree.stage);
            numScen = length(tree.leaves);
            state = sdpvar(nx, numNode);
            input = sdpvar(nu, numNode - numScen);
            constraints = [];
            objective = 0;
            matLinearH = [system.constraint.matF{1} system.constraint.matG{1}];
            constraints = constraints + [state(:,1) == system.initialState]; 
            for i = 1:numNode - numScen
                objective = objective + tree.prob(i)*(state(:, i)'*system.stageCost.matQ*...
                    state(:, i) + input(:, i)'*system.stageCost.matR*input(:,i));
                constraints = constraints + [matLinearH*[state(:, i);input(:, i)] <=...
                    system.constraint.g{1}]; 
                numChild = length(tree.children{i});
                for j = 1:numChild
                    iChild = tree.children{i}(j);
                    state(:, iChild) = dynamics.matA{1}*state(:, i) +...
                        dynamics.matB{1}*input(:, i) + tree.value(iChild, :)';
                    %constraints = constraints + [state(:, iChild) == dynamics.matA{1}*state(:, i) +...
                     %   dynamics.matB{1}*input(:, i) + tree.value(iChild, :)'];
                end
            end 
            for i = 1:numScen
                j = numNode - numScen + i;
                objective = objective + tree.prob(j)*state(:, j)'*system.terminalCost.matVf{i}*...
                    state(:, j);
                constraints = constraints + [system.terminalConstraint.matFt{i}*state(:, j) <=...
                    system.terminalConstraint.gt{i}];
            end
            options = sdpsettings('solver', 'gurobi', 'verbose',  1);
            optimize(constraints, objective, options);
            funFvar.stateX = value(state);
            funFvar.inputU = value(input);
        end
    end
end

