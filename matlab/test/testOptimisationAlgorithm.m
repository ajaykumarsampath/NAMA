classdef testOptimisationAlgorithm < matlab.unittest.TestCase

    properties
        testPath;
        testSystem;
        testScenarioTree;
        testApgAlgorithm;
        testNamaAlgorithm;
        testGlobalFbeAlgorithm;
    end
    
    methods(TestClassSetup)
        function addpathToTestClass(testCase)
            p = path;
            testCase.addTeardown(@path, p);
            testCase.testPath = fileparts(mfilename('fullpath'));
            addpath(fullfile(testCase.testPath, '..'));
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
        end 
    end
    
    methods
        function createScenarioTree(testCase)
            masses = 5;
            scenarioTreeParameter.nx = 2*masses;
            scenarioTreeParameter.branchingFactor = 2*ones(1, 2);
            scenarioTreeParameter.N = 10;
            testCase.testScenarioTree = scenarioTree(scenarioTreeParameter);
        end
        function createSpringMassSystem(testCase)
            testCase.testSystem = springMassSystem(testCase.testScenarioTree);
        end
    end
    
    properties( TestParameter)
        testInitialState = {2*[0.4; 0.7; 0.8; -0.5; 2.5; 1.7; -2.9; -3.6; 0.9; 1.9],...
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
            title('primal cost NAMA algorithm')
            subplot(3, 1, 2)
            plot(namaParameter.dualCost)
            title('dual cost NAMA algorithm')
            subplot(3, 1, 3)
            plot(namaParameter.valueArgLagran)
            title('argumented Lagrangian NAMA algorithm')
            if(namaParameter.iterate > 1)
                minDescentDirection = max(namaParameter.descentValue);
            end
            assert( minDescentDirection <= 0, 'lbfgs direction in NAMA is not decent');
            %{
            testCase.testGlobalFbeAlgorithm.system.updateInitialState(testInitialState);
            [funFvarFbeAlgo, globalFbeParameter] = testCase.testGlobalFbeAlgorithm.dualGlobalFbeAlgorithm();
            figure
            subplot(3, 1, 1)
            plot(globalFbeParameter.primalCost)
            title('primal cost FBE algorithm')
            subplot(3, 1, 2)
            plot(globalFbeParameter.dualCost)
            title('dual cost FBE algorithm')
            subplot(3, 1, 3)
            plot(globalFbeParameter.valueArgLagran)
            title('argumented Lagrangian FBE algorithm')
            if(globalFbeParameter.iterate > 1)
                minDescentDirection = max(globalFbeParameter.descentValue);
            end
            assert( minDescentDirection <= 0, 'lbfgs direction in FBE is not decent');
            %}
            testCase.testApgAlgorithm.system.updateInitialState(testInitialState);
            [funFvarApgAlgo, apgParameter] = testCase.testApgAlgorithm.dualApgAlgorithm();
            figure
            subplot(2, 1, 1)
            plot(apgParameter.primalCost)
            title('primal cost in dual APG')
            subplot(2, 1, 2)
            plot(apgParameter.dualCost)
            title('dual cost in dual APG')
            if(norm(funFvar.stateX - funFvarApgAlgo.stateX) > 0.05)
                warning('mismatchApgVsNama', 'variation in the norm of the state in accelarated and NAMA algorithm')
            end
            %if(norm(funFvar.stateX - funFvarFbeAlgo.stateX) > 0.05)
             %   warning('mismatchglobalFbeVsNama', 'variation in the norm of the state in globalFBE and NAMA algorithm')
            %end
        end
        function testOracleInNewtonAmeAlgorithm(testCase)
            system = testCase.testSystem;
            constraint = system.constraint;
            terminalConstraint = system.terminalConstraint;
            tree = system.tree;
            numScen = length(tree.leaves);
            numNode = length(tree.stage);
            
            lambda = 0.1;
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
        function testPreconditionSystem(testCase, testInitialState)
            preconditionedSystem = copy(testCase.testSystem);
            preconditionedSystem.scaleConstraintSystem();
            preconditionedSystem.preconditionSystem();
            preconditionApgAlgorithm = optimisationAlgorithm( preconditionedSystem );
            preconditionApgAlgorithm = preconditionApgAlgorithm.factorStep();
            preconditionApgAlgorithm.system.updateInitialState(testInitialState);
            [funFvarPrecndApgAlgo, apgPrecondParameter] = preconditionApgAlgorithm.dualApgAlgorithm();
            testCase.testApgAlgorithm.system.updateInitialState(testInitialState);
            [funFvarApgAlgo, apgParameter] = testCase.testApgAlgorithm.dualApgAlgorithm();
            figure
            subplot(2, 1, 1)
            plot(apgParameter.primalCost)
            hold all;
            plot(apgPrecondParameter.primalCost)
            title('primal cost in dual APG')
            subplot(2, 1, 2)
            plot(apgParameter.dualCost)
            hold all;
            plot(apgPrecondParameter.dualCost)
            title('dual cost in dual APG')
            if(norm(funFvarPrecndApgAlgo.stateX - funFvarApgAlgo.stateX) > 0.05)
               warning('mismatchglobalFbeVsNama', 'variation in the norm of the state in globalFBE and NAMA algorithm')
            end
        end 
    end
end

