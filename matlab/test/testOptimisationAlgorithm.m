classdef testOptimisationAlgorithm < matlab.unittest.TestCase

    properties
        testPath;
        testSystem;
        testScenarioTree;
        testApgAlgorithm;
        testNamaAlgorithm;
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
        end 
    end
    
    methods
        function createScenarioTree(testCase)
            masses = 5;
            scenarioTreeParameter.nx = 2*masses;
            scenarioTreeParameter.branchingFactor = 2*ones(1, 2);
            scenarioTreeParameter.N = 11;
            testCase.testScenarioTree = scenarioTree(scenarioTreeParameter);
        end
        function createSpringMassSystem(testCase)
            testCase.testSystem = springMassSystem(testCase.testScenarioTree);
        end
    end
    
    properties( TestParameter)
        testInitialState = {2*[0.4; 0.7; 0.8; -0.5; 2.5; 1.7; -2.9; -3.6; 0.9; 1.9]};
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
                dualVar.yt{i, 1} = zeros( size(terminalConstraint.matFt{i}, 1), 1);
            end
            initialState = ones(10, 1);
            testCase.testNamaAlgorithm.system.updateInitialState(initialState);
            [funFvar, solveStepParameter] = testCase.testNamaAlgorithm.solveStep(dualVar);
        end
        function testImplementNamaAlgorithm(testCase, testInitialState)
            testCase.testNamaAlgorithm.system.updateInitialState( testInitialState);
            [funFvar, namaParameter] = testCase.testNamaAlgorithm.newtonAmeAlgorithm();
            testCase.testApgAlgorithm.system.updateInitialState(testInitialState);
            [funFvarApgAlgo, apgParameter] = testCase.testApgAlgorithm.dualApgAlgorithm();
            figure
            subplot(2, 1, 1)
            plot(apgParameter.primalCost)
            subplot(2, 1, 2)
            plot(apgParameter.dualCost)
            norm(funFvar.stateX - funFvarApgAlgo.stateX)
        end
    end
end

