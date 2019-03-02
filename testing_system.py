from argparse import ArgumentParser
from os import listdir
from os.path import join
from subprocess import run as shell_run
from tqdm import tqdm
import numpy as np

from test_utils.runners import CRunner, PythonRunner
from test_utils.run_strategy import RunStrategy

SOLUTIONS_FOLDER = 'solutions'
LOG_FILE = open('log.txt', 'wt', 1)

def testing_system(tests):
    print('Building C++ solutions...')
    # remove old folder, if exist, and create new
    commands = [
        ['rm', '-rf', SOLUTIONS_FOLDER],
        ['mkdir', SOLUTIONS_FOLDER],
        ['cmake', '-B', SOLUTIONS_FOLDER, '-DCMAKE_CONFIGURATION_TYPES=Release'],
        ['make', '-C', SOLUTIONS_FOLDER],
    ]
    for command in commands:
        comp_proc = shell_run(command, stdout=LOG_FILE, stderr=LOG_FILE)
        if comp_proc.returncode != 0:
            exit(0)
    exec = [file for file in listdir(SOLUTIONS_FOLDER) if 'make' not in file.lower()]
    print(f'Builded C++ solutions: {", ".join(exec)}')

    runners = [CRunner(join(SOLUTIONS_FOLDER, ex)) for ex in exec]
    
    print('Run checking test...')
    for runner in tqdm(runners):
        runner.run(*tests['A_star1:fullgraph_10'], 'answer.txt')

    # test_names = ['A_star1:fullgraph_10', 'A_star1:fullgraph_50', 'A_star1:fullgraph_100', 'A_star1:fullgraph_200'] 
    test_names = tests.keys()
    run_strategy = RunStrategy(runners, test_names)
    results = {
        test: {
            runner.name: [] for runner in runners
        } for test in test_names
    }
    print(f'Run {len(tests.keys())} tests...')
    print(f'Using strategy: {run_strategy.description}')
    for runner, test_name in tqdm(run_strategy.strategy):
        LOG_FILE.write(f'Run {runner.name} solution on {test_name} test\n')
        try:
            time = runner.run(*tests[test_name], 'answer.txt')
            results[test_name][runner.name].append(time)
        except Exception as e:
            LOG_FILE.write(f'failed because of {e}\n')
    print('Collect statistic and saving to result.csv...')
    with open('result.csv', 'w') as f:
        header = 'Test name'
        for runner in runners:
            header += f',{runner.name}_mean,{runner.name}_std'
        f.write(header + '\n')
        for test, measure in results.items():
            line = test
            for runner in runners:
                if len(measure[runner.name]) > 0:
                    mean_time = np.mean(measure[runner.name])
                    std_time = np.std(measure[runner.name])
                else:
                    mean_time = std_time = '-'
                line += ',{},{}'.format(mean_time, std_time)
            f.write(line + '\n')


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('tests', type=str, help='Path to csv file with description of tests')
    args = arg_parser.parse_args()

    tests = {}
    with open(args.tests, 'r') as f:
        f.readline()
        for line in f:
            values = line.split(',')
            tests[values[0]] = (values[1], values[3])
    testing_system(tests)
