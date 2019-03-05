from argparse import ArgumentParser
from os import listdir
from os.path import join
from subprocess import run as shell_run
from tqdm import tqdm
import numpy as np

from test_utils.runners import CRunner, PythonRunner
from test_utils.run_strategy import RunStrategy

SOLUTIONS_FOLDER = 'solutions'
REQUIREMENTS_PATH = 'python/requirements.txt'
LOG_FILE = open('log.txt', 'wt', 1)

def testing_system(tests):
    print('Building C++ solutions...')
    # remove old folder, if exist, and create new
    commands = [
        ['rm', '-rf', SOLUTIONS_FOLDER],
        ['mkdir', SOLUTIONS_FOLDER],
        ['cmake', f'-B{SOLUTIONS_FOLDER}', '-H.', '-DCMAKE_CONFIGURATION_TYPES=Release'],
        ['make', '-C', SOLUTIONS_FOLDER],
    ]
    for command in commands:
        comp_proc = shell_run(command, stdout=LOG_FILE, stderr=LOG_FILE)
        if comp_proc.returncode != 0:
            exit(0)
    exec = [file for file in listdir(SOLUTIONS_FOLDER) if 'make' not in file.lower()]
    print(f'Builded C++ solutions: {", ".join(exec)}')

    print('Install python requirements...')
    comp_proc = shell_run(
        ['test_utils/install_requirements.sh', REQUIREMENTS_PATH],
        stdout=LOG_FILE, stderr=LOG_FILE
    )
    if comp_proc.returncode != 0:
        exit(0)

    runners = [CRunner(join(SOLUTIONS_FOLDER, ex)) for ex in exec]
    runners += [
        PythonRunner('python/main.py', **{'name': 'python_GPU_uint32', 'args': ['-t=uint32']}),
        PythonRunner('python/main.py', **{'name': 'python_GPU_uint8', 'args': ['-t=uint8']}),
        PythonRunner('python/main.py', **{'name': 'python_CPU_sparse', 'args': ['-t=sparse', '--on_cpu']})
    ]
    print(f'All runners: {", ".join(map(lambda r: r.name, runners))}')


    checking_test = ['A_star1:fullgraph_10', 'A_star1:fullgraph_50',
                     'A_star2:fullgraph_10', 'A_star2:fullgraph_50']
    print(f'Cross validation on tests {", ".join(checking_test)}')
    for check_test in tqdm(checking_test):
        compare = None
        for runner in runners:
            runner.run(*tests[check_test], f'answer.txt')
            cur_res = {}
            with open('answer.txt', 'r') as f:
                for line in f:
                    nonterm, hash, count = line.split(' ')
                    cur_res[nonterm] = (hash, count)
            if compare is None:
                compare = cur_res
            elif not compare == cur_res:
                print(f'{runner.name} and {runners[0].name} have differents answer on test {check_test}')
                exit(0)

    test_names = tests.keys()
    run_strategy = RunStrategy(runners, test_names)
    results = {
        test: {
            runner.name: [] for runner in runners
        } for test in test_names
    }

    print(f'Run {len(tests.keys())} tests...')
    print(f'Using strategy: {run_strategy.description}')
    info = {}
    for runner, test_name in tqdm(run_strategy.strategy):
        if info.get((runner, test_name), '') == 'failed':
            LOG_FILE.write(f'{runner.name} skip test {test_name} because of previos failure\n')
            continue
        if test_name in results and runner.name in results[test_name]:
            if len(results[test_name][runner.name]) >= run_strategy.STOP_REPEAT and\
                np.mean(results[test_name][runner.name]) > run_strategy.THRESHOLD:
                LOG_FILE.write(f'{runner.name} skip test {test_name} because it\' too long...\n')
                continue

        LOG_FILE.write(f'{runner.name} work on {test_name} test\n')
        try:
            time = runner.run(*tests[test_name], 'answer.txt')
            results[test_name][runner.name].append(time)
            LOG_FILE.write(f'done in {time} seconds\n')
        except Exception as e:
            LOG_FILE.write(f'failed because of {e}\n')
            info[(runner, test_name)] = 'failed'

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
