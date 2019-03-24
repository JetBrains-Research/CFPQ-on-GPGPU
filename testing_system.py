from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import join, exists
from subprocess import run as shell_run
from tqdm import tqdm
import numpy as np

from test_utils.runners import CRunner, PythonRunner, MonoRunner
from test_utils.run_strategy import RunStrategy

SOLUTIONS_FOLDER = 'solutions'
REQUIREMENTS_PATH = 'python/requirements.txt'
LOG_FILE = open('log.txt', 'wt', 1)
BUILD_LOG_FILE = open('build_log.txt', 'wt', 1)
CROSS_VAL_FOLDER = 'cross_val'

def collect_statistic(runners, results):
    with open('result.csv', 'w') as f:
        header = 'Test name'
        for runner in runners:
            header += f',{runner.name}_time_all_mean,{runner.name}_time_all_std' + \
                      f',{runner.name}_time_mult_mean,{runner.name}_time_mult_std' + \
                      f',{runner.name}_axiom_hash,{runner.name}_axiom_count'
        f.write(header + '\n')
        for test, measure in results.items():
            line = test
            for runner in runners:
                if len(measure[runner.name]) > 0:
                    times_all, times_mult = list(zip(*measure[runner.name]))[:2]
                    mean_all_time = np.mean(times_all)
                    std_all_time = np.std(times_all)
                    mean_mult_time = np.mean(times_mult)
                    std_mult_time = np.std(times_mult)
                    hash = measure[runner.name][0][2]
                    count = measure[runner.name][0][3]
                else:
                    mean_all_time = std_all_time = mean_mult_time = std_mult_time = hash = count = '-'
                line += f',{mean_all_time},{std_all_time},{mean_mult_time},{std_mult_time},{hash},{count}'
            f.write(line + '\n')

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
        comp_proc = shell_run(command, stdout=BUILD_LOG_FILE, stderr=BUILD_LOG_FILE)
        if comp_proc.returncode != 0:
            exit(0)
    exec = [file for file in listdir(SOLUTIONS_FOLDER) if 'make' not in file.lower()]
    print(f'Builded C++ solutions: {", ".join(exec)}')

    print('Install python requirements...')
    comp_proc = shell_run(
        ['test_utils/install_requirements.sh', REQUIREMENTS_PATH],
        stdout=BUILD_LOG_FILE, stderr=BUILD_LOG_FILE
    )
    if comp_proc.returncode != 0:
        exit(0)

    runners = []
    runners += [CRunner(join(SOLUTIONS_FOLDER, ex)) for ex in exec]
    runners += [
        PythonRunner('python/main.py', **{'name': 'python_GPU_uint32', 'args': ['-t=uint32']}),
        PythonRunner('python/main.py', **{'name': 'python_GPU_uint8', 'args': ['-t=uint8']}),
        PythonRunner('python/main.py', **{'name': 'python_CPU_sparse', 'args': ['-t=sparse', '--on_cpu']})
    ]
    runners += [
        MonoRunner('FSharp/CFPQ_Matrix_Performance.exe')
    ]
    print(f'All runners: {", ".join(map(lambda r: r.name, runners))}')


    checking_test = ['A_star1:fullgraph_10', 'A_star2:fullgraph_10', 'GPPerf1_cnf:skos',
                     'grammar:out_0', 'SG:G5k']
    print(f'Cross validation on tests {", ".join(checking_test)}')
    LOG_FILE.write('========== Cross validation ==========\n')
    if not exists(CROSS_VAL_FOLDER):
        mkdir(CROSS_VAL_FOLDER)
    for check_test in tqdm(checking_test):
        compare = None
        for runner in runners:
            if isinstance(runner, MonoRunner):
                continue
            answer_file = join(f'{CROSS_VAL_FOLDER}', f'{check_test}_{runner.name}.txt')
            try:
                LOG_FILE.write(f'{runner.name} work on {check_test} test\n')
                runner.run(*tests[check_test], answer_file)
            except Exception:
                LOG_FILE.write(f'Can\'t validate because of timeout\n')
                continue
            cur_res = {}
            with open(answer_file, 'r') as f:
                for line in f:
                    nonterm, hash, count = line.split(' ')
                    cur_res[nonterm] = (hash, count[:-1])
            if compare is None:
                compare = cur_res
            else:
                if not compare == cur_res:
                    print(f'{runner.name} and {runners[0].name} have different answers on test {check_test}')
                    exit(0)

    test_names = tests.keys()
    run_strategy = RunStrategy(runners, test_names, 'circle')
    results = {
        test: {
            runner.name: [] for runner in runners
        } for test in test_names
    }
    collect_statistic(runners, results)

    print(f'Run {len(test_names)} tests...')
    print(f'Using strategy: {run_strategy.description}')
    LOG_FILE.write('========== Testing  ==========\n')
    info = {}
    for runner, test_name in tqdm(run_strategy.strategy):
        if isinstance(runner, str) and runner == 'save':
            collect_statistic(runners, results)
            continue

        if info.get((runner, test_name), '') == 'failed':
            LOG_FILE.write(f'{runner.name} skip test {test_name} because of previos failure\n')
            continue

        if test_name in results and runner.name in results[test_name]:
            if len(results[test_name][runner.name]) >= run_strategy.STOP_REPEAT:
                times_all, times_mult = list(zip(*results[test_name][runner.name]))[:2]
                if np.mean(times_all) > run_strategy.THRESHOLD:
                    LOG_FILE.write(f'{runner.name} skip test {test_name} because it\' too long...\n')
                    continue

        LOG_FILE.write(f'{runner.name} work on {test_name} test\n')
        try:
            time_all, time_mult = runner.run(*tests[test_name], 'answer.txt')
            if isinstance(runner, MonoRunner):
                results[test_name][runner.name].append((time_all, time_mult, '-', '-'))
            else:
                with open('answer.txt', 'r') as f:
                    for line in f:
                        nonterm, hash, count = line.split(' ')
                        if nonterm == 's':
                            break
                results[test_name][runner.name].append((time_all, time_mult, hash, count[:-1]))
            LOG_FILE.write(f'done in {time_all} seconds\n')
        except Exception as e:
            LOG_FILE.write(f'failed because of {e}\n')
            info[(runner, test_name)] = 'failed'

    print('Collect statistic and saving to result.csv...')
    collect_statistic(runners, results)


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
