import subprocess

class Runner:

    TIMEOUT = 30

    def __init__(self, path):
        self.path = path
        self.name = path.split('/')[-1]

    def run(self, grammar_file, matrix_file, output_file):
        return 0

class CRunner(Runner):
    def run(self, grammar_file, matrix_file, output_file):
        try:
            compl_proc = subprocess.run([self.path, grammar_file, matrix_file, output_file],
                                    capture_output=True, timeout=self.TIMEOUT)
        except subprocess.TimeoutExpired:
            raise Exception('timeout')
        if compl_proc.stderr != b'':
            raise Exception(compl_proc.stderr)
        return int(compl_proc.stdout)

class PythonRunner(Runner):
    def additional_args(self, **kwargs):
        pass

    def run(self, grammar_file, matrix_file, output_file):
        pass
