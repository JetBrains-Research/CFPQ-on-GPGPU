import subprocess

class Runner:

    TIMEOUT = 60

    def __init__(self, path, **kwargs):
        self.path = path
        self.name = path.split('/')[-1]
        self.kwargs = kwargs

    def run(self, grammar_file, matrix_file, output_file):
        return 0

class CRunner(Runner):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.name = 'cuda_' + self.name

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
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.name = self.kwargs['name']

    def run(self, grammar_file, matrix_file, output_file):
        try:
            compl_proc = subprocess.run(
                ['python', self.path, grammar_file, matrix_file, '-o', output_file] + self.kwargs['args'],
                capture_output=True, timeout=self.TIMEOUT)
        except subprocess.TimeoutExpired:
            raise Exception('timeout')
        if compl_proc.stderr != b'':
            raise Exception(compl_proc.stderr)
        return int(compl_proc.stdout.decode().split(' ')[0])
