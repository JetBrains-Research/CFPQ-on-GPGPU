import subprocess

class Runner:

    TIMEOUT = 1800
    # TIMEOUT = 600

    def __init__(self, path, **kwargs):
        self.path = path
        self.name = path.split('/')[-1]
        self.kwargs = kwargs

    def run(self, grammar_file, matrix_file, output_file):
        return 0


class CRunner(Runner):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.name = 'c++_' + self.name

    def run(self, grammar_file, matrix_file, output_file):
        try:
            compl_proc = subprocess.run([self.path, grammar_file, matrix_file, output_file],
                                    capture_output=True, timeout=self.TIMEOUT)
            if compl_proc.stderr != b'':
                raise Exception(compl_proc.stderr)
            return [int(i) / 1000 for i in compl_proc.stdout.decode().split(' ')]
        except subprocess.TimeoutExpired:
            raise Exception('timeout')


class PythonRunner(Runner):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.name = self.kwargs['name']

    def run(self, grammar_file, matrix_file, output_file):
        try:
            compl_proc = subprocess.run(
                ['python', self.path, grammar_file, matrix_file, '-o', output_file] + self.kwargs['args'],
                capture_output=True, timeout=self.TIMEOUT)
            if compl_proc.stderr != b'':
                raise Exception(compl_proc.stderr)
            return [int(i) / 1000 for i in compl_proc.stdout.decode().split(' ')][-1::-1]
        except subprocess.TimeoutExpired:
            raise Exception('timeout')

class MonoRunner(Runner):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.name = 'FSharp'
    
    def run(self, grammar_file, matrix_file, output_file):
        try:
            compl_proc = subprocess.run(
                ['mono', self.path, grammar_file, matrix_file, '1'],
                capture_output=True, timeout=self.TIMEOUT)
            if compl_proc.stderr != b'':
                raise Exception(compl_proc.stderr)
            time = float(compl_proc.stdout.decode().split(', ')[1])
            return (time / 1000, -1)
        except subprocess.TimeoutExpired:
            raise Exception('timeout')
