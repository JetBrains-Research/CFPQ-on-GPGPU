class RunStrategy:

    N_REPEAT = 3

    def __init__(self, runners, tests, method='simple'):
        self.runners = runners
        self.tests = tests
        self.method = method
        if self.method == 'simple':
            self.simple_strategy()
        else:
            print('Provide correct strategy for build order of testing')
            self.description = 'No such method'
            self.strategy = []
            

    def simple_strategy(self):
        self.description = 'Each runner execute {} times'.format(self.N_REPEAT)
        self.strategy = [
            (runner, test)
                for runner in self.runners
                    for test in self.tests
                        for _ in range(self.N_REPEAT)
        ]
