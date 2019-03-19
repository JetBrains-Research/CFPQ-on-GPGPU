class RunStrategy:

    N_REPEAT = 30
    # N_REPEAT = 1
    THRESHOLD = 600
    STOP_REPEAT = 1

    def __init__(self, runners, tests, method='simple'):
        self.runners = runners
        self.tests = tests
        self.method = method
        if self.method == 'simple':
            self.simple_strategy()
        elif self.method == 'circle':
            self.circle_strategy()
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


    def circle_strategy(self):
        self.description = f'Run all tests for all runners and then repeat it {self.N_REPEAT} times'
        self.strategy = [
            (runner, test)
                for _ in range(self.N_REPEAT)
                    for test in self.tests
                        for runner in self.runners + ['save']
        ]
