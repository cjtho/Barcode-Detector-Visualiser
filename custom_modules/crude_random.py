class RandomNumberGenerator:
    """Don't mind me, just implementing a crude `random` module. Nothing to see here."""

    def __init__(self, seed=0):
        self.seed = seed

    def _generate_next_seed(self):
        # LCG method
        # params seen in "Numerical Recipes" by
        # William H. Press, Saul A. Teukolsky, William T. Vetterling and Brian P. Flannery
        a = 1664525
        c = 1013904223
        m = 2 ** 32
        self.seed = (a * self.seed + c) % m

    def randint(self, low_bound, upper_bound):  # inclusive
        self._generate_next_seed()
        return (self.seed % (upper_bound - low_bound + 1)) + low_bound