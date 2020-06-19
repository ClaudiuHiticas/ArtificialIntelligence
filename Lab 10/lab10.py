class FuzzyRule:
    """
        Define a conjunctive fuzzy rule
        X and Y and ... => Z
    """

    def __init__(self, inputs, out):
        """
            Receives the set of inputs and expected output
        """
        self.out_var = out  # the name of the output variable
        self.inputs = inputs

    def evaluate(self, inputs):
        """
            Receives a dictionary of all the input values and returns the conjunction
            of their values
        """
        return [self.out_var, min(
            [inputs[descr_name][var_name]
             for descr_name, var_name in self.inputs.items()
             ])]


class FuzzySystem:
    """
        Fuzzy system object
        Receives variable descriptions and rules and outputs the defuzzified
        result of the system
    """

    def __init__(self, rules):
        self.in_descriptions = {}
        self.out_description = None
        self.rules = rules

    def add_description(self, name, descr, out=False):
        """
        Receives a description
        """
        if out:
            if self.out_description is None:
                self.out_description = descr
            else:
                raise ValueError('System already has an output')
        else:
            self.in_descriptions[name] = descr

    def compute(self, inputs):
        fuzzy_vals = self._compute_descriptions(inputs)
        rule_vals = self._compute_rules_fuzzy(fuzzy_vals)

        fuzzy_out_vars = [(list(descr[0].values())[0], descr[1]) for descr in
                          rule_vals]
        weighted_total = 0
        weight_sum = 0
        for var in fuzzy_out_vars:
            weight_sum += var[1]
            weighted_total += self.out_description.defuzzify(*var) * var[1]

        return weighted_total / weight_sum

    def _compute_descriptions(self, inputs):
        return {
            var_name: self.in_descriptions[var_name].fuzzify(inputs[var_name])
            for var_name, val in inputs.items()
        }

    def _compute_rules_fuzzy(self, fuzzy_vals):
        """
            Returns the fuzzy output of all rules
        """
        return [rule.evaluate(fuzzy_vals) for rule in self.rules
                if rule.evaluate(fuzzy_vals)[1] != 0]

class FuzzyDescriptions:
    """
        Encapsulate a description of a fuzzy variable
        It contains a set of functions for each fuzzy region
    """
    def __init__(self):
        self.regions = {}
        self.inverse = {}

    def add_region(self, var_vame, membership_function, inverse=None):
        """
            Adds a region with a given membership function, optionally
            an inverse function for the Sugeno or Tsukamoto models
        """
        self.regions[var_vame] = membership_function
        self.inverse[var_vame] = inverse

    def fuzzify(self, value):
        """
            Return the fuzzified values for each region
        """
        return {name: membership_function(value)
                for name, membership_function in self.regions.items()
                }

    def defuzzify(self, var_name, value):
        return self.inverse[var_name](value)

class Controller:
    def __init__(self, temperature, humidity, time, rules):
        self.system = FuzzySystem(rules)
        self.system.add_description('temperature', temperature)
        self.system.add_description('humidity', humidity)
        self.system.add_description('time', time, out=True)

    def compute(self, inputs):
        return "If we have the humidity: " + str(inputs['humidity']) + \
               " and temperature: " + str(inputs['temperature']) + \
               " will probably have the operating time: " + str(self.system.compute(inputs))



def trap_region(a, b, c, d):
    """
        Returns a higher order function for a trapezoidal fuzzy region
    """
    return lambda x: max(0, min((x - a) / (b - a), 1, (d - x) / (d - c)))


def tri_region(a, b, c):
    """
        Returns a higher order function for a triangular fuzzy region
    """
    return trap_region(a, b, b, c)


def inverse_line(a, b):
    return lambda val: val * (b - a) + a


def inverse_tri(a, b, c):
    return lambda val: (inverse_line(a, b)(val) + inverse_line(c, b)(val)) / 2


if __name__ == '__main__':
    temperature = FuzzyDescriptions()
    humidity = FuzzyDescriptions()
    time = FuzzyDescriptions()
    rules = []

    temperature.add_region('very cold', trap_region(-1000, -30, -20, 5))
    temperature.add_region('cold', tri_region(-5, 0, 10))
    temperature.add_region('normal', trap_region(5, 10, 15, 20))
    temperature.add_region('warm', tri_region(15, 20, 25))
    temperature.add_region('hot', trap_region(25, 30, 35, 1000))

    humidity.add_region('dry', tri_region(-1000, 0, 50))
    humidity.add_region('normal', tri_region(0, 50, 100))
    humidity.add_region('wet', tri_region(50, 100, 1000))

    time.add_region('short', tri_region(-1000, 0, 50), inverse_line(50, 0))
    time.add_region('medium', tri_region(0, 50, 100), inverse_tri(0, 50, 100))
    time.add_region('long', tri_region(50, 100, 1000), inverse_line(50, 100))

    rules.append(FuzzyRule({'temperature': 'very cold', 'humidity': 'wet'},
                           {'time': 'short'}))
    rules.append(FuzzyRule({'temperature': 'cold', 'humidity': 'wet'},
                           {'time': 'short'}))
    rules.append(FuzzyRule({'temperature': 'normal', 'humidity': 'wet'},
                           {'time': 'short'}))
    rules.append(FuzzyRule({'temperature': 'warm', 'humidity': 'wet'},
                           {'time': 'short'}))
    rules.append(FuzzyRule({'temperature': 'hot', 'humidity': 'wet'},
                           {'time': 'medium'}))

    rules.append(FuzzyRule({'temperature': 'very cold', 'humidity': 'normal'},
                           {'time': 'short'}))
    rules.append(FuzzyRule({'temperature': 'cold', 'humidity': 'normal'},
                           {'time': 'medium'}))
    rules.append(FuzzyRule({'temperature': 'normal', 'humidity': 'normal'},
                           {'time': 'medium'}))
    rules.append(FuzzyRule({'temperature': 'warm', 'humidity': 'normal'},
                           {'time': 'medium'}))
    rules.append(FuzzyRule({'temperature': 'hot', 'humidity': 'normal'},
                           {'time': 'long'}))

    rules.append(FuzzyRule({'temperature': 'very cold', 'humidity': 'dry'},
                           {'time': 'medium'}))
    rules.append(FuzzyRule({'temperature': 'cold', 'humidity': 'dry'},
                           {'time': 'long'}))
    rules.append(FuzzyRule({'temperature': 'normal', 'humidity': 'dry'},
                           {'time': 'long'}))
    rules.append(FuzzyRule({'temperature': 'warm', 'humidity': 'dry'},
                           {'time': 'long'}))
    rules.append(FuzzyRule({'temperature': 'hot', 'humidity': 'dry'},
                           {'time': 'long'}))

    ctrl = Controller(temperature, humidity, time, rules)

    print(ctrl.compute({'humidity': 65, 'temperature': 17}))
    print(ctrl.compute({'humidity': 10, 'temperature': 17}))
    print(ctrl.compute({'humidity': 10, 'temperature': 30}))
    print(ctrl.compute({'humidity': 75, 'temperature': 20}))