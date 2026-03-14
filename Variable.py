import itertools

class Factor:
    def __init__(self, variables, values):
        self.variables = variables
        self.values = values

def multiply(f1, f2):
    vars_new = list(set(f1.variables + f2.variables))
    values = {}

    for assignment in itertools.product([0,1], repeat=len(vars_new)):
        assign_dict = dict(zip(vars_new, assignment))

        key1 = tuple(assign_dict[v] for v in f1.variables)
        key2 = tuple(assign_dict[v] for v in f2.variables)

        if key1 in f1.values and key2 in f2.values:
            values[assignment] = f1.values[key1] * f2.values[key2]

    return Factor(vars_new, values)

def sum_out(variable, factor):
    idx = factor.variables.index(variable)
    new_vars = factor.variables[:idx] + factor.variables[idx+1:]
    new_values = {}

    for assignment, val in factor.values.items():
        new_key = assignment[:idx] + assignment[idx+1:]
        new_values[new_key] = new_values.get(new_key,0) + val

    return Factor(new_vars, new_values)

def variable_elimination(factors, query, hidden_vars):

    for var in hidden_vars:
        related = [f for f in factors if var in f.variables]
        factors = [f for f in factors if var not in f.variables]

        product = related[0]
        for f in related[1:]:
            product = multiply(product, f)

        new_factor = sum_out(var, product)
        factors.append(new_factor)

    result = factors[0]
    for f in factors[1:]:
        result = multiply(result, f)

    total = sum(result.values.values())
    result.values = {k:v/total for k,v in result.values.items()}

    return result
