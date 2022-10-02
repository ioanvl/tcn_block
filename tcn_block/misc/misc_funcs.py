def get_input_args(local_dict):
    ignored_values = ['self']
    return {k: v for k, v in local_dict.items() if k not in ignored_values}


def makeGrid(input_dict):
    from itertools import product
    keys = input_dict.keys()
    combinations = product(*input_dict.values())
    grid = [dict(zip(keys, cc)) for cc in combinations]
    return grid


def gridGenerator(input_dict):
    grid = makeGrid(input_dict)
    for sample in grid:
        yield sample


def pick_dict_values(input_dict, vals):
    return {x: input_dict[x] for x in input_dict.keys() if x in vals}
