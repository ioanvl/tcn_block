
def get_input_args(local_dict):
    ignored_values = ['self']
    return {k: v for k, v in local_dict.items() if k not in ignored_values}
