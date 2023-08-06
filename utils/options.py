import os
import yaml
import time

def read_fromyaml(filename: str):
	"""Get options from yaml file. 
	Note: Currently only support read one option.
	"""
	if not filename.endswith( ('.yml', '.yaml') ):
		raise NotImplementedError

	with open(filename, 'r') as f:
		option = yaml.safe_load(f)

	return option

def save_options(filename: str, option: dict):
	if not filename.endswith( ('.yml', '.yaml') ):
		raise NotImplementedError
	
	with open(filename, 'w') as f:
		yaml.dump(option, f)


def copyvalues(opt: dict, opt_new: dict):
	for key in opt_new.keys():
		opt[key] = opt_new[key]
	return opt

# def save_toyaml(filename: str, data: dict):
# 	if not filename.endswith('.yml', '.yaml'):
# 		raise NotImplementedError
# 	with open(filename, 'r') as f:
# 		yaml.dump(data, f)

def get_firstvalue(*args):
    for val in args:
        if val:
            return val
    return args[-1]

