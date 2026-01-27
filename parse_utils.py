import ast
import argparse
import inspect

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise ValueError("Boolean value expected.")
    
def dict_or_none(v):
    if v.lower() == "None":
        return None
    try:
        return ast.literal_eval(v)
    except (SyntaxError, ValueError):
        raise argparse.ArgumentTypeError("Expected a dictionary or 'None'.")
    
def none_or_float(value):
    if value.lower() == "none":
        return None
    return float(value)

def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)

def print_parsed(args):
    for key, value in vars(args).items():
        if hasattr(value, "label"):
            print(f"{key}: {value.label}")
        else:
            print(f"{key}: {value}")