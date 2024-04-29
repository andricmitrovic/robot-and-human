import numpy as np
import pandas as pd
import ast
import re


def clean(x):
    x = x.strip('[]').strip()
    x = re.sub(r'\s+', ' ', x)
    x = x.split()
    return [int(num) for num in x]


def remove_zeroes(arr):
    return [num for num in arr if num != 0.0]


def check_array_lengths(row):
    # Get lengths of all arrays in the row
    array_lengths = [len(arr) for arr in row]

    # Check if all lengths are the same
    return all(length == array_lengths[0] for length in array_lengths)


def prep_data(path):
    df = pd.read_csv(path)
    df = df[['opTaskToSave', 'timetoSave', 'mwtoSave']]
    # print(df.iloc[3].to_dict())

    df['timetoSave'] = df['timetoSave'].apply(lambda x: re.sub(r'\s+', ',', x))
    df['timetoSave'] = df['timetoSave'].apply(lambda x: np.squeeze(ast.literal_eval(x)))
    df['timetoSave'] = df['timetoSave'].apply(remove_zeroes)

    df['opTaskToSave'] = df['opTaskToSave'].apply(lambda x: clean(x))

    df['mwtoSave'] = df['mwtoSave'].apply(lambda x: re.sub(r'\s+', ',', x))
    df['mwtoSave'] = df['mwtoSave'].apply(lambda x: np.squeeze(ast.literal_eval(x)))
    df['mwtoSave'] = df['mwtoSave'].apply(remove_zeroes)

    # print(df.iloc[3].to_dict())
    # Sanity check
    result = df.apply(check_array_lengths, axis=1)
    if not result.all():
        error_message = "Arrays in the same rows have different lengths."
        raise ValueError(error_message)
    return df