import os
import sys
import json
import pandas as pd
import numpy as np


def main():
    # input
    csv_path = ""
    # output
    json_path = ""

    dict_pf_species = dict()
    for file in os.listdir(csv_path):
        species_name = file.split('_')[0]
        full_path = csv_path + file
        data = pd.read_csv(full_path, sep='\t', header=0, usecols=[1])
        for pf_value_list in data.values[0::, 0::]:
            for pf_value in str(pf_value_list).split('\'')[1].split(';'):
                dict_pf_species.setdefault(pf_value.split(',')[0], list()).append(species_name)
    for key in dict_pf_species.keys():
        dict_pf_species[key] = list(set(dict_pf_species[key]))
    with open(json_path, 'w') as fin:
        fin.write(json.dumps(dict_pf_species))


if __name__ == '__main__':
    main()
