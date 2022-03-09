import numpy as np
import pandas as pd
import pint
import pint_pandas
import regex
import sys
from matplotlib import pyplot as plt


class MissingColumnError(Exception):
    """Something can't be found in the input file."""


diff_df = pd.read_csv('diff_test.csv')

df_cols = pd.Series(diff_df.columns)

temp_pattern = ['temp', 'T ']
esc_lst = [regex.escape(s) for s in temp_pattern]
temp_pattern = '|'.join(esc_lst)

temp_cols = [col for col in diff_df.columns if regex.search(temp_pattern, col, regex.IGNORECASE)]

try:
    temp_cols[0]
except IndexError:
    raise MissingColumnError("No temperature columns found")

# if len(temp_cols) > 1:
'''It will be very common to have multiple temperature columns, either because someone did a C to K 
conversion in Excel, because they calculated inverse temperatures, or because the data file has other
process values recorded in it (like room temperature, for example). This block attempts to find the
best one. This can all be done more efficiently with some functions or maybe just a fancier
application of regular expressions later.'''
for col in temp_cols:
    temp_K_pattern = regex.escape("(K)")
    temp_K_cols = [col for col in diff_df.columns if regex.search(temp_K_pattern, col, regex.IGNORECASE)]
    if len(temp_K_cols) == 0:
        '''If we find no Kelvin column, we search for Celsius'''
        temp_C_pattern = regex.escape("(C)")
        temp_C_cols = [col for col in diff_df.columns if regex.search(temp_C_pattern, col, regex.IGNORECASE)]
        if len(temp_C_cols) == 0:
            '''If we find no Celsius column, we search for Fahrenheit'''
            temp_F_pattern = regex.escape("(F)")
            temp_F_cols = [col for col in diff_df.columns if regex.search(temp_F_pattern, col, regex.IGNORECASE)]
            if len(temp_F_cols) == 0:
                '''If we find no recognized units, we find the best temp match with no units and assume it's C'''
                temp_unitless_full_pattern = "(?b)" + regex.escape("Temp")
                print(temp_unitless_full_pattern)
                temp_unitless_cols_best = [col for col in temp_cols if
                                    regex.search(temp_unitless_full_pattern, col, regex.IGNORECASE)]
                temp_col_pick = temp_unitless_cols_best[0]
                diff_df[temp_col_pick] = diff_df[temp_col_pick].astype("pint[degC]")
            elif len(temp_F_cols) == 1:
                temp_col_pick = temp_F_cols[0]
                diff_df[temp_col_pick] = diff_df[temp_col_pick].astype("pint[degF]")
            else:
                temp_F_full_pattern = "(?b)" + regex.escape("Temp ") + temp_F_pattern
                temp_F_cols_best = [col for col in temp_F_cols if
                                    regex.search(temp_F_full_pattern, col, regex.IGNORECASE)]
                temp_col_pick = temp_F_cols_best[0]
                diff_df[temp_col_pick] = diff_df[temp_col_pick].astype("pint[degF]")
        elif len(temp_C_cols) == 1:
            temp_col_pick = temp_C_cols[0]
            diff_df[temp_col_pick] = diff_df[temp_col_pick].astype("pint[degC]")
        else:
            temp_C_full_pattern = "(?b)" + regex.escape("Temp ") + temp_C_pattern
            temp_C_cols_best = [col for col in temp_C_cols if regex.search(temp_C_full_pattern, col, regex.IGNORECASE)]
            temp_col_pick = temp_C_cols_best[0]
            diff_df[temp_col_pick] = diff_df[temp_col_pick].astype("pint[degC]")
    elif len(temp_K_cols) == 1:
        temp_col_pick = temp_K_cols[0]
        diff_df[temp_col_pick] = diff_df[temp_col_pick].astype("pint[kelvin]")
    else:
        temp_K_full_pattern = "(?b)" + regex.escape("Temp ") + temp_K_pattern
        temp_K_cols_best = [col for col in temp_K_cols if regex.search(temp_K_full_pattern, col, regex.IGNORECASE)]
        temp_col_pick = temp_K_cols_best[0]
        diff_df[temp_col_pick] = diff_df[temp_col_pick].astype("pint[kelvin]")
# else:
#     temp_col_pick = temp_cols[0]


diff_df["Inverse Temp (10000/K)"] = 10000/diff_df[temp_col_pick].pint.to("kelvin")

gas_pattern = ["He", "Ne", "Ar", "Kr", "Xe", "3", "4", "20", "21", "22", "36", "37", "38", "39", "40"]
esc_lst = [regex.escape(s) for s in gas_pattern]
gas_pattern = '|'.join(esc_lst)

gas_cols = [col for col in diff_df.columns if regex.search(gas_pattern, col, regex.IGNORECASE)]

for gas in gas_cols:
    '''In the case of the gases, we just use all that we find. Maybe we can add a clever way to select which ones are
    displayed once we make this into a gui.'''
    diff_df[gas + " Fraction"] = diff_df[gas]/sum(diff_df[gas])
    diff_df[gas + " Cumulative Fraction"] = diff_df[gas + " Fraction"].cumsum()



