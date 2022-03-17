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


#%%

#section
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


#%%

'''Use equations 5a - 5c from Fechtig and Kalbitzer to calculate D for each gas'''

for gas in gas_cols:
    D = []
    count = 0
    for x in diff_df[gas + ' Cumulative Fraction']:
        F = diff_df[gas + ' Cumulative Fraction']
        t = diff_df['Time (hr)'] * 60 * 60
        
        '''For the first step where F_prior = 0'''
        if x <= 0.1 and count == 0:
            F_prior = 0
            D_first = (F[count]**2 * np.pi)/(36 * t[count])
            D.append(D_first)
            count = count + 1
        
            '''For fraction dg <= 10% (5a)'''
        elif x <= 0.1:
            F_prior = F[count - 1]
            D_step = ((F[count]**2 - F_prior**2) * np.pi)/(36 * t[count])
            D.append(D_step)
            count = count + 1
            
            ''' For fraction dg > 10% but less than 90% (5b)'''
        elif x < 0.9:
            F_prior = F[count - 1]
            D_step = 1/(np.pi**2 * t[count]) * (-1 * (np.pi**2)/3 * (F[count] - F_prior) 
                                                - 2 * np.pi * (np.sqrt(1 - np.pi/3 * F[count]) 
                                                               - np.sqrt(1 - np.pi/3 * F_prior)))
            count = count + 1
            D.append(D_step)
            
            ''' For fraction dg >= 90% (5c)'''
        else:
            F_prior = F[count - 1]
            D_step = 1/(np.pi**2 * t[count]) * np.log((1 - F_prior)/(1 - F[count]))
            print(F[count])
            count = count + 1 
            D.append(D_step)
            
            
    diff_df[gas + ' Diffusivity'] = D

#%% Make Arrhenius plots

from numpy.polynomial.polynomial import polyfit

plot_inverse_temp = diff_df['Inverse Temp (10000/K)'].astype(float)
y_int = [];
grad = [];

for gas in gas_cols:
    
    
    diff_plot = diff_df[gas + ' Diffusivity']
    log_diff_plot = np.log(diff_plot)
    
    ''' find and ignore steps with complete degassing is reached, to avoid NaNs '''
    ''' when calculating diffusivity and line of best fit '''
    
    end_idx = (diff_df[gas + ' Cumulative Fraction'] > 0.9999).idxmax()
    
    ''' find line of best fit '''
    b, m = polyfit(plot_inverse_temp[0:end_idx], log_diff_plot[0:end_idx] , 1)
    y_int.append(b)
    grad.append(m)
    
    plt.scatter(plot_inverse_temp[0:end_idx], log_diff_plot[0:end_idx])
    plt.plot(plot_inverse_temp[0:end_idx], b + m * plot_inverse_temp[0:end_idx], '-')
    
    plt.ylabel('ln D/$\mathregular{a^{2}}$')
    plt.xlabel('10000/T ($\mathregular{K^{-1}}$)')
    plt.title('Arrhenius plot - ' + gas)
    plt.show()
    
    print(gas + ' y-intercept = ' + str(b))
    print(gas + ' gradient = ' + str(m))

    

