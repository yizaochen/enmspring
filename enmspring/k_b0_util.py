import copy
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.family'] = 'Arial' # Set font family to Arial for plot

category_list = ['PP', 'st', 'PB', 'R', 'RB', 'bp']
category_dict = {'PP': ['PP0', 'PP1', 'PP2', 'PP3'],
                 'st': ['st'],
                 'PB': ['PB', 'RR'],
                 'R': ['R0', 'R1', 'R2'],
                 'RB': ['RB0', 'RB1', 'RB2', 'RB3'],
                 'bp': ['hb', 'bp1', 'bp2']
                }
subcategory_reverse = {'PP0': 'PP', 'PP1': 'PP', 'PP2': 'PP', 'PP3': 'PP',
                       'PB': 'PB', 'RR': 'PB',
                       'R0': 'R', 'R1': 'R', 'R2': 'R',
                       'RB0': 'RB', 'RB1': 'RB', 'RB2': 'RB', 'RB3': 'RB',
                       'st': 'st',
                       'hb': 'bp', 'bp1': 'bp', 'bp2': 'bp'}
color_dict = {'PP': {'PP0': 'red', 'PP1': 'green', 'PP2': 'blue', 'PP3': 'magenta'},
              'st': {'st': 'red'},
              'PB': {'PB': 'orange', 'RR': 'cyan'},
              'R':  {'R0': 'red', 'R1': 'green', 'R2': 'violet'},
              'RB': {'RB0': 'blue', 'RB1': 'orange', 'RB2': 'cyan', 'RB3': 'magenta'},
              'bp': {'hb': 'red', 'bp1': 'green', 'bp2': 'blue'}
             }
abbr_dict = {'arna+arna': 'dsRNA', 'bdna+bdna': 'dsDNA'}


### Functions for some general purposes
###
def get_central_bps_df(df):
    clean_criteria = 1e-3
    start_resid = 4
    end_resid = 18
    mask = (df['k'] > clean_criteria)
    df_1 = df[mask]
    mask = df_1['Resid_i'].between(start_resid, end_resid)
    df_2 = df_1[mask]
    mask = df_2['Resid_j'].between(start_resid, end_resid)
    return df_2[mask]


### Functions for number of bonds
###    
def make_n_bonds_for_category(df0, category, n_bp):
    d_result = {subcategory: list() for subcategory in category_dict[category]}
    for subcategory in category_dict[category]:
        df_temp = filter_mapping[category](df0, subcategory)
        n_bond = df_temp.shape[0]
        d_result[subcategory].append(n_bond)
        d_result[subcategory].append(n_bond/n_bp)
    df = pd.DataFrame(d_result)
    index_map = {0: 'total_nbonds', 1: 'nbonds_per_bp'}
    df = df.rename(index=index_map)
    return df[category_dict[category]]

### Functions which getting k-b0 dictionaries for different interactions
###
def get_k_b0_mean_std_by_category(df0, category):
    d_result_0 = dict()
    for subcategory in category_dict[category]:
        df_temp = filter_mapping[category](df0, subcategory)
        d_result_0 = _get_d_result(d_result_0, subcategory, df_temp)
    return d_result_0

### Fucntions for generating df after filter for four interactions
###
def get_df_by_filter_PP(df0, category):
    PP01_b0 = 2.
    PP12_b0 = 2.6
    PP23_b0 = 3.9
    mask = (df0['PairType'] == 'same-P-P-0') | (df0['PairType'] == 'same-P-P-1') | (df0['PairType'] == 'same-P-S-0') | (df0['PairType'] == 'same-P-S-1')
    df1 = df0[mask]
    if category == 'PP0':
        mask = df1['b0'] <= PP01_b0
    elif category == 'PP1':
        mask = (df1['b0'] > PP01_b0) & (df1['b0'] <= PP12_b0)
    elif category == 'PP2':
        mask = (df1['b0'] > PP12_b0) & (df1['b0'] <= PP23_b0)
    elif category == 'PP3':
        mask = df1['b0'] > PP23_b0
    return df1[mask]


def get_df_by_filter_st(df0, category):
    if category == 'st':
        mask = (df0['PairType'] == 'STACK-1')
    return df0[mask] 


def get_df_by_filter_PB(df0, category):
    if category == 'PB':
        mask = (df0['PairType'] == 'same-P-B-0') | (df0['PairType'] == 'same-P-B-1') | (df0['PairType'] == 'same-P-B-2') | (df0['PairType'] == 'same-S-B-1')
    elif category == 'RR':
        mask = (df0['PairType'] == 'same-S-S-1') | (df0['PairType'] == 'same-S-S-2')
    return df0[mask]


def get_df_by_filter_R(df0, category):
    R01_b0 = 2. 
    R12_b0 = 2.7
    mask = (df0['PairType'] == 'same-S-S-0') 
    df1 = df0[mask]
    if category == 'R0':
        mask = df1['b0'] <= R01_b0
    elif category == 'R1':
        mask = (df1['b0'] > R01_b0) & (df1['b0'] <= R12_b0)
    elif category == 'R2':
        mask = df1['b0'] > R12_b0
    return df1[mask]


def get_df_by_filter_RB(df0, category):
    RB01_b0 = 2. 
    RB12_b0 = 2.6 
    RB23_b0 = 4.0 
    mask = (df0['PairType'] == 'same-S-B-0') 
    df1 = df0[mask]
    if category == 'RB0':
        mask = df1['b0'] <= RB01_b0
    elif category == 'RB1':
        mask = (df1['b0'] > RB01_b0) & (df1['b0'] <= RB12_b0)
    elif category == 'RB2':
        mask = (df1['b0'] > RB12_b0) & (df1['b0'] <= RB23_b0)
    elif category == 'RB3':
        mask = df1['b0'] > RB23_b0
    return df1[mask]


def get_df_by_filter_bp(df0, category):
    hb_b0 = 3.3   # The bond length which separate real HB and pseudo HB
    if category == 'hb':
        mask = (df0['PairType'] == 'HB-0') & (df0['b0'] <= hb_b0)
    elif category == 'bp1':
        mask = ((df0['PairType'] == 'HB-0') | (df0['PairType'] == 'Oppo-Ring-0')) & (df0['b0'] > hb_b0)
    elif category == 'bp2':
        mask = (df0['PairType'] == 'HB-1') | (df0['PairType'] == 'Oppo-Ring-1')
    return df0[mask]

    
filter_mapping = {'PP': get_df_by_filter_PP, 'st': get_df_by_filter_st, 'PB': get_df_by_filter_PB,
                  'R': get_df_by_filter_R, 'RB': get_df_by_filter_RB, 'bp': get_df_by_filter_bp}
### Private Functions used in this module
###
def _get_d_result(d_result_0, category, df_temp):
    klist = df_temp['k'].values
    b0list = df_temp['b0'].values
    if len(klist) == 0:
        d_result_0[category] = {'k': klist, 'b0': b0list, 'mean': None, 'std': None, 'nbond': len(klist)}
    else:
        d_result_0[category] = {'k': klist, 'b0': b0list, 'mean': klist.mean(), 'std': klist.std(), 'nbond': len(klist)}
    return d_result_0


def _get_mean_std_string(k_b0_dict, subcategory):
    kmean = k_b0_dict[subcategory]['mean']
    kstd = k_b0_dict[subcategory]['std']
    nbond = k_b0_dict[subcategory]['nbond']
    kmean_str = f'{kmean:.1f}'
    kstd_str = f'{kstd:.1f}'
    return r'$\mathrm{k}_{' + r'\mathrm{' + subcategory + '}}$' + '= '+ kmean_str + r' $\pm$ ' + kstd_str + f', N={nbond}'


def _get_legend_label_string(label):
    return r'$\mathrm{' + label + '}$'

