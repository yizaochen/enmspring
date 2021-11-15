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

def get_df_by_filter_PP2_angles(df0):
    mask = (df0['Atomname_i'] == 'P') & (df0['Atomname_j'] == "C5'")
    df1 = df0[~mask]
    mask = (df1['Atomname_i'] == "C5'") & (df1['Atomname_j'] == 'P')
    df2 = df1[~mask]
    mask = (df2['Atomname_i'] == "C3'") & (df2['Atomname_j'] == 'P')
    df3 = df2[~mask]
    mask = (df3['Atomname_i'] == 'P') & (df3['Atomname_j'] == "C3'")
    return df3[~mask]

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

def get_df_same_resid(df0):
    mask = (df0['Resid_i'] == df0['Resid_j'])
    return df0[mask]

def get_df_not_same_resid(df0):
    mask = (df0['Resid_i'] == df0['Resid_j'])
    return df0[~mask]

class FilterSB0Agent:
    d_resname_lst = {'a_tract_21mer': ['A', 'T'], 'g_tract_21mer': ['G', 'C'], 
                     'atat_21mer': ['A', 'T'], 'gcgc_21mer': ['G', 'C']}

    def __init__(self, host, df, d_seq):
        self.host = host
        self.df = df
        self.d_seq = d_seq
        self.resname_lst = self.d_resname_lst[self.host]

    def filterSB0_main(self):
        self.add_resname_col()
        df_R, df_Y = self.get_split_df()
        df_R = self.filter_R(df_R)
        df_Y = self.filter_Y(df_Y)
        return pd.concat([df_R, df_Y])

    def add_resname_col(self):
        def get_resname(strand_id, resid):
            return self.d_seq[strand_id][resid-1]
        self.df['resname_i'] = self.df.apply(lambda x: get_resname(strand_id = x['Strand_i'], resid = x['Resid_i']), axis=1)

    def get_split_df(self):
        df_R = self.df[self.df['resname_i'] == self.resname_lst[0]]
        df_Y = self.df[self.df['resname_i'] == self.resname_lst[1]]
        return df_R, df_Y

    def filter_R(self, df0):
        # A, G: remove C1'-N3, C2'-C8, C2'-N7
        mask = ((df0['Atomname_i'] == "C1'") & (df0['Atomname_j'] == "N3")) | ((df0['Atomname_j'] == "C1'") & (df0['Atomname_i'] == "N3"))
        df1 = df0[~mask]
        mask = ((df1['Atomname_i'] == "C2'") & (df1['Atomname_j'] == "C8")) | ((df1['Atomname_j'] == "C2'") & (df1['Atomname_i'] == "C8"))
        df2 = df1[~mask]
        mask = ((df2['Atomname_i'] == "C2'") & (df2['Atomname_j'] == "N7")) | ((df2['Atomname_j'] == "C2'") & (df2['Atomname_i'] == "N7"))
        df3 = df2[~mask]
        return df3

    def filter_Y(self, df0):
        # C, T: remove C1'-O2, C2'-C6, C2'-C5
        mask = ((df0['Atomname_i'] == "C1'") & (df0['Atomname_j'] == "O2")) | ((df0['Atomname_j'] == "C1'") & (df0['Atomname_i'] == "O2"))
        df1 = df0[~mask]
        mask = ((df1['Atomname_i'] == "C2'") & (df1['Atomname_j'] == "C6")) | ((df1['Atomname_j'] == "C2'") & (df1['Atomname_i'] == "C6"))
        df2 = df1[~mask]
        mask = ((df2['Atomname_i'] == "C2'") & (df2['Atomname_j'] == "C5")) | ((df2['Atomname_j'] == "C2'") & (df2['Atomname_i'] == "C5"))
        df3 = df2[~mask]
        return df3

    
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

