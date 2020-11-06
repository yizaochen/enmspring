import pandas as pd
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from enmspring.k_b0_util import get_df_by_filter_bp
from enmspring.na_seq import sequences
from enmspring.spring import Spring

atomname_map = {'A': {'type1': 'N6', 'type2': 'N1', 'type3': 'C2'}, 
                'T': {'type1': 'O4', 'type2': 'N3', 'type3': 'O2'},
                'C': {'type1': 'N4', 'type2': 'N3', 'type3': 'O2'},
                'G': {'type1': 'O6', 'type2': 'N1', 'type3': 'N2'}}

class InputException(Exception):
    pass

class HBAgent:
    cutoff = 4.7
    type_na = 'bdna+bdna'
    d_atcg = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    def __init__(self, host, rootfolder, n_bp):
        self.rootfolder = rootfolder
        self.host = host
        self.n_bp = n_bp
        self.seq_guide = sequences[host]['guide']

        self.df_hb = self.__read_df_hb()

        self.basepairs = None

    def initialize_basepair(self):
        basepairs = dict()
        for idx, resname_i in enumerate(self.seq_guide):
            resid_i = idx + 1
            bp_obj = BasePair(resname_i, resid_i, self.df_hb)
            basepairs[resid_i] = bp_obj
        self.basepairs = basepairs

    def __read_df_hb(self):
        spring_obj = Spring(self.rootfolder, self.host, self.type_na, self.n_bp)
        df = spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff)
        df1 = get_df_by_filter_bp(df, 'hb')
        df2 = self.__read_df_at_type3()
        if len(df2) == 0:
            return df1
        else:
            df3 = pd.concat([df1,df2])
            df3 = df3.sort_values(by=['Resid_i'])
            df3 = df3.reset_index()
            return df3

    def __read_df_at_type3(self):
        spring_obj = Spring(self.rootfolder, self.host, self.type_na, self.n_bp)
        df0 = spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff)
        df1 = get_df_by_filter_bp(df0, 'bp1')
        df2_1 = self.__filter_C2_O2(df1)
        df2_2 = self.__filter_O2_C2(df1)
        return pd.concat([df2_1, df2_2])

    def __filter_C2_O2(self, df):
        mask0 = (df['Atomname_i'] == 'C2')
        df0 = df[mask0]
        mask1 = (df0['Atomname_j'] == 'O2')
        return df0[mask1]

    def __filter_O2_C2(self, df):
        mask0 = (df['Atomname_i'] == 'O2')
        df0 = df[mask0]
        mask1 = (df0['Atomname_j'] == 'C2')
        return df0[mask1]

    def get_resid_klist_by_type(self, bptype, typename):
        resid_list = list()
        k_list = list()
        for resid in range(1, self.n_bp+1):
            basepair = self.basepairs[resid]
            if basepair.bp_type == bptype:
                resid_list.append(resid)
                k_list.append(basepair.k_dict[typename])
        return resid_list, k_list

    def get_resid_klist_all(self, typename):
        resid_list = list()
        k_list = list()
        for resid in range(1, self.n_bp+1):
            basepair = self.basepairs[resid]
            resid_list.append(resid)
            k_list.append(basepair.k_dict[typename])
        return resid_list, k_list

    def get_klist_by_type(self, bptype, typename):
        k_list = list()
        for resid in range(4, 19): # For central basepair
            basepair = self.basepairs[resid]
            if basepair.bp_type == bptype:
                k_list.append(basepair.k_dict[typename])
        return k_list

    def get_d_hb_contain_atomid_k_all_basepair(self):
        typelist = ['type1', 'type2', 'type3']
        d_hb_new = {'Atomid_i': list(), 'Atomid_j': list(), 'k': list()}
        for resid_i in range(1, self.n_bp+1):
            mask = self.df_hb['Resid_i'] == resid_i
            df1 = self.df_hb[mask]
            for typename in typelist:
                atomname_i, atomname_j = self.__get_atomname_ij(resid_i, typename)
                mask = (df1['Atomname_i'] == atomname_i) & (df1['Atomname_j'] == atomname_j)
                df2 = df1[mask]
                if len(df2) == 0:
                    continue
                else:
                    d_hb_new = self.__process_d_hb_new(d_hb_new, df2)
        return d_hb_new

    def __get_atomname_ij(self, resid_i, typename):
        resname_i = self.seq_guide[resid_i-1]
        resname_j = self.d_atcg[resname_i]
        atomname_i = atomname_map[resname_i][typename]
        atomname_j = atomname_map[resname_j][typename]
        return atomname_i, atomname_j

    def __process_d_hb_new(self, d_hb_new, df_sele):
        d_hb_new['Atomid_i'].append(df_sele['Atomid_i'].iloc[0])
        d_hb_new['Atomid_j'].append(df_sele['Atomid_j'].iloc[0])
        d_hb_new['k'].append(df_sele['k'].iloc[0])
        return d_hb_new

class BasePair:
    def __init__(self, resname_i, resid_i, df):
        self.resname_i = resname_i
        self.resname_j = self.get_resname_j()
        self.resid_i = resid_i
        self.bp_type = self.determine_bptype(resname_i) # AT or GC
        self.k_dict = self.get_k_dict(df)
        self.df = df

    def determine_bptype(self, resname):
        if resname in ['A', 'T']:
            return 'AT'
        elif resname in ['G', 'C']:
            return 'GC'
        else:
            raise InputException('Something wrong with the DNA sequence.')

    def get_resname_j(self):
        d_atcg = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return d_atcg[self.resname_i]

    def get_k_dict(self, df):
        typelist = ['type1', 'type2', 'type3']
        return self.get_k_by_df(df, typelist)

    def get_k_by_df(self, df0, typelist):
        d_result = dict()
        mask1 = (df0['Resid_i'] == self.resid_i)
        df1 = df0[mask1]
        for typename in typelist:
            atomname_i = atomname_map[self.resname_i][typename]
            atomname_j = atomname_map[self.resname_j][typename]
            mask = (df1['Atomname_i'] == atomname_i)
            df2 = df1[mask]
            mask = (df2['Atomname_j'] == atomname_j)
            df3 = df2[mask]
            if len(df3) == 0:
                d_result[typename] = 0
            else:
                d_result[typename] = df3['k'].iloc[0]
        return d_result

class HBSixPlot:
    hosts = ['a_tract_21mer', 'gcgc_21mer', 'tgtg_21mer',
             'atat_21mer', 'ctct_21mer', 'g_tract_21mer']
    abbr_hosts = {'a_tract_21mer': 'A-tract', 'ctct_21mer': 'CTCT', 'gcgc_21mer': 'GCGC',
                  'g_tract_21mer': 'G-tract', 'atat_21mer': 'ATAT', 'tgtg_21mer': 'TGTG'}
    n_bp = 21
    typelist = ['type1', 'type2', 'type3']

    def __init__(self, figsize, rootfolder, lbfz, lgfz, ttfz, title_pos):
        self.figsize = figsize
        self.rootfolder = rootfolder
        self.fig, self.d_axes = self.__make_layout()

        self.lbfz = lbfz
        self.lgfz = lgfz
        self.ttfz = ttfz
        self.title_pos = title_pos

    def main(self):
        for host in self.hosts:
            h_agent = HBAgent(host, self.rootfolder, self.n_bp)
            h_agent.initialize_basepair()
            ax = self.d_axes[host]
            for typename in self.typelist:
                self.__plot_k_by_type(ax, h_agent, typename)
            self.__set_xticks(ax, host)
            self.__plot_boundary(ax)
            self.__plot_assist_lines(ax)
            self.__set_ylabel_ylim(ax)
            self.__set_title(ax, host)
    
        self.__set_legend()

    def __plot_k_by_type(self, ax, h_agent, typename):
        resids, klist = h_agent.get_resid_klist_all(typename)
        ax.plot(resids, klist, '-o', label=typename)

    def __plot_boundary(self, ax):
        ax.axvline(3, color='red', alpha=0.5)
        ax.axvline(19, color='red', alpha=0.5)

    def __plot_assist_lines(self, ax):
        ax.axhline(6, color='grey', alpha=0.4)
        ax.axhline(2, color='grey', alpha=0.4)

    def __set_ylabel_ylim(self, ax):
        ylim = (-0.5, 10)
        ax.set_ylim(ylim)
        ax.set_ylabel('k (kcal/mol/Å$^2$)', fontsize=self.lbfz)

    def __set_title(self, ax, host):
        txt = self.abbr_hosts[host]
        x = self.title_pos[0]
        y = self.title_pos[1]
        ax.text(x, y, txt, fontsize=self.ttfz)
    
    def __set_legend(self):
        self.d_axes['a_tract_21mer'].legend(fontsize=self.lgfz, ncol=3)

    def __set_xticks(self, ax, host):
        if host == 'g_tract_21mer':
            ax.set_xticks(range(1, 22))
            ax.tick_params(axis='x', labelsize=self.lbfz)
        else:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    def __set_xlabel(self):
        ax = self.d_axes['g_tract_21mer']
        ax.set_xlabel('Resid', fontsize=self.lbfz)
         
    def __make_layout(self):
        d_axes = dict()
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(6, 1, hspace=0)
        for i, host in enumerate(self.hosts):
            d_axes[host] = fig.add_subplot(gs[i, 0])
        return fig, d_axes


class HBTwoBarPlots:
    hosts = ['a_tract_21mer', 'gcgc_21mer', 'tgtg_21mer',
             'atat_21mer', 'ctct_21mer', 'g_tract_21mer']
    abbr_hosts = {'a_tract_21mer': 'A-tract', 'ctct_21mer': 'CTCT', 'gcgc_21mer': 'GCGC',
                  'g_tract_21mer': 'G-tract', 'atat_21mer': 'ATAT', 'tgtg_21mer': 'TGTG'}
    n_bp = 21
    typelist = ['type1', 'type2', 'type3']
    plot_types = ['GC', 'AT']
    width = 1

    def __init__(self, figsize, rootfolder, lbfz, lgfz, ttfz, title_pos):
        self.figsize = figsize
        self.rootfolder = rootfolder
        self.fig, self.d_axes = self.__make_layout()

        self.lbfz = lbfz
        self.lgfz = lgfz
        self.ttfz = ttfz
        self.title_pos = title_pos
        
    def main(self):
        self.__plot_by_ax('GC')
        self.__plot_by_ax('AT')
        self.__set_legend()

    def __plot_by_ax(self, hb_type):
        ylim = (0, 10)
        ax = self.d_axes[hb_type]
        for bond_type in self.typelist:
            xlist = self.__get_xlist(bond_type)
            ylist, yerrorbar = self.__get_ylist(hb_type, bond_type)
            ax.bar(xlist, ylist, self.width, yerr=yerrorbar, label=bond_type)
        self.__plot_assist_lines(ax)
        self.__set_xticks(ax)
        self.__set_title(ax, hb_type)   
        ax.set_ylim(ylim)
        ax.set_ylabel('k (kcal/mol/Å$^2$)', fontsize=self.lbfz)

    def __plot_assist_lines(self, ax):
        ax.axhline(6, color='grey', alpha=0.4)
        ax.axhline(2, color='grey', alpha=0.4)

    def __set_legend(self):
        ax = self.d_axes['AT']
        ax.legend(fontsize=self.lgfz, ncol=3)

    def __set_title(self, ax, hb_type):
        x = self.title_pos[0]
        y = self.title_pos[1]
        ax.text(x, y, hb_type, fontsize=self.ttfz)

    def __get_ylist(self, hb_type, bond_type):
        ylist = list()
        yerrorbar = list()
        for host in self.hosts:
            h_agent = HBAgent(host, self.rootfolder, self.n_bp)
            h_agent.initialize_basepair()
            klist = h_agent.get_klist_by_type(hb_type, bond_type)
            if len(klist) == 0:
                ylist.append(0)
                yerrorbar.append(0)
            else:
                karray = np.array(klist)
                ylist.append(karray.mean())
                yerrorbar.append(karray.std())
        return ylist, yerrorbar

    def __get_xlist(self, bond_type):
        starts = {'type1': 1, 'type2': 2, 'type3': 3}
        start = starts[bond_type]
        return range(start, 24, 4)

    def __set_xticks(self, ax):
        labels = [self.abbr_hosts[host] for host in self.hosts]
        ax.set_xticks(range(2,24,4))
        ax.set_xticklabels(labels, fontsize=self.lbfz)

    def __make_layout(self):
        d_axes = dict()
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=self.figsize, sharey=True)
        for hb_type, ax in zip(self.plot_types, axes):
            d_axes[hb_type] = ax
        return fig, d_axes

class HBPainter:

    complementary = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    d_typelist = {'A': ['type1', 'type2'],
                  'T': ['type1', 'type2'],
                  'G': ['type1', 'type2', 'type3'],
                  'C': ['type1', 'type2', 'type3']}
    d_color = {'A': {'type1': 27, 'type2': 10},
               'T': {'type1': 27, 'type2': 10},
               'G': {'type1': 0, 'type2': 3, 'type3': 7},
               'C': {'type1': 0, 'type2': 3, 'type3': 7}}
    n_bp = 21
    width = 5
    
    def __init__(self, host, gro, f_out, mol_id):
        self.host = host
        self.gro = gro
        self.seq_guide = sequences[host]['guide']
        self.f_out = f_out
        self.mol_id = mol_id

        self.u = mda.Universe(gro, gro)

    def main(self):
        f = open(self.f_out, 'w')
        vmdlines = self.get_vmdlines()
        for line in vmdlines:
            f.write(line)
        f.close()
        print(f'Write draw codes into {self.f_out}')

    def get_vmdlines(self):
        vmdlines = list()
        for idx, resname_i in enumerate(self.seq_guide):
            resid_i = idx + 1
            resid_j = self.get_resid_j(resid_i)
            resname_j = self.complementary[resname_i]
            typelist = self.d_typelist[resname_i]
            for hb_type in typelist:
                name_i = atomname_map[resname_i][hb_type]
                name_j = atomname_map[resname_j][hb_type]
                vmdlines.append(self.get_color_line(resname_i, hb_type))
                vmdlines.append(self.get_line(resid_i, resid_j, name_i, name_j))
        return vmdlines

    def get_resid_j(self, resid_i):
        return (2 * self.n_bp + 1) - resid_i

    def get_color_line(self, resname, hb_type):
        colorid = self.d_color[resname][hb_type]
        return f'graphics {self.mol_id} color {colorid}\n'

    def get_line(self, resid_i, resid_j, name_i, name_j):
        pos_str_i = self.get_position_str(resid_i, name_i)
        pos_str_j = self.get_position_str(resid_j, name_j)
        return f'graphics {self.mol_id} line {pos_str_i} {pos_str_j} width {self.width}\n'

    def get_position_str(self, resid, name):
        atom_select = self.u.select_atoms(f'resid {resid} and name {name}')
        if atom_select.n_atoms != 1:
            raise InputException('Something wrong with the input gro.')
        position = atom_select.positions[0]
        return '{' + f'{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}' + '}'

