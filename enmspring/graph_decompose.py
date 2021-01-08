import matplotlib.pyplot as plt
from enmspring.graphs import hosts
from enmspring.graphs import Stack

class SixStack:
    d_titles = {'a_tract_21mer': ('A-Tract: (AA)', 'A-Tract: (TT)'), 'g_tract_21mer': ('G-Tract: (GG)', 'G-Tract: (CC)'),
                'atat_21mer': ('AT: (AT)', 'AT: (AT)'), 'gcgc_21mer':  ('GC: (GC)', 'GC: (GC)'),
                'ctct_21mer':  ('CT: (CT)', 'CT: (GA)'), 'tgtg_21mer': ('TG: (TG)', 'TG: (AC)')}

    def __init__(self, rootfolder):
        self.rootfolder = rootfolder
        self.g_agents = self.get_g_agents_and_preprocess()

    def get_g_agents_and_preprocess(self):
        g_agents = dict()
        for host in hosts:
            g_agents[host] = Stack(host, self.rootfolder)
            g_agents[host].pre_process()
        return g_agents

    def get_lambda_id_list(self, host, start_mode, end_mode, strandid):
        d_strand = {'STRAND1': self.g_agents[host].strand1_array, 'STRAND2': self.g_agents[host].strand2_array}
        return d_strand[strandid][start_mode:end_mode+1]

    def get_Alist_Dlist(self, host, start_mode, end_mode, strandid):
        Alist = list()
        Dlist = list()
        lambda_id_list = self.get_lambda_id_list(host, start_mode, end_mode, strandid)
        for lambda_id in lambda_id_list:
            qtAq = self.g_agents[host].get_qtAq(lambda_id)
            qtDq = self.g_agents[host].get_qtDq(lambda_id)
            Alist.append(qtAq)
            Dlist.append(qtDq)
        return Alist, Dlist

    def plot_qtAq_qtDq(self, figsize, start_mode, end_mode, strandid, ylim, yvalues):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, sharey=True)
        idx = 0
        for row_id in range(2):
            for col_id in range(3):
                host = hosts[idx]
                ax = axes[row_id, col_id]
                self.bar_qtAq_qtDq_by_ax(ax, host, start_mode, end_mode, strandid)
                if ylim is not None:
                    ax.set_ylim(ylim)
                self.draw_assist_lines(ax, yvalues)
                idx += 1
        return fig, axes

    def bar_qtAq_qtDq_by_ax(self, ax, host, start_mode, end_mode, strandid):
        w_small = 0.5
        w_big = 0.8
        x_Alist, x_Dlist, xticks = self.get_xAD_xticks(w_small, w_big)
        Alist, Dlist = self.get_Alist_Dlist(host, start_mode, end_mode, strandid)
        ax.bar(x_Alist, Alist, w_small, label=r'$\mathbf{q}_i^{T} A \mathbf{q}_i$')
        ax.bar(x_Dlist, Dlist, w_small, label=r'$\mathbf{q}_i^{T} D \mathbf{q}_i$')
        ax.set_xticks(xticks)
        ax.set_xticklabels(range(start_mode, end_mode+1))
        ax.set_ylabel(r'Decomposed $\lambda$ (kcal/mol/Å$^2$)')
        ax.set_xlabel('Mode id, $i$')
        ax.legend()
        d_strandid = {'STRAND1': 0, 'STRAND2': 1}
        title = self.d_titles[host][d_strandid[strandid]]
        ax.set_title(title)

    def get_xAD_xticks(self, w_small, w_big):
        x_Alist = list()
        x_Dlist = list()
        xticks = list()
        x = 0.
        for idx in range(20):
            x_Alist.append(x)
            x += w_small
            x_Dlist.append(x)
            x += w_big
            xticks.append((x_Alist[idx]+x_Dlist[idx]) / 2)
        return x_Alist, x_Dlist, xticks

    def draw_assist_lines(self, ax, yvalues):
        for yvalue in yvalues:
            ax.axhline(yvalue, color='grey', alpha=0.2)
        

class AtomImportance:
    d_atomlist = {'A': ['N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N6', 'N7', 'C8', 'N9'],
                  'T': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6', 'C7'],
                  'C': ['N1', 'C2', 'O2', 'N3', 'N4', 'C4', 'C5', 'C6'],
                  'G': ['N1', 'C2', 'N2', 'N3', 'C4', 'C5', 'C6', 'O6', 'N7', 'C8', 'N9']}
    
    def __init__(self, host, rootfolder):
        self.host = host
        self.rootfolder = rootfolder
        self.g_agent = self.get_g_agent_and_preprocess()

    def get_g_agent_and_preprocess(self):
        g_agent = Stack(self.host, self.rootfolder)
        g_agent.pre_process()
        return g_agent

