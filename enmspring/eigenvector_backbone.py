import matplotlib.pyplot as plt
import numpy as np

class NresidEigenvaluePlot:
    lbfz = 12
    ttfz = 14

    def __init__(self, host, strand, s_agent):
        self.host = host
        self.strand = strand
        self.s_agent = s_agent

        self.eigenvalue_array = self.get_eigenvalue_array()
        self.n_data = len(self.eigenvalue_array)

    def plot(self, figsize, criteria=0.01, ylim1=None, ylim2=None, assist_hlines=None):
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
        x_array = list(range(self.n_data))
        y_array_2 = self.get_number_of_resid_involve_all_eigenvector(criteria)

        color = 'blue'
        ax1.plot(x_array, self.eigenvalue_array, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylabel('Eigenvalue', fontsize=self.lbfz, color=color)

        color = 'red'
        ax2.plot(x_array, y_array_2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('Number of Resid Involved', fontsize=self.lbfz, color=color)

        ax1.set_xlabel('Mode Index', fontsize=self.lbfz)
        ax1.set_title(f'{self.host}-{self.strand}', fontsize=self.ttfz)
        if ylim1 is not None:
            ax1.set_ylim(ylim1)
        if ylim2 is not None:
            ax1.set_ylim(ylim1)
        if assist_hlines is not None:
            for hline in assist_hlines:
                ax2.axhline(hline, color='grey', alpha=0.2)
        return fig, ax1, ax2

    def get_number_of_resid_involve_all_eigenvector(self, criteria):
        d_temp = {'STRAND1': self.s_agent.strand1_array, 'STRAND2': self.s_agent.strand2_array}
        return [self.get_number_of_resid(eigv_id, criteria) for eigv_id in d_temp[self.strand]]

    def get_number_of_resid(self, sele_id, criteria):
        eigenvector = np.abs(self.s_agent.get_eigenvector_by_id(sele_id))
        idx_array = np.where(eigenvector > criteria)[0]
        cgname_lst = [self.s_agent.d_idx_inverse[key] for key in idx_array]
        resid_lst = [self.s_agent.resid_map[cgname] for cgname in cgname_lst]
        return len(list(set(resid_lst)))

    def get_eigenvalue_array(self):
        return self.s_agent.get_lambda_by_strand(self.strand)