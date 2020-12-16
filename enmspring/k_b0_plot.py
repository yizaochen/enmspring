from os import path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from enmspring.spring import Spring
from enmspring import k_b0_util

plt.rcParams['font.family'] = 'sans-serif'

class ScatterPlot:
    hosts = ['a_tract_21mer', 'ctct_21mer', 'gcgc_21mer',
             'g_tract_21mer', 'atat_21mer', 'tgtg_21mer']
    n_bp = 21
    type_na = 'bdna+bdna'
    
    def __init__(self, rootfolder, cutoff):
        self.rootfolder = rootfolder
        self.cutoff = cutoff
        self.d_df = self.__initialize_df()

    def plot_main(self, category, axes):
        """
        category: PP, bp, st, RB, R, PB
        """
        d_axes = self.__get_axes_index(axes)
        for host in self.hosts:
            ax = d_axes[host]
            df_temp = self.d_df[host]
            k_b0_dict = k_b0_util.get_k_b0_mean_std_by_category(df_temp, category)
            self.__scatter_subcategory(ax, k_b0_dict, category)
            self.__write_text_for_mean_std(ax, k_b0_dict, category)
            self.__set_legend(ax, category)
            self.__set_xylim(ax, category)
            self.__set_xylabel(ax, host)
            ax.set_title(f'{host}', fontsize=14)
        return d_axes

    def __initialize_df(self):
        d_temp = dict()
        for host in self.hosts:
            spring_obj = Spring(self.rootfolder, host, self.type_na, self.n_bp)
            df0 = spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff)
            d_temp[host] = k_b0_util.get_central_bps_df(df0)
        return d_temp

    def __get_axes_index(self, axes):
        d_axes = dict()
        idx = 0
        for row_id in range(2):
            for col_id in range(3):
                host = self.hosts[idx]
                d_axes[host] = axes[row_id, col_id]
                idx += 1
        return d_axes

    def __scatter_subcategory(self, ax, k_b0_dict, category):
        for subcategory in k_b0_util.category_dict[category]:
            if len(k_b0_dict[subcategory]['k']) > 2:
                x = k_b0_dict[subcategory]['b0']
                y = k_b0_dict[subcategory]['k']
                label = subcategory
                color = k_b0_util.color_dict[category][subcategory]
                ax.scatter(x, y, label=label, color=color, alpha=1, s=1)

    def __write_text_for_mean_std(self, ax, k_b0_dict, category):
        x_text, y_text, y_intv = self.__get_xytext(category)
        for subcategory in k_b0_util.category_dict[category]:
            if len(k_b0_dict[subcategory]['k']) > 2:
                mean_std_string = self.__get_mean_std_string(k_b0_dict, subcategory)
                ax.text(x_text, y_text, mean_std_string, fontsize=12)
                y_text += y_intv

    def __get_xytext(self, category):
        d_xtext = {'PP': 1.2, 'st': 1.1, 'PB': 1.1,
                   'R': 1.1, 'RB': 1.6, 'bp': 1.0}
        d_ytext = {'PP': 200, 'st': 9.0, 'PB': 9.0,
                   'R': 200, 'RB': 200, 'bp': 9.3}
        d_yintv = {'PP': -18, 'st': -1, 'PB': -1,
                   'R': -18, 'RB': -19, 'bp': -0.8}
        return d_xtext[category], d_ytext[category], d_yintv[category]

    def __set_legend(self, ax, category):
        d_lganc = {'PP': (0.9, 0.4), 'st': (0.3, 0.3), 'PB': (0.6, 0.3),
                   'R': (1, 0.3), 'RB': (0.9, 0.45), 'bp': (0.5, 0.3)}
        lgfz = 12
        lg_anchor = d_lganc[category]
        ax.legend(frameon=False, fontsize=lgfz, bbox_to_anchor=lg_anchor, loc='center right', markerscale=1.5, handletextpad=-0.1, ncol=2)

    def __get_mean_std_string(self, k_b0_dict, subcategory):
        nbond = k_b0_dict[subcategory]['nbond'] / self.n_bp
        nbond_str = f'{nbond:.1f}'
        if len(subcategory) == 2:
            str1 = ' ' + f'{subcategory}'
        else:
            str1 = f'{subcategory}'
        if len(nbond_str) == 2:
            str2 = ' ' + nbond_str
        else:
            str2 = nbond_str
        return str1 + ': ' + str2 + ' per base-step'

    def __set_xylim(self, ax, category):
        d_xlim = {'PP': (0.95, 4.8), 'st': (0.95, 4.8), 'PB': (0.95, 4.8),
                   'R': (0.95, 4.8), 'RB': (0.95, 4.8), 'bp': (0.95, 4.8)}
        d_ylim = {'PP': (-11.4, 226.5), 'st': (-0.5, 10), 'PB': (-0.5, 10),
                   'R': (-11.4, 226.5), 'RB': (-11.4, 226.5), 'bp': (-0.5, 10)}
        ax.set_xlim(d_xlim[category])
        ax.set_ylim(d_ylim[category])

    def __set_xylabel(self, ax, host):
        lbfz = 12
        xlabel = r'$b_{m}^{0}$ ($\mathrm{\AA}$) in dsDNA'
        ylabel = r'$k_m$ (kcal/mol/$\mathrm{\AA^{2}}$)'
        if host in ['g_tract_21mer', 'atat_21mer', 'tgtg_21mer']:
            ax.set_xlabel(xlabel, fontsize=lbfz)
        if host in ['a_tract_21mer', 'g_tract_21mer']:
            ax.set_ylabel(ylabel, fontsize=lbfz)

class BoxPlot:
    hosts = ['a_tract_21mer', 'ctct_21mer', 'gcgc_21mer',
             'g_tract_21mer', 'atat_21mer', 'tgtg_21mer']
    n_bp = 21
    type_na = 'bdna+bdna'
    
    def __init__(self, rootfolder, cutoff):
        self.rootfolder = rootfolder
        self.cutoff = cutoff
        self.d_df = self.__initialize_df()

    def plot_main(self, category, axes, key, nrows, ncols):
        """
        category: PP, bp, st, RB, R, PB
        key: b0, k
        """
        subcategories = k_b0_util.category_dict[category]
        d_axes = self.__get_axes_index(axes, category, nrows, ncols)
        for subcategory in subcategories:
            ax = d_axes[subcategory]
            self.__boxplot_subcategory(ax, subcategory, category, key)
            self.__set_xticklabels(ax)
            self.__set_ylabel(ax, key)
            ax.set_title(f'{subcategory}', fontsize=14)
        return d_axes

    def __boxplot_subcategory(self, ax, subcategory, category, key):
        """
        key: b0, k
        """
        x = list()
        for host in self.hosts:
            df_temp = self.d_df[host]
            k_b0_dict = k_b0_util.get_k_b0_mean_std_by_category(df_temp, category)
            if len(k_b0_dict[subcategory]['k']) > 2:
                x.append(k_b0_dict[subcategory][key])
            else:
                x.append(list())
        ax.boxplot(x)

    def __set_xticklabels(self, ax):
        xticklabels = self.hosts
        xticks = range(1, len(xticklabels)+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    def __set_ylabel(self, ax, key):
        lbfz = 12
        d_label = {'b0': r'$b_{m}^{0}$ ($\mathrm{\AA}$) in dsDNA',
                   'k': r'$k_m$ (kcal/mol/$\mathrm{\AA^{2}}$)'}
        ylabel = d_label[key]
        ax.set_ylabel(ylabel, fontsize=lbfz)

    def __initialize_df(self):
        d_temp = dict()
        for host in self.hosts:
            spring_obj = Spring(self.rootfolder, host, self.type_na, self.n_bp)
            df0 = spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff)
            d_temp[host] = k_b0_util.get_central_bps_df(df0)
        return d_temp

    def __get_axes_index(self, axes, category, nrows, ncols):
        subcategories = k_b0_util.category_dict[category]
        d_axes = dict()
        idx = 0
        for row_id in range(nrows):
            for col_id in range(ncols):
                subcategory = subcategories[idx]
                if (nrows == 1) & (ncols != 1):
                    d_axes[subcategory] = axes[col_id]
                elif (nrows == 1) & (ncols == 1):
                    d_axes[subcategory] = axes
                else:
                    d_axes[subcategory] = axes[row_id, col_id]
                idx += 1
        return d_axes

class CorrelationPlot:
    hosts = ['a_tract_21mer', 'ctct_21mer', 'gcgc_21mer',
             'g_tract_21mer', 'atat_21mer', 'tgtg_21mer']
    abbr_hosts = {'a_tract_21mer': 'A-tract', 'ctct_21mer': 'CTCT', 'gcgc_21mer': 'GCGC',
                  'g_tract_21mer': 'G-tract', 'atat_21mer': 'ATAT', 'tgtg_21mer': 'TGTG'}    
    #subcategories = ['PP0', 'PP1', 'PP2', 'PP3', 'R0', 'R1', 'RB0', 'RB1', 'RB2', 'RB3', 'PB', 
    #                 'st', 'hb', 'bp1', 'bp2']
    subcategories = ['PP0', 'PP1', 'PP2', 'R0', 'R1', 'RB0', 'RB1', 'RB2', 'st', 'hb', 'bp1']
    n_bp = 21
    type_na = 'bdna+bdna'
    
    def __init__(self, rootfolder, cutoff, corre_type):
        self.rootfolder = rootfolder
        self.metadata_folder = path.join(rootfolder, 'metadata')
        self.cutoff = cutoff
        self.d_df = self.__initialize_df()
        self.corre_type = corre_type  # 'mean' or 'median'

    def plot_single_one_curve(self, ax, pairtype_x, pairtype_y):
        x = np.array(self.get_statistics(pairtype_x))
        y = np.array(self.get_statistics(pairtype_y))

        coefs = stats.linregress(x, y)
        slope = coefs[0]
        intercept = coefs[1]
        rvalue = coefs[2]
        ax.plot(x, y, 'o', label='original data')
        ax.plot(x, intercept + slope * x, 'r', label=f'fitted line $R={rvalue:.3f}$')
        ax.legend()
        ax.set_xlabel(f'k of {pairtype_x}')
        ax.set_ylabel(f'k of {pairtype_y}')
        ax.set_title(f'{self.corre_type} of 6 systems')
        for i, host in enumerate(self.hosts):
            txt = self.abbr_hosts[host]
            ax.text(x[i], y[i], txt, fontsize=12)

    def heatmap_pearson_coefficient(self, ax, cmap='bwr', lbfz=14, textfz=14, textcolor='black'):
        heatmap_array = self.read_heatmap_array()
        im = ax.imshow(heatmap_array, cmap=cmap)
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(self.subcategories)))
        ax.set_yticks(np.arange(len(self.subcategories)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(self.subcategories, fontsize=lbfz)
        ax.set_yticklabels(self.subcategories, fontsize=lbfz)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # Loop over data dimensions and create text annotations
        for i in range(len(self.subcategories)):
            for j in range(len(self.subcategories)):
                value = heatmap_array[i, j]
                value = f'{value:.2f}'
                ax.text(j, i, value, ha="center", va="center", color=textcolor, fontsize=textfz)
        return im

    def write_heatmap_array(self):
        results = list()
        for subcategory1 in self.subcategories:
            temp = list()
            for subcategory2 in self.subcategories:
                x = np.array(self.get_statistics(subcategory1))
                y = np.array(self.get_statistics(subcategory2))
                coefs = stats.linregress(x, y)
                rvalue = coefs[2]
                temp.append(rvalue)
            results.append(temp)
        f_out = path.join(self.metadata_folder, 'heatmap_array.npy')
        np.save(f_out, results)
        print(f'Write Heatmap array to {f_out}')
        return np.array(results)

    def read_heatmap_array(self):
        f_in = path.join(self.metadata_folder, 'heatmap_array.npy')
        results = np.load(f_in)
        print(f'Read Heatmap array from {f_in}')
        return results
        
    def get_statistics(self, subcategory):
        key = 'k'
        category = k_b0_util.subcategory_reverse[subcategory]
        results = list()
        for host in self.hosts:
            df = self.d_df[host]
            k_b0_dict = k_b0_util.get_k_b0_mean_std_by_category(df, category)
            result = k_b0_dict[subcategory][key]
            if self.corre_type == 'mean':
                results.append(np.mean(result))
            elif self.corre_type == 'median':
                results.append(np.median(result))
        return results

    def __initialize_df(self):
        d_temp = dict()
        for host in self.hosts:
            spring_obj = Spring(self.rootfolder, host, self.type_na, self.n_bp)
            df0 = spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff)
            d_temp[host] = k_b0_util.get_central_bps_df(df0)
        return d_temp