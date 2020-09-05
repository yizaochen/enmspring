import matplotlib.pyplot as plt
import k_b0_util
plt.rcParams['font.family'] = 'Arial' # Set font family to Arial for plot
lbfz = 10  # x,y label fontsize
lgfz = 8
xtickfz = 9
ytickfz = 9
ylabel = r'$k_m$ (kcal/mol/$\mathrm{\AA^{2}}$)'
xlabel = r'$b_{m}^{0}$ ($\mathrm{\AA}$) in dsDNA'
xlim = (0.95, 4.8)
n_bp = 10
mkfz = 1
bp_st_pb_yticks = [0, 5, 10]

class dsDNA():
    def __init__(self, df0, figsize, dpi):
        self.figsize = figsize
        self.dpi = dpi
        self.df0 = df0
        self.fig = None
        self.axes = None
        
    def plot_main(self):
        self.fig, self.axes = plt.subplots(nrows=2, ncols=3, 
                figsize=self.figsize, tight_layout={'pad': 0})
        self._plot_PP()
        self._plot_bp()
        self._plot_st()
        self._plot_RB()
        self._plot_R()
        self._plot_PB()
        plt.savefig('dsDNA.eps', format='eps')
        plt.show()
    
    def _scatter_for_subcategory(self, ax, k_b0_dict, category, subcategory):
        if len(k_b0_dict[subcategory]['k']) > 2: 
            ax.scatter(k_b0_dict[subcategory]['b0'], k_b0_dict[subcategory]['k'], 
                       label=subcategory, 
                       color=k_b0_util.color_dict[category][subcategory], 
                       alpha=1, s=mkfz)
            
    def _get_mean_std_string(self, k_b0_dict, subcategory):
        nbond = k_b0_dict[subcategory]['nbond'] / n_bp
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
            
    def _write_text_for_mean_std(self, ax, x_text, y_text, k_b0_dict, subcategory):
        if len(k_b0_dict[subcategory]['k']) > 2:
            mean_std_string = self._get_mean_std_string(k_b0_dict, subcategory)
            ax.text(x_text, y_text, mean_std_string, fontsize=8)
    
    def _plot_PP(self):
        ax = self.axes[0,0]
        category = 'PP'
        x_text = 1.2
        y_text = 200
        y_text_interval = -18
        lg_anchor = (1.06, 0.4)
        ylim = (-11.4, 226.5)
        k_b0_dict = k_b0_util.get_k_b0_mean_std_by_category(self.df0, category)
        for subcategory in k_b0_util.category_dict[category]:
            self._scatter_for_subcategory(ax, k_b0_dict, category, subcategory)
            self._write_text_for_mean_std(ax, x_text, y_text, k_b0_dict, subcategory)
            y_text += y_text_interval
        ax.legend(frameon=False, fontsize=lgfz, bbox_to_anchor=lg_anchor, loc='center right', markerscale=1.5, handletextpad=-0.1, ncol=2)
        ax.set_xlim(xlim)
        ax.set_ylabel(ylabel, fontsize=lbfz)
        ax.set_ylim(ylim)
        ax.tick_params(axis='x', labelsize=xtickfz)
        ax.tick_params(axis='y', labelsize=ytickfz)
        
    def _plot_bp(self):
        ax = self.axes[0,1]
        category = 'bp'
        x_text = 1.0
        y_text = 9.3
        y_text_interval = -0.8
        lg_anchor = (0.5, 0.3)
        ylim = (-0.5, 10)
        k_b0_dict = k_b0_util.get_k_b0_mean_std_by_category(self.df0, category)
        for subcategory in k_b0_util.category_dict[category]:
            self._scatter_for_subcategory(ax, k_b0_dict, category, subcategory)
            self._write_text_for_mean_std(ax, x_text, y_text, k_b0_dict, subcategory)
            y_text += y_text_interval
        ax.legend(frameon=False, fontsize=lgfz, bbox_to_anchor=lg_anchor, loc='center right', markerscale=1.5, handletextpad=-0.1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_yticks(bp_st_pb_yticks)
        ax.tick_params(axis='x', labelsize=xtickfz)
        ax.tick_params(axis='y', labelsize=ytickfz)
        
    def _plot_st(self):
        ax = self.axes[0,2]
        category = 'st'
        x_text = 1.1
        y_text = 9.0
        y_text_interval = -1
        lg_anchor = (0.3, 0.3)
        ylim = (-0.5, 10)
        k_b0_dict = k_b0_util.get_k_b0_mean_std_by_category(self.df0, category)
        for subcategory in k_b0_util.category_dict[category]:
            self._scatter_for_subcategory(ax, k_b0_dict, category, subcategory)
            self._write_text_for_mean_std(ax, x_text, y_text, k_b0_dict, subcategory)
            y_text += y_text_interval
        ax.legend(frameon=False, fontsize=lgfz, bbox_to_anchor=lg_anchor, loc='center right', markerscale=1.5, handletextpad=-0.1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_yticks(bp_st_pb_yticks)
        ax.tick_params(axis='x', labelsize=xtickfz)
        ax.tick_params(axis='y', labelsize=ytickfz)
        
    def _plot_RB(self):
        ax = self.axes[1,0]
        category = 'RB'
        x_text = 1.6
        y_text = 200
        y_text_interval = -19
        lg_anchor = (1.05, 0.45)
        ylim = (-11.4, 226.5)
        k_b0_dict = k_b0_util.get_k_b0_mean_std_by_category(self.df0, category)
        for subcategory in k_b0_util.category_dict[category]:
            self._scatter_for_subcategory(ax, k_b0_dict, category, subcategory)
            self._write_text_for_mean_std(ax, x_text, y_text, k_b0_dict, subcategory)
            y_text += y_text_interval
        ax.legend(frameon=False, fontsize=lgfz, bbox_to_anchor=lg_anchor, loc='center right', markerscale=1.5, handletextpad=-0.1, ncol=2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel, fontsize=lbfz)
        ax.set_xlabel(xlabel, fontsize=lbfz)
        ax.tick_params(axis='x', labelsize=xtickfz)
        ax.tick_params(axis='y', labelsize=ytickfz)
    
    def _plot_R(self):
        ax = self.axes[1,1]
        category = 'R'
        x_text = 1.1
        y_text = 200
        y_text_interval = -18
        lg_anchor = (1, 0.3)
        ylim = (-11.4, 226.5)
        k_b0_dict = k_b0_util.get_k_b0_mean_std_by_category(self.df0, category)
        for subcategory in k_b0_util.category_dict[category]:
            self._scatter_for_subcategory(ax, k_b0_dict, category, subcategory)
            self._write_text_for_mean_std(ax, x_text, y_text, k_b0_dict, subcategory)
            y_text += y_text_interval
        ax.legend(frameon=False, fontsize=lgfz, bbox_to_anchor=lg_anchor, loc='center right', markerscale=1.5, handletextpad=-0.1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel, fontsize=lbfz)
        ax.tick_params(axis='x', labelsize=xtickfz)
        ax.tick_params(axis='y', labelsize=ytickfz)
        
    def _plot_PB(self):
        ax = self.axes[1,2]
        category = 'PB'
        x_text = 1.1
        y_text = 9.0
        y_text_interval = -1
        lg_anchor = (0.3, 0.3)
        ylim = (-0.5, 10)
        k_b0_dict = k_b0_util.get_k_b0_mean_std_by_category(self.df0, category)
        for subcategory in k_b0_util.category_dict[category]:
            self._scatter_for_subcategory(ax, k_b0_dict, category, subcategory)
            self._write_text_for_mean_std(ax, x_text, y_text, k_b0_dict, subcategory)
            y_text += y_text_interval
        ax.legend(frameon=False, fontsize=lgfz, bbox_to_anchor=lg_anchor, loc='center right', markerscale=1.5, handletextpad=-0.1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_yticks(bp_st_pb_yticks)
        ax.set_xlabel(xlabel, fontsize=lbfz)
        ax.tick_params(axis='x', labelsize=xtickfz)
        ax.tick_params(axis='y', labelsize=ytickfz)
        
