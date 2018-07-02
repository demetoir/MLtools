import itertools
import numpy as np
import warnings
from PIL import Image
from script.util.misc_util import time_stamp, path_join, setup_file

color_set = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
marker_set = [
    # '.',
    # ',',
    'o', 'v', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd',
    '|', '_', ]
"""
    ``'.'``          point marker
    ``','``          pixel marker
    ``'o'``          circle marker
    ``'v'``          triangle_down marker
    ``'^'``          triangle_up marker
    ``'<'``          triangle_left marker
    ``'>'``          triangle_right marker
    ``'1'``          tri_down marker
    ``'2'``          tri_up marker
    ``'3'``          tri_left marker
    ``'4'``          tri_right marker
    ``'s'``          square marker
    ``'p'``          pentagon marker
    ``'*'``          star marker
    ``'h'``          hexagon1 marker
    ``'H'``          hexagon2 marker
    ``'+'``          plus marker
    ``'x'``          x marker
    ``'D'``          diamond marker
    ``'d'``          thin_diamond marker
    ``'|'``          vline marker
    ``'_'``          hline marker
"""

line_set = ['-', '--', '-.', ':']
"""
    ``'-'``          solid line style
    ``'--'``         dashed line style
    ``'-.'``         dash-dot line style
    ``':'``          dotted line style
"""

scatter_markers = list(itertools.product(marker_set, color_set))
# scatter_markers = map(lambda x: "".join(x), scatter_markers)

line_marker = list(itertools.product(line_set, color_set))
line_marker = map(lambda x: "".join(x), line_marker)



class PlotTools:

    def __init__(self, dpi=300):
        self.fig_count = 0
        self.dpi = dpi

        import seaborn as _sns
        self.sns = _sns
        del _sns

        import matplotlib.pyplot as _plt
        self.plt = _plt
        del _plt


    @staticmethod
    def to_2d_square(np_arr_1d, padding=None):

        size = len(np_arr_1d)
        import math
        sqrt = int(math.sqrt(size))
        del math

        if sqrt ** 2 < size:
            sqrt += 1
        sqrt = int(sqrt)
        np_arr_1d = np.concatenate([np_arr_1d, np.zeros([abs(sqrt ** 2 - size)])], axis=0)
        ret = np_arr_1d.reshape([sqrt, sqrt])
        return ret

    @staticmethod
    def finger_print(size, head='_'):
        ret = head
        h_list = [c for c in '01234556789QWERTYUIOPASDFGHJKLZXCVBNM']
        for i in range(size):
            ret += np.random.choice(h_list)
        return ret

    @property
    def figure(self):
        self.fig_count += 1
        return self.plt.figure(self.fig_count)

    @staticmethod
    def _fig2data(fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        return buf

    @staticmethod
    def _fig2PILimg(fig):
        """
        @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
        @param fig a matplotlib figure
        @return a Python Imaging Library ( PIL ) image
        """
        # put the figure pixmap into a numpy array
        buf = PlotTools._fig2data(fig)
        w, h, d = buf.shape
        return Image.frombytes("RGBA", (w, h), buf.tostring())

    def plt_common_teardown(self, fig, path=None, show=False, title=None, extend='.png', dpi=None):
        if title is None:
            title = time_stamp() + self.finger_print(6)
        self.plt.title(title)

        if path is None:
            path = path_join('.', 'matplot', title + extend)
        setup_file(path)

        if dpi is None:
            dpi = 300

        fig.savefig(path, dpi=300)

        if show:
            self.plt.show()

        self.plt.close(fig)

    def sns_setup(self):
        self.sns.set()
        self.sns.set_style('whitegrid')
        self.sns.set_color_codes()

    def dist(self, np_x, bins=None, ax=None, axlabel=None, path=None, show=False, title=None, extend='.png'):
        warnings.filterwarnings(module='matplotlib*', action='ignore', category=UserWarning)

        self.sns_setup()

        sns_plot = self.sns.distplot(np_x, bins=bins, rug=True, hist=True, ax=ax, axlabel=axlabel,
                                         rug_kws={"color": "g", 'label': 'rug'},
                                         kde_kws={"color": "r", "label": "KDE"},
                                         hist_kws={"color": "b", "label": 'hist'})
        fig = sns_plot.figure

        self.plt_common_teardown(fig, path=path, show=show, title=title, extend=extend)

    def count(self, df, column, hue=None, path=None, show=False, title=None, extend='.png'):
        self.sns_setup()

        # fig = self.figure
        sns_plot = self.sns.countplot(x=column, data=df, hue=hue)
        fig = sns_plot.figure

        self.plt_common_teardown(fig, path=path, show=show, title=title, extend=extend)

    def line(self, dots, linewidth=1, path=None, show=False, title=None, extend='.png'):
        self.sns_setup()
        fig = self.figure

        if type(dots) is np.array:
            print(type(dots))

        for idx, (x, marker) in enumerate(zip(dots, line_marker)):
            # print(f"{marker}")
            self.plt.plot(x, marker, label=str(idx), linewidth=linewidth)

        self.plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        self.plt_common_teardown(fig, path=path, show=show, title=title, extend=extend)

    def scatter_2d(self, dots, marker_size=2, path=None, show=False, title=None, extend='.png'):
        self.sns_setup()
        fig = self.figure

        for idx, (xy, (marker, color)) in enumerate(zip(dots, scatter_markers)):
            x, y = xy
            self.plt.scatter(x, y, marker_size, color, marker, label=str(idx))

        self.plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        self.plt_common_teardown(fig, path=path, show=show, title=title, extend=extend)

    def joint_2d(self, x_col, y_col, df, kind='reg', path=None, show=False, title=None, extend='.png'):
        self.sns_setup()
        # fig = self.figure

        sns_plot = self.sns.jointplot(x_col, y_col, data=df, kind=kind)
        fig = sns_plot.fig

        self.plt_common_teardown(fig, path=path, show=show, title=title, extend=extend)

    def violin_plot(self, x_col, y_col, df, hue=None, with_swarmplot=True, path=None, show=False, title=None,
                    extend='.png'):
        self.sns_setup()
        sns_plot = self.sns.violinplot(x_col, y_col, hue=hue, data=df)
        if with_swarmplot:
            dodge = True if hue is not None else False
            sns_plot = self.sns.swarmplot(x_col, y_col, hue=hue, data=df, dodge=dodge,
                                          color='magenta', size=5)
        fig = sns_plot.figure

        self.plt_common_teardown(fig, path=path, show=show, title=title, extend=extend)

    def heatmap(self, np_arr, vmin=-1.0, vmax=1.0, center=0, annot=False, fmt='.2f', cmap=None, mask=False,
                mask_color='white', path=None, show=False, title=None, extend='.png'):
        self.sns_setup()

        dim = len(np_arr.shape)
        if dim == 1:
            np_arr = self.to_2d_square(np_arr)
        elif dim > 2:
            raise TypeError(f'np_arr dimension is {dim}, dimension must lower than 3')

        if mask:
            mask = np.zeros_like(np_arr)
            mask[np.triu_indices_from(mask)] = True
            with self.sns.axes_style(mask_color):
                sns_plot = self.sns.heatmap(np_arr, mask=mask, vmax=vmax, vmin=vmin, center=center, fmt=fmt,
                                            cmap=cmap,
                                            annot=annot, annot_kws={})
        else:
            sns_plot = self.sns.heatmap(np_arr, mask=mask, vmax=vmax, vmin=vmin, center=center, fmt=fmt, cmap=cmap,
                                        annot=annot)
        fig = sns_plot.figure

        self.plt_common_teardown(fig, path=path, show=show, title=title, extend=extend)

    def cluster_map(self, df, row_group_key=None, col_group_key=None, metric='correlation', path=None, show=False,
                    title=None, extend='.png'):
        self.sns_setup()

        # if row_group_key is not None:
        #     row = df[row_group_key]
        #     lut = dict(zip(row.unique(), "rbgcym"))
        #     row_colors = row.map(lut)
        # else:
        #     row_colors = None
        #
        # if col_group_key is not None:
        #     col = df[col_group_key]
        #     lut = dict(zip(col.unique(), "rbgcym"))
        #     col_colors = col.map(lut)
        # else:
        #     col_colors = None

        sns_plot = self.sns.clustermap(df, metric=metric)
        fig = sns_plot.fig

        self.plt_common_teardown(fig, path=path, show=show, title=title, extend=extend)

    def pair_plot(self, df, hue=None, path=None, show=False, title=None, extend='.png'):
        self.sns_setup()

        sns_plot = self.sns.pairplot(df, hue=hue)
        sns_plot.map_diag(self.plt.hist)
        sns_plot.map_offdiag(self.plt.scatter)
        if hue is not None:
            sns_plot.add_legend()

        fig = sns_plot.fig
        self.plt_common_teardown(fig, path=path, show=show, title=title, extend=extend)
