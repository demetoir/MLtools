from script.util.JobPool import JobPool
import multiprocessing as mp

from script.util.PlotTools import PlotTools

CPU_COUNT = mp.cpu_count() - 1


class asyncPlotTools(JobPool):
    def __init__(self, save=True, show=False, extend='.png', dpi=300, n_job=CPU_COUNT):
        super().__init__(n_job)
        self.instance = PlotTools(save=save, show=show, extend=extend, dpi=dpi)

    def dist(self, df, column, bins=None, ax=None, axlabel=None, rug=False, path=None, **kwargs):
        args = [df, column]
        kwargs.update({
            'bins':    bins,
            'ax':      ax,
            'axlabel': axlabel,
            'rug':     rug,
            'path':    path,
        })

        self.apply_async(self.instance.dist, args=args, kwargs=kwargs)

    def dist_groupby(self, df, col, groupby_col, path=None, ax=None, axlabel=None, **kwargs):
        args = [df, col, groupby_col]
        kwargs.update({
            'path':    path,
            'ax':      ax,
            'axlabel': axlabel,
        })

        self.apply_async(self.instance.dist_groupby, args=args, kwargs=kwargs)

    def count(self, df, column, groupby_col=None, path=None, **kwargs):
        args = [df, column]
        kwargs.update({
            'groupby_col': groupby_col,
            'path':        path,
        })

        self.apply_async(self.instance.count, args=args, kwargs=kwargs)

    def line(self, dots, linewidth=1, path=None, **kwargs):
        args = [dots]
        kwargs.update({
            'linewidth': linewidth,
            'path':      path,
        })

        self.apply_async(self.instance.line, args=args, kwargs=kwargs)

    def scatter_2d(self, *np_arr, labels=None, marker_size=3, path=None, **kwargs):
        args = [*np_arr]
        kwargs.update({
            'labels':      labels,
            'marker_size': marker_size,
            'path':        path,
        })

        self.apply_async(self.instance.scatter_2d, args=args, kwargs=kwargs)

    def joint_2d(self, df, x_col, y_col, kind='reg', path=None, **kwargs):
        args = [df, x_col, y_col]
        kwargs.update({
            'kind': kind,
            'path': path,
        })

        self.apply_async(self.instance.joint_2d, args=args, kwargs=kwargs)

    def violin_plot(self, df, x_col, y_col, grouby_col=None, with_swarmplot=False, path=None, **kwargs):
        args = [df, x_col, y_col]
        kwargs.update({
            'grouby_col':     grouby_col,
            'with_swarmplot': with_swarmplot,
            'path':           path,
        })

        self.apply_async(self.instance.violin_plot, args=args, kwargs=kwargs)

    def heatmap(self, np_arr, vmin=-1.0, vmax=1.0, center=0, annot=False, fmt='.2f', cmap='BrBG_r', mask=False,
                mask_color='white', path=None, **kwargs):
        args = [np_arr]
        kwargs.update({
            'vmin':       vmin,
            'vmax':       vmax,
            'center':     center,
            'annot':      annot,
            'fmt':        fmt,
            'cmap':       cmap,
            'mask':       mask,
            'mask_color': mask_color,
            'path':       path,
        })

        self.apply_async(self.instance.heatmap, args=args, kwargs=kwargs)

    def cluster_map(self, df, row_group_key=None, col_group_key=None, metric='correlation', path=None, **kwargs):
        args = [df]
        kwargs.update({
            'row_group_key': row_group_key,
            'col_group_key': col_group_key,
            'metric':        metric,
            'path':          path,
        })

        self.apply_async(self.instance.cluster_map, args=args, kwargs=kwargs)

    def pair_plot(self, df, groupby_col=None, path=None, **kwargs):
        args = [df]
        kwargs.update({
            'groupby_col': groupby_col,
            'path':        path,
        })

        self.apply_async(self.instance.pair_plot, args=args, kwargs=kwargs)

    def plot_image(self, np_img, title=None, path=None, **kwargs):
        args = [np_img]
        kwargs.update({
            'title': title,
            'path':  path,
        })

        self.apply_async(self.instance.plot_image, args=args, kwargs=kwargs)

    def plot_image_tile(self, np_imgs, column=10, path=None, title=None, **kwargs):
        args = [np_imgs]
        kwargs.update({
            'column': column,
            'path':   path,
            'title':  title,
        })

        self.apply_async(self.instance.plot_image_tile, args=args, kwargs=kwargs)
