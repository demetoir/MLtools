import seaborn as sns
import numpy as np
from pprint import pprint
from script.util.JobPool import JobPool
from script.util.PlotTools import PlotTools
import pandas

DF = pandas.DataFrame


def test_plot_tools_pair_plot():
    iris = sns.load_dataset("iris")

    plt_tools = PlotTools()
    plt_tools.pair_plot(iris)


def test_plot_tool_to_2d_square():
    plt_tools = PlotTools()
    rand_x_1d = np.random.normal(3, 5, [15])
    pprint(plt_tools.to_2d_square(rand_x_1d))
    pprint(plt_tools.to_2d_square(rand_x_1d).shape)

    rand_x_1d = np.random.normal(3, 5, [16])
    pprint(plt_tools.to_2d_square(rand_x_1d))
    pprint(plt_tools.to_2d_square(rand_x_1d).shape)

    rand_x_1d = np.random.normal(3, 5, [17])
    pprint(plt_tools.to_2d_square(rand_x_1d))
    pprint(plt_tools.to_2d_square(rand_x_1d).shape)


def test_plot_tools_cluster_map():
    rand_2d = np.random.normal(0, 1, [20, 20])
    plt_tools = PlotTools()
    plt_tools.cluster_map(rand_2d)


def test_plot_tools_heatmap():
    rand_2d = np.random.normal(0, 1, [20, 20])
    plt_tools = PlotTools()
    plt_tools.heatmap(rand_2d)

    rand_x_1d = np.random.normal(3, 5, [17])
    plt_tools.heatmap(rand_x_1d)


def test_plot_tools_violine():
    tips = sns.load_dataset("tips")

    plt_tools = PlotTools()
    plt_tools.violin_plot(tips, "day", "total_bill", with_swarmplot=True, hue="sex")


def test_plt_joint_2d():
    tips = sns.load_dataset("tips")
    plt_tools = PlotTools()
    plt_tools.joint_2d(tips, "total_bill", "tip")


def test_plt_tool_count():
    titanic = sns.load_dataset("titanic")
    plt_tools = PlotTools()
    plt_tools.count(titanic, column='class', hue='who')


def test_plt_line():
    xs = [np.array([i * float(k) for i in range(32)]) for k in range(-10, 10)]

    plt_tools = PlotTools()
    plt_tools.line(xs)


def test_plt_scatter_2d():
    xys = [(np.random.uniform(10, -10, [100]), np.random.normal(k, 1, [100])) for k in range(1, 20)]

    plt_tools = PlotTools()
    plt_tools.scatter_2d(xys)


def test_plt_dist():
    rand_x_1d = np.random.normal(3, 5, [20])
    plt_tools = PlotTools()
    plt_tools.dist(rand_x_1d)


def test_plot_tool_timeit():
    rand_x_1d = np.random.normal(3, 5, [100])
    tool = PlotTools()
    for i in range(10):
        tool.dist(rand_x_1d)


def test_plot_tool_async_timeit():
    rand_x_1d = np.random.normal(3, 5, [100])
    tool = PlotTools()

    pool = JobPool(processes=3)

    childs = []
    for i in range(10):
        child = pool.apply_async(tool.dist, args=[rand_x_1d])
        childs += [child]

    for child in childs:
        child.get()


def test_plot_font():
    df = DF({'한글': [0, 1, 2, 3], 'english': [4, 5, 6, 7]})
    plot = PlotTools(save=True, show=False)
    corr = df.corr()
    plot.heatmap(corr, title='heatmap_labeled, 한글')
