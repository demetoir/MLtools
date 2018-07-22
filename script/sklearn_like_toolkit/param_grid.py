from sklearn.model_selection import ParameterGrid, ParameterSampler


def to_list_grid(grid, depth=0, recursive=True):
    for key, val in grid.items():
        if type(val) is dict and recursive:
            grid[key] = to_list_grid(val, depth=depth + 1)

    if depth is not 0:
        grid = ParameterGrid(grid)
    return grid


def param_grid_random(grid, n_iter):
    return ParameterSampler(to_list_grid(grid), n_iter=n_iter)


def param_grid_full(grid):
    return ParameterGrid(to_list_grid(grid))
