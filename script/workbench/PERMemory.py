import numpy as np


# copy and modified from bellow
# https://gist.github.com/simoninithomas/d6adc6edb0a7f37d6323a5e3d2ab72ec#file-dueling-deep-q-learning-with-doom-double-dqns-and-prioritized-experience-replay-ipynb

class SumTree:
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences

        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        # self.data_pointer = (self.data_pointer + 1) % self.leaf_size

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        diff = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the diff through tree
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += diff

    def get_leaf(self, v):
        parent_index = 0
        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            if v <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                v -= self.tree[left_child_index]
                parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class PERMemory:  # stored as ( s, a, r, s_ ) in SumTree

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity, e=0.01, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.e = e
        # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.alpha = alpha
        # importance-sampling, from initial value increasing to 1
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

        self.tree = SumTree(capacity)
        self.cnt = 0

    def __len__(self):
        return self.cnt

    @property
    def capacity(self):
        return self.tree.capacity

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

        self.cnt = min(self.cnt + 1, self.capacity)

    def sample(self, n):
        batch = []
        idxs = np.empty(n, dtype=np.int32)
        IS_weights = np.empty(n, dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)

            # p(i)
            proba = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            IS_weights[i] = np.power((proba * self.capacity), -self.beta)

            idxs[i] = index

            batch.append(data)

        # normalize
        IS_weights /= np.max(IS_weights)

        return idxs, batch, IS_weights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
