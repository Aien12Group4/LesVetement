import numpy as np
import scipy.sparse as sp
import json
import time

from .Dataloader import Dataloader

class DataLoaderPolyvore(Dataloader):
    """
    Load polyvore data.
    """
    def __init__(self):
        super(DataLoaderPolyvore, self).__init__(path='data/polyvore/dataset/')

    def init_phase(self, phase):
        print('init phase: {}'.format(phase))
        assert phase in ['train', 'valid', 'test']
        path_dataset = self.path_dataset
        adj_file = path_dataset + 'adj_{}.npz'.format(phase)
        feats_file = path_dataset + 'features_{}.npz'.format(phase)
        np.random.seed(1234)

        adj = sp.load_npz(adj_file).astype(np.int32)
        setattr(self, '{}_adj'.format(phase), adj)
        node_features = sp.load_npz(feats_file)
        setattr(self, '{}_features'.format(phase), node_features)

        # get lower tiangle of the adj matrix to avoid duplicate edges
        setattr(self, 'lower_{}_adj'.format(phase), sp.tril(adj).tocsr())

    def get_phase(self, phase):
        print('get phase: {}'.format(phase))
        assert phase in ['train', 'valid', 'test']

        lower_adj = getattr(self, 'lower_{}_adj'.format(phase))

        # get the positive edges

        pos_r_idx, pos_c_idx = lower_adj.nonzero()
        pos_labels = np.array(lower_adj[pos_r_idx, pos_c_idx]).squeeze()

        # split the positive edges into the ones used for evaluation and the ones used as message passing
        n_pos = pos_labels.shape[0] # number of positive edges
        perm = list(range(n_pos))
        np.random.shuffle(perm)
        pos_labels, pos_r_idx, pos_c_idx = pos_labels[perm], pos_r_idx[perm], pos_c_idx[perm]
        n_eval = int(n_pos/2)
        mp_pos_labels, mp_pos_r_idx, mp_pos_c_idx = pos_labels[n_eval:], pos_r_idx[n_eval:], pos_c_idx[n_eval:]
        # this are the positive examples that will be used to compute the loss function
        eval_pos_labels, eval_pos_r_idx, eval_pos_c_idx = pos_labels[:n_eval], pos_r_idx[:n_eval], pos_c_idx[:n_eval]

        # get the negative edges

        print('Sampling negative edges...')
        before = time.time()
        n_train_neg = eval_pos_labels.shape[0] # set the number of negative training edges that will be needed to sample at each iter
        neg_labels = np.zeros((n_train_neg))
        # get the possible indexes to be sampled (basically all indexes if there aren't restrictions)
        poss_nodes = np.arange(lower_adj.shape[0])

        neg_r_idx = np.zeros((n_train_neg))
        neg_c_idx = np.zeros((n_train_neg))

        for i in range(n_train_neg):
            r_idx, c_idx = self.get_negative_training_edge(poss_nodes, poss_nodes.shape[0], lower_adj)
            neg_r_idx[i] = r_idx
            neg_c_idx[i] = c_idx
        print('Sampling done, time elapsed: {}'.format(time.time() - before))

        # build adj matrix
        adj = sp.csr_matrix((
                    np.hstack([mp_pos_labels, mp_pos_labels]),
                    (np.hstack([mp_pos_r_idx, mp_pos_c_idx]), np.hstack([mp_pos_c_idx, mp_pos_r_idx]))
                ),
                shape=(lower_adj.shape[0], lower_adj.shape[0])
            )
        # remove the labels of the negative edges which are 0
        adj.eliminate_zeros()

        labels = np.append(eval_pos_labels, neg_labels)
        r_idx = np.append(eval_pos_r_idx, neg_r_idx)
        c_idx = np.append(eval_pos_c_idx, neg_c_idx)

        return getattr(self, '{}_features'.format(phase)), adj, labels, r_idx, c_idx

    def get_test_compatibility(self):
        """
        This function is not used now, becaue full_adj is empty because all the edges have been removed
        """
        self.setup_test_compatibility()

        flat_questions = []
        gt = []
        q_ids = []
        q_id = 0
        for outfit in self.comp_outfits:
            items = outfit[0]
            for i in range(len(items)):
                for to_idx in items[i+1:]:
                    from_idx = items[i]
                    flat_questions.append([from_idx, to_idx])
                    gt.append(outfit[1])
                    q_ids.append(q_id)
            q_id += 1

        assert len(flat_questions) == len(gt) and len(q_ids) == len(gt)
        assert len(self.comp_outfits) == max(q_ids)+1

        flat_questions = np.array(flat_questions)
        gt = np.array(gt)
        q_ids = np.array(q_ids)

        # now build the adj for message passing for the questions, by removing the edges that will be evaluated
        # lower_adj = getattr(self, 'lower_{}_adj'.format('test'))
        lower_adj = getattr(self, 'lower_{}_adj'.format('test'))

        full_adj = lower_adj + lower_adj.transpose()
        full_adj = full_adj.tolil()
        for edge, label in zip(flat_questions, gt):
            u, v = edge
            full_adj[u, v] = 0
            full_adj[v, u] = 0

        full_adj = full_adj.tocsr()
        full_adj.eliminate_zeros()

        # make sure that none of the query edges are in the adj matrix
        count_edges = 0
        count_pos = 0
        for edge in flat_questions:
            u,v = edge
            if full_adj[u,v] > 0:
                count_pos += 1
            count_edges += 1
        assert count_pos == 0

        return full_adj, flat_questions[:, 0], flat_questions[:, 1], gt, q_ids

    def setup_test_compatibility(self, resampled=False):
        """
        """
        comp_file = self.path_dataset + 'compatibility_test.json'
        if resampled:
            comp_file = self.path_dataset + 'compatibility_RESAMPLED_test.json'
        with open(comp_file) as f:
            self.comp_outfits = json.load(f)
