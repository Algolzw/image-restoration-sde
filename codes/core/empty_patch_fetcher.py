import numpy as np
from tqdm import tqdm


class EmptyPatchFetcher:
    """
    The idea is to fetch empty patches so that real content can be replaced with this. 
    """

    def __init__(self, idx_manager, patch_size, data_frames, max_val_threshold=None):
        self._frames = data_frames
        self._idx_manager = idx_manager
        self._max_val_threshold = max_val_threshold
        self._idx_list = []
        self._patch_size = patch_size
        self._grid_size = 1
        self.set_empty_idx()

        print(f'[{self.__class__.__name__}] MaxVal:{self._max_val_threshold}')

    def compute_max(self, window):
        """
        Rolling compute.
        """
        N, H, W = self._frames.shape
        randnum = -954321
        assert self._grid_size == 1
        max_data = np.zeros((N, H - window, W - window)) * randnum

        for h in tqdm(range(H - window)):
            for w in range(W - window):
                max_data[:, h, w] = self._frames[:, h:h + window, w:w + window].max(axis=(1, 2))

        assert (max_data != 954321).any()
        return max_data

    def set_empty_idx(self):
        max_data = self.compute_max(self._patch_size)
        empty_loc = np.where(np.logical_and(max_data >= 0, max_data < self._max_val_threshold))
        # print(max_data.shape, len(empty_loc))
        self._idx_list = []
        for idx in range(len(empty_loc[0])):
            n_idx = empty_loc[0][idx]
            h_start = empty_loc[1][idx]
            w_start = empty_loc[2][idx]
            # print(n_idx,h_start,w_start)
            self._idx_list.append(self._idx_manager.idx_from_hwt(h_start, w_start, n_idx, grid_size=self._grid_size))
        
        self._idx_list = np.array(self._idx_list)
        
        assert len(self._idx_list) > 0

    def sample(self):
        return (np.random.choice(self._idx_list), self._grid_size)
