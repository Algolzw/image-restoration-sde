"""
We would like to have a common logic to map between an index and location on the image.
We assume the data to be of shape N * H * W * C (C: channels, H,W: spatial dimensions, N: time/number of frames)
We assume the square patches.
The extra content on the right side will not be used( as shown below). 
.-----------.-.
|           | |
|           | |
|           | |
|           | |
.-----------.-.

"""
from tkinter import Grid

from core.custom_enum import Enum


class GridAlignement(Enum):
    """
    A patch is formed by padding the grid with content. If the grids are 'Center' aligned, then padding is to done equally on all 4 sides.
    On the other hand, if grids are 'LeftTop' aligned, padding is to be done on the right and bottom end of the grid.
    In the former case, one needs (patch_size - grid_size)//2 amount of content on the right end of the frame. 
    In the latter case, one needs patch_size - grid_size amount of content on the right end of the frame. 
    """
    LeftTop = 0
    Center = 1


class GridIndexManager:

    def __init__(self, data_shape, grid_size, patch_size, grid_alignement) -> None:
        self._data_shape = data_shape
        self._default_grid_size = grid_size
        self.patch_size = patch_size
        self.N = self._data_shape[0]
        self._align = grid_alignement

    def use_default_grid(self, grid_size):
        return grid_size is None or grid_size < 0

    def grid_rows(self, grid_size):
        if self._align == GridAlignement.LeftTop:
            extra_pixels = (self.patch_size - grid_size)
        elif self._align == GridAlignement.Center:
            # Center is exclusively used during evaluation. In this case, we use the padding to handle edge cases.
            # So, here, we will ideally like to cover all pixels and so extra_pixels is set to 0.
            # If there was no padding, then it should be set to (self.patch_size - grid_size) // 2
            extra_pixels = 0

        return ((self._data_shape[-3] - extra_pixels) // grid_size)

    def grid_cols(self, grid_size):
        if self._align == GridAlignement.LeftTop:
            extra_pixels = (self.patch_size - grid_size)
        elif self._align == GridAlignement.Center:
            extra_pixels = 0

        return ((self._data_shape[-2] - extra_pixels) // grid_size)

    def grid_count(self, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        return self.N * self.grid_rows(grid_size) * self.grid_cols(grid_size)

    def hwt_from_idx(self, index, grid_size=None):
        t = self.get_t(index)
        return (*self.get_deterministic_hw(index, grid_size=grid_size), t)

    def idx_from_hwt(self, h_start, w_start, t, grid_size=None):
        """
        Given h,w,t (where h,w constitutes the top left corner of the patch), it returns the corresponding index.
        """
        if grid_size is None:
            grid_size = self._default_grid_size

        nth_row = h_start // grid_size
        nth_col = w_start // grid_size

        index = self.grid_cols(grid_size) * nth_row + nth_col
        return index * self._data_shape[0] + t

    def get_t(self, index):
        return index % self.N

    def get_top_nbr_idx(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        ncols = self.grid_cols(grid_size)
        index -= ncols * self.N
        if index < 0:
            return None

        return index

    def get_bottom_nbr_idx(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        ncols = self.grid_cols(grid_size)
        index += ncols * self.N
        if index > self.grid_count(grid_size=grid_size):
            return None

        return index

    def get_left_nbr_idx(self, index, grid_size=None):
        if self.on_left_boundary(index, grid_size=grid_size):
            return None

        index -= self.N
        return index

    def get_right_nbr_idx(self, index, grid_size=None):
        if self.on_right_boundary(index, grid_size=grid_size):
            return None
        index += self.N
        return index

    def on_left_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        factor = index // self.N
        ncols = self.grid_cols(grid_size)

        left_boundary = (factor // ncols) != (factor - 1) // ncols
        return left_boundary

    def on_right_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        factor = index // self.N
        ncols = self.grid_cols(grid_size)

        right_boundary = (factor // ncols) != (factor + 1) // ncols
        return right_boundary

    def on_top_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        ncols = self.grid_cols(grid_size)
        return index < self.N * ncols

    def on_bottom_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        ncols = self.grid_cols(grid_size)
        return index + self.N * ncols > self.grid_count(grid_size=grid_size)

    def on_boundary(self, idx, grid_size=None):
        if self.on_left_boundary(idx, grid_size=grid_size):
            return True

        if self.on_right_boundary(idx, grid_size=grid_size):
            return True

        if self.on_top_boundary(idx, grid_size=grid_size):
            return True

        if self.on_bottom_boundary(idx, grid_size=grid_size):
            return True
        return False

    def get_deterministic_hw(self, index: int, grid_size=None):
        """
        Fixed starting position for the crop for the img with index `index`.
        """
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        # _, h, w, _ = self._data_shape
        # assert h == w
        factor = index // self.N
        ncols = self.grid_cols(grid_size)

        ith_row = factor // ncols
        jth_col = factor % ncols
        h_start = ith_row * grid_size
        w_start = jth_col * grid_size
        return h_start, w_start


if __name__ == '__main__':
    grid_size = 32
    patch_size = 64
    index = 13
    manager = GridIndexManager((1, 499, 469, 2), grid_size, patch_size, GridAlignement.Center)
    h_start, w_start = manager.get_deterministic_hw(index)
    print(h_start, w_start, manager.grid_count())
    print(manager.grid_rows(grid_size), manager.grid_cols(grid_size))
