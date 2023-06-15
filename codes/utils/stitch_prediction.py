import numpy as np


class PatchLocation:
    """
    Encapsulates t_idx and spatial location.
    """

    def __init__(self, h_idx_range, w_idx_range, t_idx):
        self.t = t_idx
        self.h_start, self.h_end = h_idx_range
        self.w_start, self.w_end = w_idx_range

    def __str__(self):
        msg = f'T:{self.t} [{self.h_start}-{self.h_end}) [{self.w_start}-{self.w_end}) '
        return msg


def _get_location(extra_padding, hwt, pred_h, pred_w):
    h_start, w_start, t_idx = hwt
    h_start -= extra_padding
    h_end = h_start + pred_h
    w_start -= extra_padding
    w_end = w_start + pred_w
    return PatchLocation((h_start, h_end), (w_start, w_end), t_idx)


def get_location_from_idx(dset, dset_input_idx, pred_h, pred_w):
    """
    For a given idx of the dataset, it returns where exactly in the dataset, does this prediction lies. 
    Note that this prediction also has padded pixels and so a subset of it will be used in the final prediction.
    Which time frame, which spatial location (h_start, h_end, w_start,w_end)
    Args:
        dset:
        dset_input_idx:
        pred_h:
        pred_w:

    Returns:
    """
    extra_padding = dset.per_side_overlap_pixelcount()
    htw = dset.get_idx_manager().hwt_from_idx(dset_input_idx, grid_size=dset.get_grid_size())
    return _get_location(extra_padding, htw, pred_h, pred_w)


def set_skip_boundary_pixels_mask(mask, loc, skip_count):
    if skip_count == 0:
        return mask
    assert skip_count > 0
    assert loc.h_end - skip_count >= 0
    assert loc.w_end - skip_count >= 0
    mask[loc.t, :, loc.h_start:loc.h_start + skip_count, loc.w_start:loc.w_end] = False
    mask[loc.t, :, loc.h_end - skip_count:loc.h_end, loc.w_start:loc.w_end] = False
    mask[loc.t, :, loc.h_start:loc.h_end, loc.w_start:loc.w_start + skip_count] = False
    mask[loc.t, :, loc.h_start:loc.h_end, loc.w_end - skip_count:loc.w_end] = False


def set_skip_central_pixels_mask(mask, loc, skip_count):
    if skip_count == 0:
        return mask
    assert skip_count > 0
    h_mid = (loc.h_start + loc.h_end) // 2
    w_mid = (loc.w_start + loc.w_end) // 2
    l_skip = skip_count // 2
    r_skip = skip_count - l_skip
    mask[loc.t, :, h_mid - l_skip:h_mid + r_skip, w_mid - l_skip:w_mid + r_skip] = False


def stitched_prediction_mask(dset, padded_patch_shape, skip_boundary_pixel_count, skip_central_pixel_count):
    """
    Returns the boolean matrix. It will be 0 if it lies either in skipped boundaries or skipped central pixels
    Args:
        dset:
        padded_patch_shape:
        skip_boundary_pixel_count:
        skip_central_pixel_count:

    Returns:
    """
    N, H, W, C = dset.get_data_shape()
    mask = np.full((N, C, H, W), True)
    hN, wN = padded_patch_shape
    for dset_input_idx in range(len(dset)):
        loc = get_location_from_idx(dset, dset_input_idx, hN, wN)
        set_skip_boundary_pixels_mask(mask, loc, skip_boundary_pixel_count)
        set_skip_central_pixels_mask(mask, loc, skip_central_pixel_count)

    old_img_sz = dset.get_img_sz()
    dset.set_img_sz(dset._img_sz_for_hw)
    mask = stitch_predictions(mask, dset)
    dset.set_img_sz(old_img_sz)
    return mask


def _get_smoothing_mask(cropped_pred_shape, smoothening_pixelcount, loc, frame_size):
    """
    It returns a mask. If the mask is multipled with all predictions and predictions are then added to 
    the overall frame at their corect location, it would simulate following scenario:
    take all patches belonging to a row. join these patches by smoothening their vertical boundaries. 
    Then take all these combined and smoothened rows. join them vertically by smoothening the horizontal boundaries.
    For this to happen, one needs *= operation as used here.  
    """
    mask = np.ones(cropped_pred_shape)
    on_leftb = loc.w_start == 0
    on_rightb = loc.w_end >= frame_size
    on_topb = loc.h_start == 0
    on_bottomb = loc.h_end >= frame_size

    if smoothening_pixelcount == 0:
        return mask

    assert 2 * smoothening_pixelcount <= min(cropped_pred_shape)
    if (not on_leftb) and (not on_rightb) and (not on_topb) and (not on_bottomb):
        assert 4 * smoothening_pixelcount <= min(cropped_pred_shape)

    w_levels = np.arange(1, 0, step=-1 * 1 / (2 * smoothening_pixelcount + 1))[1:].reshape((1, -1))
    if not on_rightb:
        mask[:, -2 * smoothening_pixelcount:] *= w_levels
    if not on_leftb:
        mask[:, :2 * smoothening_pixelcount] *= w_levels[:, ::-1]

    if not on_bottomb:
        mask[-2 * smoothening_pixelcount:, :] *= w_levels.T

    if not on_topb:
        mask[:2 * smoothening_pixelcount, :] *= w_levels[:, ::-1].T

    return mask


def remove_pad(pred, loc, extra_padding, smoothening_pixelcount, frame_shape):
    assert smoothening_pixelcount == 0
    if extra_padding - smoothening_pixelcount > 0:
        h_s = extra_padding - smoothening_pixelcount

        # rows
        h_N = frame_shape[0]
        if loc.h_end > h_N:
            assert loc.h_end - extra_padding + smoothening_pixelcount <= h_N
        h_e = extra_padding - smoothening_pixelcount

        w_s = extra_padding - smoothening_pixelcount

        # columns
        w_N = frame_shape[1]
        if loc.w_end > w_N:
            assert loc.w_end - extra_padding + smoothening_pixelcount <= w_N

        w_e = extra_padding - smoothening_pixelcount

        return pred[h_s:-h_e, w_s:-w_e]

    return pred


def update_loc_for_final_insertion(loc, extra_padding, smoothening_pixelcount):
    extra_padding = extra_padding - smoothening_pixelcount
    loc.h_start += extra_padding
    loc.w_start += extra_padding
    loc.h_end -= extra_padding
    loc.w_end -= extra_padding
    return loc


def stitch_predictions(predictions, dset, smoothening_pixelcount=0):
    """
    Args:
        smoothening_pixelcount: number of pixels which can be interpolated
    """
    assert smoothening_pixelcount >= 0 and isinstance(smoothening_pixelcount, int)

    extra_padding = dset.per_side_overlap_pixelcount()
    # if there are more channels, use all of them.
    shape = list(dset.get_data_shape())
    shape[-1] =  predictions.shape[1]

    output = np.zeros(shape, dtype=predictions.dtype)
    frame_shape = shape[1:3]
    for dset_input_idx in range(predictions.shape[0]):
        loc = get_location_from_idx(dset, dset_input_idx, predictions.shape[-2], predictions.shape[-1])

        mask = None
        cropped_pred_list = []
        for ch_idx in range(predictions.shape[1]):
            # class i
            cropped_pred_i = remove_pad(predictions[dset_input_idx, ch_idx], loc, extra_padding, smoothening_pixelcount,
                                        frame_shape)

            if mask is None:
                # NOTE: don't need to compute it for every patch.
                assert smoothening_pixelcount == 0, "For smoothing,enable the get_smoothing_mask. It is disabled since I don't use it and it needs modification to work with non-square images"
                mask = 1
                # mask = _get_smoothing_mask(cropped_pred_i.shape, smoothening_pixelcount, loc, frame_size)

            cropped_pred_list.append(cropped_pred_i)

        loc = update_loc_for_final_insertion(loc, extra_padding, smoothening_pixelcount)
        for ch_idx in range(predictions.shape[1]):
            output[loc.t, loc.h_start:loc.h_end, loc.w_start:loc.w_end, ch_idx] += cropped_pred_list[ch_idx] * mask

    return output


if __name__ == '__main__':
    from data.patch_index_manager import GridAlignement, GridIndexManager
    grid_size = 32
    patch_size = 64
    data_shape = (1, 1550, 1920, 2)
    N = data_shape[0] * (data_shape[1] // grid_size) * (data_shape[2] // grid_size)
    predictions = np.zeros((N, 2, patch_size, patch_size))
    # data_shape, grid_size, patch_size, grid_alignement
    idx_manager = GridIndexManager(data_shape, grid_size, patch_size, GridAlignement.Center)

    class TestDataSet:

        def __init__(self) -> None:
            self.idx_manager = idx_manager

        def per_side_overlap_pixelcount(self):
            return (patch_size - grid_size) // 2

        def get_data_shape(self):
            return data_shape

        def get_grid_size(self):
            return grid_size

        def get_idx_manager(self):
            return self.idx_manager

    dset = TestDataSet()
    # import pdb;pdb.set_trace()
    stitch_predictions = stitch_predictions(predictions, dset, smoothening_pixelcount=0)
    print(stitch_predictions.shape)
    # loc = PatchLocation((0, 32), (0, 32), 5)
    # extra_padding = 16
    # smoothening_pixelcount = 4
    # frame_size = 2720
    # out = remove_pad(np.ones((64, 64)), loc, extra_padding, smoothening_pixelcount, frame_size)
    # mask = _get_smoothing_mask(out.shape, smoothening_pixelcount, loc, frame_size)
    # print(loc)
    # loc = update_loc_for_final_insertion(loc, extra_padding, smoothening_pixelcount, frame_size)
    # print(loc, mask.shape, out.shape)

    # import matplotlib.pyplot as plt
    # plt.imshow(mask, cmap='hot')
    # plt.show()
    # extra_padding = 0
    # hwt1 = (0, 0, 0)
    # pred_h = 4
    # pred_w = 4
    # hwt2 = (pred_h, pred_w, 2)
    # loc1 = _get_location(extra_padding, hwt1, pred_h, pred_w)
    # loc2 = _get_location(extra_padding, hwt2, pred_h, pred_w)
    # mask = np.full((10, 8, 8), 1)
    # set_skip_boundary_pixels_mask(mask, loc1, 1)
    # set_skip_boundary_pixels_mask(mask, loc2, 1)
    # print(mask[hwt1[-1]])
    # print(mask[hwt2[-1]])
