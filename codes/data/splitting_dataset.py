from typing import Tuple, Union
import numpy as np
import albumentations as A

from core.data_split_type import DataSplitType
from core.data_type import DataType
from core.empty_patch_fetcher import EmptyPatchFetcher
from data.patch_index_manager import GridAlignement, GridIndexManager
from data.split_train_val_data import get_train_val_data


class SplittingDataset:

    def __init__(self,
                 data_config,
                #  fpath: str,
                #  datasplit_type: DataSplitType = None,
                #  val_fraction=None,
                #  test_fraction=None,
                #  enable_rotation_aug: bool = False,
                #  enable_random_cropping: bool = False,
                #  max_val=None,
                #  grid_alignment=GridAlignement.LeftTop,
                #  overlapping_padding_kwargs=None
                ):
        """
        Here, an image is split into grids of size img_sz.
        Args:
            repeat_factor: Since we are doing a random crop, repeat_factor is
                given which can repeatedly sample from the same image. If self.N=12
                and repeat_factor is 5, then index upto 12*5 = 60 is allowed.
            use_one_mu_std: If this is set to true, then one mean and stdev is used
                for both channels. Otherwise, two different meean and stdev are used.

        """

        self._fpath = data_config['fpath']
        self._data = self.N = None

        if data_config['datasplit_type'] == 'train':
            datasplit_type = DataSplitType.Train
        elif data_config['datasplit_type'] == 'val':
            datasplit_type = DataSplitType.Val
        
        self.load_data(data_config,
                       datasplit_type,
                       val_fraction=data_config['val_fraction'],
                       test_fraction=data_config['test_fraction'],
                       allow_generation=False)
        self._normalized_input = True
        self.ch1_weight = data_config['ch1_weight']
        self.target_channel_idx = data_config["target_channel_idx"]
        self._quantile = data_config['clip_percentile']
        self._channelwise_quantile = data_config.get('channelwise_quantile', False)
        self._background_quantile = data_config.get('background_quantile', 0.0)
        self._clip_background_noise_to_zero = data_config.get('clip_background_noise_to_zero', False)
        self._skip_normalization_using_mean = data_config.get('skip_normalization_using_mean', False)

        self._background_values = None

        self._grid_alignment = data_config.get('grid_alignment',GridAlignement.LeftTop)
        self._overlapping_padding_kwargs = data_config.get('overlapping_padding_kwargs',None)
        if self._grid_alignment == GridAlignement.LeftTop:
            assert self._overlapping_padding_kwargs is None or data_config['multiscale_lowres_count'] is not None, "Padding is not used with this alignement style"
        elif self._grid_alignment == GridAlignement.Center:
            assert self._overlapping_padding_kwargs is not None, 'With Center grid alignment, padding is needed.'

        self._is_train = datasplit_type == DataSplitType.Train

        self._img_sz = self._grid_sz = self._repeat_factor = self.idx_manager = None
        if self._is_train:
            self.set_img_sz(data_config['GT_size'],
                            data_config['grid_size'] if 'grid_size' in data_config else data_config['GT_size'])
        else:
            self.set_img_sz(data_config['GT_size'],
                            data_config['val_grid_size'] if 'val_grid_size' in data_config else data_config['GT_size'])

        self._empty_patch_replacement_enabled = data_config.get("empty_patch_replacement_enabled",
                                                                False) and self._is_train
        if self._empty_patch_replacement_enabled:
            self._empty_patch_replacement_channel_idx = data_config['empty_patch_replacement_channel_idx']
            self._empty_patch_replacement_probab = data_config['empty_patch_replacement_probab']
            data_frames = self._data[..., self._empty_patch_replacement_channel_idx]
            # NOTE: This is on the raw data. So, it must be called before removing the background.
            self._empty_patch_fetcher = EmptyPatchFetcher(self.idx_manager,
                                                          self._img_sz,
                                                          data_frames,
                                                          max_val_threshold=data_config['empty_patch_max_val_threshold'])

        self.rm_bkground_set_max_val_and_upperclip_data(data_config.get('max_val',None), datasplit_type)

        # For overlapping dloader, image_size and repeat_factors are not related. hence a different function.

        self._mean = None
        self._std = None
        self._use_one_mu_std = True
        self._enable_rotation = data_config['enable_rotation_aug']
        self._enable_random_cropping = data_config['enable_random_cropping']
        # Randomly rotate [-90,90]

        self._rotation_transform = None
        if self._enable_rotation:
            self._rotation_transform = A.Compose([A.Flip(), A.RandomRotate90()])

        msg = self._init_msg()
        print(msg)

    def get_data_shape(self):
        return self._data.shape

    def load_data(self, data_config, datasplit_type, val_fraction=None, test_fraction=None, allow_generation=None):
        self._data = get_train_val_data(data_config,
                                        self._fpath,
                                        datasplit_type,
                                        val_fraction=val_fraction,
                                        test_fraction=test_fraction,
                                        allow_generation=allow_generation)
        self.N = len(self._data)

    def save_background(self, channel_idx, frame_idx, background_value):
        self._background_values[frame_idx, channel_idx] = background_value

    def get_background(self, channel_idx, frame_idx):
        return self._background_values[frame_idx, channel_idx]

    def remove_background(self):

        self._background_values = np.zeros((self._data.shape[0], self._data.shape[-1]))

        if self._background_quantile == 0.0:
            assert self._clip_background_noise_to_zero is False, 'This operation currently happens later in this function.'
            return

        if self._data.dtype in [np.uint16]:
            # unsigned integer creates havoc
            self._data = self._data.astype(np.int32)
        else:
            raise Exception('Handle other datatypes')

        for ch in range(self._data.shape[-1]):
            for idx in range(self._data.shape[0]):
                qval = np.quantile(self._data[idx, ..., ch], self._background_quantile)
                assert np.abs(
                    qval
                ) > 20, "We are truncating the qval to an integer which will only make sense if it is large enough"
                # NOTE: Here, there can be an issue if you work with normalized data
                qval = int(qval)
                self.save_background(ch, idx, qval)
                self._data[idx, ..., ch] -= qval

        if self._clip_background_noise_to_zero:
            self._data[self._data < 0] = 0

    def rm_bkground_set_max_val_and_upperclip_data(self, max_val, datasplit_type):
        self.remove_background()
        self.set_max_val(max_val, datasplit_type)
        self.upperclip_data()

    def upperclip_data(self):
        if isinstance(self.max_val, list):
            chN = self._data.shape[-1]
            assert chN == len(self.max_val)
            for ch in range(chN):
                ch_data = self._data[..., ch]
                ch_q = self.max_val[ch]
                ch_data[ch_data > ch_q] = ch_q
                self._data[..., ch] = ch_data
        else:
            self._data[self._data > self.max_val] = self.max_val

    def compute_max_val(self):
        if self._channelwise_quantile:
            max_val_arr = [np.quantile(self._data[..., i], self._quantile) for i in range(self._data.shape[-1])]
            return max_val_arr
        else:
            return np.quantile(self._data, self._quantile)

    def set_max_val(self, max_val, datasplit_type):
        if datasplit_type == DataSplitType.Train:
            assert max_val is None
            self.max_val = self.compute_max_val()
        else:
            assert max_val is not None
            self.max_val = max_val

    def get_max_val(self):
        return self.max_val

    def get_img_sz(self):
        return self._img_sz

    def set_img_sz(self, image_size, grid_size):
        """
        If one wants to change the image size on the go, then this can be used.
        Args:
            image_size: size of one patch
            grid_size: frame is divided into square grids of this size. A patch centered on a grid having size `image_size` is returned.
        """

        self._img_sz = image_size
        self._grid_sz = grid_size
        self.idx_manager = GridIndexManager(self._data.shape, self._grid_sz, self._img_sz, self._grid_alignment)
        self.set_repeat_factor()

    def set_repeat_factor(self):
        self._repeat_factor = self.idx_manager.grid_rows(self._grid_sz) * self.idx_manager.grid_cols(self._grid_sz)

    def _init_msg(self, ):
        msg = f'[{self.__class__.__name__}] Sz:{self._img_sz}'
        msg += f' Train:{int(self._is_train)} N:{self.N} NumPatchPerN:{self._repeat_factor}'
        msg += f' Ch1w:{self.ch1_weight}'
        msg += f' NormInp:{self._normalized_input}'
        msg += f' SingleNorm:{self._use_one_mu_std}'
        msg += f' Rot:{self._enable_rotation}'
        msg += f' RandCrop:{self._enable_random_cropping}'
        msg += f' Q:{self._quantile}'
        msg += f' ReplaceWithRandSample:{self._empty_patch_replacement_enabled}'
        if self._empty_patch_replacement_enabled:
            msg += f'-{self._empty_patch_replacement_channel_idx}-{self._empty_patch_replacement_probab}'

        msg += f' BckQ:{self._background_quantile}'
        return msg

    def _crop_imgs(self, index, *img_tuples: np.ndarray):
        h, w = img_tuples[0].shape[-2:]
        if self._img_sz is None:
            return (*img_tuples, {'h': [0, h], 'w': [0, w], 'hflip': False, 'wflip': False})

        if self._enable_random_cropping:
            h_start, w_start = self._get_random_hw(h, w)
        else:
            h_start, w_start = self._get_deterministic_hw(index)

        cropped_imgs = []
        for img in img_tuples:
            img = self._crop_flip_img(img, h_start, w_start, False, False)
            cropped_imgs.append(img)

        return (*tuple(cropped_imgs), {
            'h': [h_start, h_start + self._img_sz],
            'w': [w_start, w_start + self._img_sz],
            'hflip': False,
            'wflip': False,
        })

    def _crop_img(self, img: np.ndarray, h_start: int, w_start: int):
        if self._grid_alignment == GridAlignement.LeftTop:
            # In training, this is used.
            # NOTE: It is my opinion that if I just use self._crop_img_with_padding, it will work perfectly fine.
            # The only benefit this if else loop provides is that it makes it easier to see what happens during training.
            new_img = img[..., h_start:h_start + self._img_sz, w_start:w_start + self._img_sz]
            return new_img
        elif self._grid_alignment == GridAlignement.Center:
            # During evaluation, this is used. In this situation, we can have negative h_start, w_start. Or h_start +self._img_sz can be larger than frame
            # In these situations, we need some sort of padding. This is not needed  in the LeftTop alignement.
            return self._crop_img_with_padding(img, h_start, w_start)

    def get_begin_end_padding(self, start_pos, max_len):
        """
        The effect is that the image with size self._grid_sz is in the center of the patch with sufficient
        padding on all four sides so that the final patch size is self._img_sz.
        """
        pad_start = 0
        pad_end = 0
        if start_pos < 0:
            pad_start = -1 * start_pos

        pad_end = max(0, start_pos + self._img_sz - max_len)

        return pad_start, pad_end

    def _crop_img_with_padding(self, img: np.ndarray, h_start: int, w_start: int):
        _, H, W = img.shape
        h_on_boundary = self.on_boundary(h_start, H)
        w_on_boundary = self.on_boundary(w_start, W)

        assert h_start < H
        assert w_start < W

        assert h_start + self._img_sz <= H or h_on_boundary
        assert w_start + self._img_sz <= W or w_on_boundary
        # max() is needed since h_start could be negative.
        new_img = img[..., max(0, h_start):h_start + self._img_sz, max(0, w_start):w_start + self._img_sz]
        padding = np.array([[0, 0], [0, 0], [0, 0]])

        if h_on_boundary:
            pad = self.get_begin_end_padding(h_start, H)
            padding[1] = pad
        if w_on_boundary:
            pad = self.get_begin_end_padding(w_start, W)
            padding[2] = pad

        if not np.all(padding == 0):
            new_img = np.pad(new_img, padding, **self._overlapping_padding_kwargs)

        return new_img

    def _crop_flip_img(self, img: np.ndarray, h_start: int, w_start: int, h_flip: bool, w_flip: bool):
        new_img = self._crop_img(img, h_start, w_start)
        if h_flip:
            new_img = new_img[..., ::-1, :]
        if w_flip:
            new_img = new_img[..., :, ::-1]

        return new_img.astype(np.float32)

    def __len__(self):
        return self.N * self._repeat_factor

    def _load_img(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx = index[0]

        imgs = self._data[self.idx_manager.get_t(idx)]
        loaded_imgs = [imgs[None, ..., i] for i in range(imgs.shape[-1])]
        return tuple(loaded_imgs)

    def get_mean_std(self):
        return self._mean, self._std

    def set_mean_std(self, mean_val, std_val):
        self._mean = mean_val
        self._std = std_val

    def normalize_img(self, *img_tuples):
        mean, std = self.get_mean_std()
        mean = mean.squeeze()
        std = std.squeeze()
        normalized_imgs = []
        for i, img in enumerate(img_tuples):
            img = (img - mean[i]) / std[i]
            normalized_imgs.append(img)
        return tuple(normalized_imgs)

    def get_grid_size(self):
        return self._grid_sz

    def get_idx_manager(self):
        return self.idx_manager

    def per_side_overlap_pixelcount(self):
        return (self._img_sz - self._grid_sz) // 2

    def on_boundary(self, cur_loc, frame_size):
        return cur_loc + self._img_sz > frame_size or cur_loc < 0

    def _get_deterministic_hw(self, index: Union[int, Tuple[int, int]]):
        """
        It returns the top-left corner of the patch corresponding to index.
        """
        if isinstance(index, int):
            idx = index
            grid_size = self._grid_sz
        else:
            idx, grid_size = index

        h_start, w_start = self.idx_manager.get_deterministic_hw(idx, grid_size=grid_size)
        if self._grid_alignment == GridAlignement.LeftTop:
            return h_start, w_start
        elif self._grid_alignment == GridAlignement.Center:
            pad = self.per_side_overlap_pixelcount()
            return h_start - pad, w_start - pad

    def compute_individual_mean_std(self):
        # numpy 1.19.2 has issues in computing for large arrays. https://github.com/numpy/numpy/issues/8869
        # mean = np.mean(self._data, axis=(0, 1, 2))
        # std = np.std(self._data, axis=(0, 1, 2))
        mean_arr = []
        std_arr = []
        for ch_idx in range(self._data.shape[-1]):
            mean_ = 0.0 if self._skip_normalization_using_mean else self._data[..., ch_idx].mean()
            std_ = self._data[..., ch_idx].std()
            mean_arr.append(mean_)
            std_arr.append(std_)

        mean = np.array(mean_arr)
        std = np.array(std_arr)

        return mean[None, :, None, None], std[None, :, None, None]

    def compute_mean_std(self, allow_for_validation_data=False):
        """
        Note that we must compute this only for training data.
        """
        assert self._is_train is True or allow_for_validation_data, 'This is just allowed for training data'
        if self._use_one_mu_std is True:
            mean = [np.mean(self._data[..., k:k + 1], keepdims=True) for k in range(self._data.shape[-1])]
            mean = np.sum(mean, keepdims=True)[0]
            std = np.linalg.norm(
                [np.std(self._data[..., k:k + 1], keepdims=True) for k in range(self._data.shape[-1])],
                keepdims=True)[0]
            mean = np.repeat(mean, 2, axis=1)
            std = np.repeat(std, 2, axis=1)

            if self._skip_normalization_using_mean:
                mean = np.zeros_like(mean)

            return mean, std

        elif self._use_one_mu_std is False:
            return self.compute_individual_mean_std()

        elif self._use_one_mu_std is None:
            return np.array([0.0, 0.0]).reshape(1, 2, 1, 1), np.array([1.0, 1.0]).reshape(1, 2, 1, 1)

    def _get_random_hw(self, h: int, w: int):
        """
        Random starting position for the crop for the img with index `index`.
        """
        if h != self._img_sz:
            h_start = np.random.choice(h - self._img_sz)
            w_start = np.random.choice(w - self._img_sz)
        else:
            h_start = 0
            w_start = 0
        return h_start, w_start

    def _get_img(self, index: Union[int, Tuple[int, int]]):
        """
        Loads an image.
        Crops the image such that cropped image has content.
        """
        img_tuples = self._load_img(index)
        cropped_img_tuples = self._crop_imgs(index, *img_tuples)[:-1]
        return cropped_img_tuples

    def replace_with_empty_patch(self, img_tuples):
        empty_index = self._empty_patch_fetcher.sample()
        empty_img_tuples = self._get_img(empty_index)
        final_img_tuples = []
        for tuple_idx in range(len(img_tuples)):
            if tuple_idx == self._empty_patch_replacement_channel_idx:
                final_img_tuples.append(empty_img_tuples[tuple_idx])
            else:
                final_img_tuples.append(img_tuples[tuple_idx])
        return tuple(final_img_tuples)

    def __getitem__(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        img_tuples = self._get_img(index)
        if self._empty_patch_replacement_enabled:
            if np.random.rand() < self._empty_patch_replacement_probab:
                img_tuples = self.replace_with_empty_patch(img_tuples)

        if self._enable_rotation:
            # passing just the 2D input. 3rd dimension messes up things.
            assert len(img_tuples) == 2
            rot_dic = self._rotation_transform(image=img_tuples[0][0], mask=img_tuples[1][0])
            img1 = rot_dic['image'][None]
            img2 = rot_dic['mask'][None]
            img_tuples = (img1, img2)

        # target = np.concatenate(img_tuples, axis=0)
        img_tuples = self.normalize_img(*img_tuples)
        target = img_tuples[self.target_channel_idx]
        assert len(img_tuples) == 2
        inp = img_tuples[0]*self.ch1_weight + (1-self.ch1_weight) * img_tuples[1]

        inp = inp.astype(np.float32)
        return {"LQ": inp, "GT": target, "LQ_path": str(index), "GT_path": str(index)}


if __name__ == '__main__':
    import config.splitting.options as option
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default='codes/config/splitting/options/train/refusion.yml', help="Path to option YMAL file.")
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    dset = SplittingDataset(opt["datasets"]['train'])
    mean_val, std_val = dset.compute_mean_std()
    dset.set_mean_std(mean_val, std_val)

    datadict = dset[0]
    _,ax = plt.subplots(figsize=(8,4), ncols=2)
    ax[0].imshow(datadict['LQ'][0])
    ax[1].imshow(datadict['GT'][0])
    plt.show()
    

    