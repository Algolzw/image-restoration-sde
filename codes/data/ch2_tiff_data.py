from typing import Union

import numpy as np

from core.tiff_reader import load_tiff
from core.data_type import DataType
from core.data_split_type import DataSplitType, get_datasplit_tuples


def train_val_data(fpath, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    print(f"Loading {fpath} with Channels {data_config['channel_1']},{data_config['channel_2']},"
          f"datasplit mode:{DataSplitType.name(datasplit_type)}")
    data = load_tiff(fpath)
    if data_config['data_type'] == DataType.Prevedel_EMBL:
        # Ensure that the last dimension is the channel dimension.
        data = data[..., None]
        data = np.swapaxes(data, 1, 4)
        data = data.squeeze()

    return _train_val_data(data,
                           datasplit_type,
                           data_config['channel_1'],
                           data_config['channel_2'],
                           val_fraction=val_fraction,
                           test_fraction=test_fraction)


def _train_val_data(data, datasplit_type: DataSplitType, channel_1, channel_2, val_fraction=None, test_fraction=None):
    assert data.shape[-1] > max(channel_1, channel_2), 'Invalid channels'
    data = data[..., [channel_1, channel_2]]
    if datasplit_type == DataSplitType.All:
        return data.astype(np.float32)

    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(data))

    if datasplit_type == DataSplitType.Train:
        return data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        return data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        return data[test_idx].astype(np.float32)
    else:
        raise Exception("invalid datasplit")