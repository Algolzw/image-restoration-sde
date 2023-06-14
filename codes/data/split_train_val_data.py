"""
Here, the idea is to load the data from different data dtypes into a single interface.
"""
from typing import Union

from core.data_split_type import DataSplitType
from core.data_type import DataType
# from data_loader.allencell_rawdata_loader import get_train_val_data as _loadallencellmito
# from data_loader.embl_semisup_rawdata_loader import get_train_val_data as _loadembl2_semisup
# from data_loader.ht_iba1_ki67_rawdata_loader import get_train_val_data as _load_ht_iba1_ki67
from data.ch2_tiff_data import train_val_data as _load_tiff_train_val
# from data_loader.pavia2_rawdata_loader import get_train_val_data as _loadpavia2
# from data_loader.pavia2_rawdata_loader import get_train_val_data_vanilla as _loadpavia2_vanilla
# from data_loader.schroff_rawdata_loader import get_train_val_data as _loadschroff_mito_er
# from data_loader.sinosoid_dloader import train_val_data as _loadsinosoid
# from data_loader.sinosoid_threecurve_dloader import train_val_data as _loadsinosoid3curve
# from data_loader.two_tiff_rawdata_loader import get_train_val_data as _loadseparatetiff


def get_train_val_data(data_config,
                       fpath,
                       datasplit_type: DataSplitType,
                       val_fraction=None,
                       test_fraction=None,
                       allow_generation=None,
                       ignore_specific_datapoints=None):
    """
    Ensure that the shape of data should be N*H*W*C: N is number of data points. H,W are the image dimensions.
    C is the number of channels.
    """
    assert isinstance(datasplit_type, int)
    return _load_tiff_train_val(fpath,
                                data_config,
                                datasplit_type,
                                val_fraction=val_fraction,
                                test_fraction=test_fraction)

