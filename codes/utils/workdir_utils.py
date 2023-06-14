from pathlib import Path
from datetime import datetime
import os

def get_new_model_version(model_dir: str) -> str:
    """
    A model will have multiple runs. Each run will have a different version.
    """
    versions = []
    for version_dir in os.listdir(model_dir):
        try:
            versions.append(int(version_dir))
        except:
            print(f'Invalid subdirectory:{model_dir}/{version_dir}. Only integer versions are allowed')
            continue
    if len(versions) == 0:
        return '0'
    return f'{max(versions) + 1}'


def get_model_name(config):
    mtype = config.model.model_type
    dtype = config.data.data_type
    ltype = config.loss.loss_type
    stype = config.data.sampler_type

    return f'D{dtype}-M{mtype}-S{stype}-L{ltype}'


def get_month():
    return datetime.now().strftime("%y%m")


def get_workdir(root_dir, use_max_version):
    rel_path = get_month()
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    if use_max_version:
        # Used for debugging.
        version = int(get_new_model_version(cur_workdir))
        if version > 0:
            version = f'{version - 1}'

        rel_path = os.path.join(rel_path, str(version))
    else:
        rel_path = os.path.join(rel_path, get_new_model_version(cur_workdir))

    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)
    return cur_workdir, rel_path
