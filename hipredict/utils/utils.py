import os
import numpy as np
import matplotlib.pyplot as plt
import threading
import aiofiles
from pathlib import Path


# Define a thread-local storage to hold our session for each thread.
_thread_local = threading.local()


def check_obs_list(obs_list):
    if isinstance(obs_list, str):
        assert obs_list.endswith('.txt'), "If obs_list is a string, it must be a path to a .txt file"
        assert os.path.isfile(obs_list), f"File {obs_list} does not exist"
        obs_ids = Path(obs_list).read_text().splitlines()
    elif isinstance(obs_list, list):
        obs_ids = obs_list
    else:
        raise ValueError("obs_list must be a list or a string path to an existing .txt file")
    return obs_ids


async def download_img_async(session, url, local_file):
    """Download file async and save locally"""
    tmp_file = f"{local_file}.part"

    try:
        async with session.get(url, timeout=120) as response:
            response.raise_for_status()
            async with aiofiles.open(tmp_file, "wb") as f:
                while True:
                    chunk = await response.content.read(2**17)
                    if not chunk:
                        break
                    await f.write(chunk)

        os.replace(tmp_file, local_file)
        return local_file
    except Exception:
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except OSError:
            pass
        raise


def save_img(array, save_dir, image_name, title, lims=None, dpi=100.0):
    """Saves a 2D or 3D image array to a file using mpl"""

    if not isinstance(array, np.ndarray):
        raise ValueError("Input array must be a NumPy array")
    if array.ndim not in [2, 3]:
        raise ValueError("Input array must be 2D or 3D")

    fig, ax = plt.subplots()

    if array.ndim == 2:
        img = ax.imshow(array, cmap="gray", vmin=lims[0] if lims else None, vmax=lims[1] if lims else None)
    elif array.ndim == 3:
        if array.max() > 1:
            array = _normalize_3d_array(array)
        img = ax.imshow(array)

    if array.ndim == 2:
        fig.colorbar(img, ax=ax)

    ax.set_title(title)
    ax.axis("off")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, image_name)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _normalize_3d_array(array):
    """Normalizes a 3D array to the range [0, 255]"""
    array_min = array.min()
    array_max = array.max()

    if array_max == array_min:
        print(f"WARNING: Normalization failed, array has constant value {array_max}")
        return np.zeros_like(array, dtype=np.uint8)  # Avoid division by zero

    normalized_array = (array - array_min) / (array_max - array_min) * 255
    return normalized_array.astype(np.uint8)


def get_max_workers(max_cap=4, leave_free=1, verbose=True):
    """
    Detect a sensible ProcessPool worker count on HPC or local machines.

    Priority:
      1) CPU affinity (best signal of actual CPUs available to this process)
      2) PBS_NCPUS
      3) PBS_NODEFILE line count (cluster-dependent semantics)
      4) NCPUS
      5) os.cpu_count()

    Args:
        max_cap: Hard cap on workers (helps avoid I/O / RAM thrash).
        leave_free: Leave this many CPUs free for main process / overhead.
        verbose: Print detection info.

    Returns:
        max_workers (int): Worker count to use for ProcessPoolExecutor.
    """
    ncpus = None
    source = None

    # 1) CPU affinity (usually the best runtime truth on managed HPC systems)
    try:
        if hasattr(os, "sched_getaffinity"):
            aff = os.sched_getaffinity(0)
            if aff:
                ncpus = len(aff)
                source = "os.sched_getaffinity(0)"
    except Exception as e:
        if verbose:
            print(f"WARNING: Failed reading CPU affinity: {e}")

    # 2) PBS_NCPUS
    if ncpus is None:
        ncpus_env = os.environ.get("PBS_NCPUS")
        if ncpus_env:
            try:
                ncpus = int(ncpus_env)
                source = "PBS_NCPUS"
            except ValueError:
                if verbose:
                    print(f"WARNING: Invalid PBS_NCPUS='{ncpus_env}'")

    # 3) PBS_NODEFILE (semantics vary by cluster; may be hosts not slots)
    if ncpus is None:
        nodefile = os.environ.get("PBS_NODEFILE")
        if nodefile and os.path.exists(nodefile):
            try:
                with open(nodefile, "r") as f:
                    ncpus = sum(1 for _ in f)
                source = f"PBS_NODEFILE ({nodefile})"
            except Exception as e:
                if verbose:
                    print(f"WARNING: Failed reading PBS_NODEFILE '{nodefile}': {e}")

    # 4) NCPUS
    if ncpus is None:
        ncpus_env = os.environ.get("NCPUS")
        if ncpus_env:
            try:
                ncpus = int(ncpus_env)
                source = "NCPUS"
            except ValueError:
                if verbose:
                    print(f"WARNING: Invalid NCPUS='{ncpus_env}'")

    # 5) local fallback
    if ncpus is None:
        ncpus = os.cpu_count() or 1
        source = "os.cpu_count()"

    ncpus = max(1, int(ncpus))
    max_workers = ncpus - int(leave_free)
    max_workers = max(1, max_workers)
    max_workers = min(int(max_cap), max_workers)

    if verbose:
        print(f"CPU detect: source={source}, ncpus={ncpus}, max_workers={max_workers}")

    return max_workers
