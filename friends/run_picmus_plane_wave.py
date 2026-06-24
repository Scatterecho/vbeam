"""Run the vbeam PICMUS plane-wave example as a small Python script.

This is a command-line version of docs/examples/plane_wave_dataset.ipynb. It downloads
the PICMUS carotid cross UFF dataset, imports it through PyUFF, builds a JAX-jitted DAS
beamformer, and saves the reconstructed image.
"""

from pathlib import Path
import os
import site
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("PYTHONNOUSERSITE", "1")

USER_SITE = Path(site.getusersitepackages()).resolve()
sys.path = [
    path
    for path in sys.path
    if not path or Path(path).resolve() != USER_SITE
]

import jax

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyuff_ustb import Uff

from vbeam.beamformers import get_das_beamformer
from vbeam.data_importers import import_pyuff
from vbeam.fastmath import backend_manager
from vbeam.util.download import cached_download


DATA_URL = "http://www.ustb.no/datasets/PICMUS_carotid_cross.uff"


def main() -> None:
    backend_manager.active_backend = "jax"

    data_dir = SCRIPT_DIR / "data"
    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    import pyuff_ustb

    print(f"Python executable: {sys.executable}")
    print(f"pyuff_ustb: {pyuff_ustb.__file__}")
    print("Downloading/loading dataset...")
    uff_path = cached_download(DATA_URL, local_dir=str(data_dir))
    print(f"UFF path: {uff_path}")

    print("Reading UFF channel_data and scan...")
    uff = Uff(uff_path)
    channel_data = uff.read("/channel_data")
    scan = uff.read("/scan")

    print("Importing into vbeam setup...")
    setup = import_pyuff(channel_data, scan, frames=0)
    setup.scan = setup.scan.resize(x=96, z=128)
    print(f"Setup sizes: {setup.size()}")
    print(f"Setup spec: {setup.spec}")

    print("Building and JIT-compiling DAS beamformer...")
    beamformer = jax.jit(
        get_das_beamformer(
            setup,
            compensate_for_apodization_overlap=True,
            log_compress=True,
            scan_convert=True,
        )
    )

    print("Running beamformer...")
    result = beamformer(**setup.data).block_until_ready()
    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
    print(f"JAX devices: {jax.devices()}")

    output_path = output_dir / "picmus_plane_wave_das.png"
    plt.figure(figsize=(6, 5), dpi=160)
    plt.imshow(result.T, aspect="auto", cmap="gray", vmin=-60)
    plt.colorbar(label="dB")
    plt.title("PICMUS carotid cross - vbeam DAS")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved image: {output_path}")


if __name__ == "__main__":
    main()
