"""Minimal vbeam setup without UFF.

This script shows the smallest practical path from custom arrays to a vbeam DAS
beamformer. The signal is synthetic and not physically meaningful; the goal is to
make the data contract explicit.
"""

from pathlib import Path
import os
import site
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

USER_SITE = Path(site.getusersitepackages()).resolve()
sys.path = [
    path
    for path in sys.path
    if not path or Path(path).resolve() != USER_SITE
]

import jax

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spekk import Spec

from vbeam.apodization import NoApodization
from vbeam.beamformers import get_das_beamformer
from vbeam.core import ElementGeometry, WaveData
from vbeam.data_importers import SignalForPointSetup
from vbeam.fastmath import backend_manager
from vbeam.interpolation import FastInterpLinspace
from vbeam.scan import linear_scan
from vbeam.wavefront import PlaneWavefront, ReflectedWavefront


def make_setup() -> SignalForPointSetup:
    backend_manager.active_backend = "jax"

    n_transmits = 3
    n_receivers = 16
    n_samples = 512
    pitch = 0.0003
    sampling_frequency = 20e6
    speed_of_sound = jnp.array(1540.0, dtype=jnp.float32)

    receiver_x = (jnp.arange(n_receivers) - (n_receivers - 1) / 2) * pitch
    receiver_position = jnp.stack(
        [receiver_x, jnp.zeros(n_receivers), jnp.zeros(n_receivers)], axis=-1
    )
    receiver = ElementGeometry(
        position=receiver_position,
        theta=jnp.zeros(n_receivers),
        phi=jnp.zeros(n_receivers),
    )

    sender = ElementGeometry(
        position=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        theta=jnp.array(0.0, dtype=jnp.float32),
        phi=jnp.array(0.0, dtype=jnp.float32),
    )

    azimuth = jnp.deg2rad(jnp.array([-8.0, 0.0, 8.0], dtype=jnp.float32))
    wave_data = WaveData(
        source=jnp.stack(
            [
                jnp.full(n_transmits, jnp.inf),
                jnp.zeros(n_transmits),
                jnp.full(n_transmits, jnp.inf),
            ],
            axis=-1,
        ),
        azimuth=azimuth,
        elevation=jnp.zeros(n_transmits, dtype=jnp.float32),
        t0=jnp.zeros(n_transmits, dtype=jnp.float32),
    )

    # Synthetic complex data. This is intentionally simple: it verifies the data
    # contract rather than simulating a real scatterer.
    t = jnp.arange(n_samples, dtype=jnp.float32) / sampling_frequency
    carrier = jnp.exp(1j * 2 * jnp.pi * 3e6 * t)
    envelope = jnp.exp(-((t - 14e-6) / 1.8e-6) ** 2)
    pulse = (carrier * envelope).astype(jnp.complex64)
    signal = jnp.broadcast_to(pulse, (n_transmits, n_receivers, n_samples))

    scan = linear_scan(
        jnp.linspace(-0.006, 0.006, 48),
        jnp.linspace(0.008, 0.035, 64),
    )

    spec = Spec(
        {
            "signal": ["transmits", "receivers", "signal_time"],
            "receiver": ["receivers"],
            "point_position": ["points"],
            "wave_data": ["transmits"],
        }
    )

    return SignalForPointSetup(
        sender=sender,
        point_position=None,
        receiver=receiver,
        signal=signal,
        transmitted_wavefront=PlaneWavefront(),
        reflected_wavefront=ReflectedWavefront(),
        speed_of_sound=speed_of_sound,
        wave_data=wave_data,
        interpolate=FastInterpLinspace(
            min=0.0,
            d=1 / sampling_frequency,
            n=n_samples,
        ),
        modulation_frequency=jnp.array(0.0, dtype=jnp.float32),
        apodization=NoApodization(),
        spec=spec,
        scan=scan,
    )


def main() -> None:
    setup = make_setup()
    print("setup.size():", setup.size())
    print("setup.spec:", setup.spec)
    for key, value in setup.data.items():
        print(f"{key:24s}", type(value).__name__, getattr(value, "shape", None))

    beamformer = jax.jit(
        get_das_beamformer(
            setup,
            compensate_for_apodization_overlap=False,
            log_compress=False,
            scan_convert=False,
        )
    )
    result = beamformer(**setup.data).block_until_ready()
    print("result shape:", result.shape)
    print("result dtype:", result.dtype)

    output_path = SCRIPT_DIR / "outputs" / "manual_setup_demo_abs.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(5, 4), dpi=160)
    plt.imshow(jnp.abs(result).T, aspect="auto", cmap="gray")
    plt.colorbar(label="abs")
    plt.title("Manual SignalForPointSetup demo")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print("saved:", output_path)


if __name__ == "__main__":
    main()
