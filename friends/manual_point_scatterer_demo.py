"""Manual vbeam setup with a physically meaningful point scatterer.

This example bypasses UFF and builds a SignalForPointSetup directly. Unlike
manual_setup_demo.py, the synthetic signal is generated from one known scatterer
position, so the beamformed image should peak near that position.
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


def make_setup() -> tuple[SignalForPointSetup, jnp.ndarray]:
    backend_manager.active_backend = "jax"

    n_transmits = 5
    n_receivers = 32
    n_samples = 2048
    pitch = 0.0003
    sampling_frequency = 40e6
    speed_of_sound = jnp.array(1540.0, dtype=jnp.float32)
    center_frequency = 5e6

    scatterer = jnp.array([0.0015, 0.0, 0.0220], dtype=jnp.float32)

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

    azimuth = jnp.deg2rad(jnp.linspace(-8.0, 8.0, n_transmits))
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

    scatterer_x = scatterer[0]
    scatterer_z = scatterer[2]
    tx_distance = (
        scatterer_x * jnp.sin(azimuth)
        + scatterer_z * jnp.cos(azimuth)
    )
    rx_distance = jnp.linalg.norm(scatterer[None, :] - receiver_position, axis=-1)
    delay = (tx_distance[:, None] + rx_distance[None, :]) / speed_of_sound

    t = jnp.arange(n_samples, dtype=jnp.float32) / sampling_frequency
    pulse_width = 0.45e-6
    tau = t[None, None, :] - delay[:, :, None]
    envelope = jnp.exp(-((tau / pulse_width) ** 2))
    carrier = jnp.exp(1j * 2 * jnp.pi * center_frequency * tau)
    signal = (envelope * carrier).astype(jnp.complex64)

    scan = linear_scan(
        jnp.linspace(-0.006, 0.006, 96),
        jnp.linspace(0.010, 0.035, 128),
    )

    spec = Spec(
        {
            "signal": ["transmits", "receivers", "signal_time"],
            "receiver": ["receivers"],
            "point_position": ["points"],
            "wave_data": ["transmits"],
        }
    )

    setup = SignalForPointSetup(
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
    return setup, scatterer


def main() -> None:
    setup, scatterer = make_setup()
    print("setup.size():", setup.size())
    print("setup.spec:", setup.spec)

    beamformer = jax.jit(
        get_das_beamformer(
            setup,
            compensate_for_apodization_overlap=False,
            log_compress=False,
            scan_convert=False,
        )
    )
    result = beamformer(**setup.data).block_until_ready()
    abs_result = jnp.abs(result)

    peak_flat = int(jnp.argmax(abs_result))
    peak_ix, peak_iz = jnp.unravel_index(peak_flat, abs_result.shape)
    peak_x = setup.scan.x[int(peak_ix)]
    peak_z = setup.scan.z[int(peak_iz)]
    print("result shape:", result.shape)
    print("expected scatterer x/z (m):", float(scatterer[0]), float(scatterer[2]))
    print("beamformed peak x/z (m):", float(peak_x), float(peak_z))
    print(
        "absolute localization error (mm):",
        float(
            jnp.linalg.norm(
                jnp.array([peak_x, peak_z]) - scatterer[jnp.array([0, 2])]
            )
            * 1e3
        ),
    )

    output_path = SCRIPT_DIR / "outputs" / "manual_point_scatterer_abs.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(5, 4), dpi=160)
    extent = [
        float(setup.scan.x[0] * 1e3),
        float(setup.scan.x[-1] * 1e3),
        float(setup.scan.z[-1] * 1e3),
        float(setup.scan.z[0] * 1e3),
    ]
    plt.imshow(abs_result.T, aspect="auto", cmap="gray", extent=extent)
    plt.scatter(
        [float(scatterer[0] * 1e3)],
        [float(scatterer[2] * 1e3)],
        c="red",
        s=18,
        marker="x",
        label="true scatterer",
    )
    plt.xlabel("x (mm)")
    plt.ylabel("z (mm)")
    plt.title("Manual point scatterer DAS")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print("saved:", output_path)


if __name__ == "__main__":
    main()
