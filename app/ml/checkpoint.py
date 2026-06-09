"""CPU/GPU-safe loader for SteganoGAN ``.steg`` checkpoints.

A ``.steg`` file is a full ``torch.save`` of a SteganoGAN object, including a
training-only ``Trainer`` whose ``__setstate__`` rebuilds a loss and moves it to
the *training-time* device (often ``cuda``). That crashes on a CPU-only host.

Inference never needs the trainer, so we unpickle with the training classes
replaced by a harmless stub, then place the encoder/decoder on the target device.
"""
from __future__ import annotations

import logging
import pickle
import types

import torch

log = logging.getLogger(__name__)


class _Stub:
    """Absorbs pickled state without running any device-moving logic."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        pass

    def __setstate__(self, state) -> None:
        try:
            self.__dict__.update(state)
        except Exception:
            pass

    def __getattr__(self, name):
        return lambda *a, **k: None


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        # torch.amp.* — version-specific, training-only.
        if module.startswith("torch.amp."):
            return _Stub
        # The training-only Trainer forces .to(cuda) in __setstate__.
        if module == "steganogan.training.trainer" and name == "Trainer":
            return _Stub
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError) as exc:
            # Surface this loudly: a missing dependency would otherwise silently
            # stub a real object (e.g. the inference services), making encode a no-op.
            log.warning(
                "Stubbing %s.%s during checkpoint load (%s) — install missing deps if this is not a training-only class.",
                module, name, exc,
            )
            return _Stub


_compat = types.ModuleType("_veil_compat_pickle")
_compat.Unpickler = _Unpickler  # type: ignore[attr-defined]
_compat.loads = pickle.loads  # type: ignore[attr-defined]
_compat.dumps = pickle.dumps  # type: ignore[attr-defined]


def load_steganogan(path: str, device: torch.device):
    """Load a SteganoGAN checkpoint ready for inference on *device*."""
    model = torch.load(
        path,
        map_location=device,
        weights_only=False,
        pickle_module=_compat,
    )
    model.verbose = False

    # Drop transient training state.
    for attr, val in (
        ("encoder_decoder_optimizer", None),
        ("fit_metrics", None),
        ("history", []),
    ):
        try:
            setattr(model, attr, val)
        except Exception:
            pass

    # Place inference modules on the device and switch to eval.
    for attr in ("encoder", "decoder", "critic"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        if hasattr(sub, "upgrade_legacy"):
            try:
                sub.upgrade_legacy()
            except Exception:
                pass
        sub.to(device)
        sub.eval()

    model.device = device
    try:
        model.gpu = device.type != "cpu"
    except Exception:
        pass

    # Sync device on any serialised sub-services that cache it.
    for svc in ("_payload_factory", "_encoder_svc", "_decoder_svc", "_trainer"):
        obj = getattr(model, svc, None)
        if obj is not None and hasattr(obj, "device"):
            try:
                obj.device = device
            except Exception:
                pass

    return model
