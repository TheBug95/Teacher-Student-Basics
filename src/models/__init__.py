"""Colección de modelos de segmentación."""

from typing import Any

from .sam_lora import load_model


def get_model(name: str, **kwargs: Any):
    """Devuelve una instancia de modelo dada su cadena de nombre.

    Parameters
    ----------
    name: str
        Nombre del modelo. Opciones soportadas: ``unet``, ``sam2``,
        ``medsam2`` y ``mobilesam``.
    **kwargs: Any
        Argumentos adicionales pasados al constructor del modelo.
    """
    name = name.lower()
    if name == "unet":
        from .unet import UNet
        return UNet(**kwargs)
    if name == "sam2":
        from .sam import SAM2
        return SAM2(**kwargs)
    if name == "medsam2":
        from .sam import MedSAM2
        return MedSAM2(**kwargs)
    if name == "mobilesam":
        from .sam import MobileSAM
        return MobileSAM(**kwargs)
    raise ValueError(f"Modelo desconocido: {name}")


__all__ = ["get_model", "load_model"]
