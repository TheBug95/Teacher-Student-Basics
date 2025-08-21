"""
Herramientas para adaptar modelos SAM2 y MedSAM2 con LoRA o QLoRA.

Este módulo usa la librería `peft` para añadir adaptadores de bajo
rango a los modelos de segmentación.  Utilice `load_model` para obtener
un modelo con los adaptadores deseados.
"""

from typing import Iterable, Optional
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def _apply_lora(
    model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[Iterable[str]] = None,
    qlora: bool = False,
):
    """Aplicar configuración LoRA/QLoRA a un modelo."""
    if target_modules is None:
        target_modules = ("q_proj", "v_proj")
    if qlora:
        model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
    )
    return get_peft_model(model, config)


def load_model(model_name: str, qlora: bool = False, **kwargs):
    """Cargar un modelo SAM2 o MedSAM2 con adaptadores LoRA/QLoRA.

    Parameters
    ----------
    model_name: str
        Identificador de Hugging Face del modelo base, por ejemplo
        ``facebook/sam2-huge`` o ``UNMC-Medical-Seg/SAM2-medical``.
    qlora: bool, optional
        Si es ``True`` se usa QLoRA (4-bit) en lugar de LoRA estándar.
    **kwargs:
        Parámetros adicionales pasados a ``_apply_lora``.
    """
    model = AutoModel.from_pretrained(model_name)
    model = _apply_lora(model, qlora=qlora, **kwargs)
    return model
