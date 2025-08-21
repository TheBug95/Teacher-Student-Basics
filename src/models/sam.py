import torch.nn as nn


class SAM2(nn.Module):
    """Wrapper para SAM2.

    Requiere la librería ``segment-anything-2`` disponible en
    https://github.com/facebookresearch/segment-anything-2.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        try:
            from segment_anything_2 import sam_model_registry  # type: ignore
        except Exception as e:  # pragma: no cover - depende de librerías externas
            raise ImportError(
                "SAM2 no está instalado. Consulte la documentación del proyecto"
            ) from e
        model_type = kwargs.pop("model_type", "vit_b")
        checkpoint = kwargs.pop("checkpoint", None)
        self.model = sam_model_registry[model_type](checkpoint=checkpoint)

    def forward(self, x):
        return self.model(x)


class MedSAM2(nn.Module):
    """Wrapper para MedSAM2."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        try:
            from medsam2 import sam_model_registry  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "MedSAM2 no está instalado. Consulte la documentación del proyecto"
            ) from e
        model_type = kwargs.pop("model_type", "vit_b")
        checkpoint = kwargs.pop("checkpoint", None)
        self.model = sam_model_registry[model_type](checkpoint=checkpoint)

    def forward(self, x):
        return self.model(x)


class MobileSAM(nn.Module):
    """Wrapper para MobileSAM."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        try:
            from mobile_sam import sam_model_registry  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "MobileSAM no está instalado. Consulte la documentación del proyecto"
            ) from e
        model_type = kwargs.pop("model_type", "vit_t")
        checkpoint = kwargs.pop("checkpoint", None)
        self.model = sam_model_registry[model_type](checkpoint=checkpoint)

    def forward(self, x):
        return self.model(x)
