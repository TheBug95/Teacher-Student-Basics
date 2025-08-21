# Teacher-Student-Basics
Desarrollo de el framework teacher student basico para comprenderlo a detalle

## LoRA/QLoRA para SAM2 y MedSAM2

El proyecto ahora incluye utilidades para adaptar los modelos **SAM2** y **MedSAM2** con técnicas de ajuste eficiente. El módulo `src/models/sam_lora.py` permite cargar estos modelos aplicando [LoRA](https://arxiv.org/abs/2106.09685) o su variante cuantizada QLoRA para reducir el consumo de memoria y cómputo durante el entrenamiento.

```python
from src.models.sam_lora import load_model

# LoRA tradicional
model = load_model("facebook/sam2-huge")

# QLoRA (4-bit)
model = load_model("facebook/sam2-huge", qlora=True)
```

Los parámetros `r`, `alpha` y `dropout` pueden ajustarse según las necesidades del entrenamiento.

