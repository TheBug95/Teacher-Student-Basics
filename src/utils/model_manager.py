"""Gestor de descargas para modelos externos.

Permite instalar las dependencias oficiales de cada familia
(SAM2, MobileSAM y MedSAM2) y descargar los *checkpoints*
correspondientes en una carpeta local.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import re
from typing import Dict, Tuple

from huggingface_hub import hf_hub_download


class ModelManager:
    """Descarga y gestiona modelos externos.

    Parameters
    ----------
    root_dir:
        Carpeta donde se almacenarán los pesos descargados.
    """

    _MODELS: Dict[str, Dict[str, Dict[str, Tuple[str, str]]]] = {
        # -----------------------------------------------------------------
        "sam2": {
            "pip": "git+https://github.com/facebookresearch/segment-anything-2.git",
            "variants": {
                # SAM 2.1 -------------------------------------------------
                "sam2.1_hiera_tiny": (
                    "url",
                    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
                ),
                "sam2.1_hiera_small": (
                    "url",
                    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
                ),
                "sam2.1_hiera_base_plus": (
                    "url",
                    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
                ),
                "sam2.1_hiera_large": (
                    "url",
                    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
                ),
                # SAM 2 ---------------------------------------------------
                "sam2_hiera_tiny": (
                    "url",
                    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
                ),
                "sam2_hiera_small": (
                    "url",
                    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
                ),
                "sam2_hiera_base_plus": (
                    "url",
                    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
                ),
                "sam2_hiera_large": (
                    "url",
                    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
                ),
            },
        },
        # -----------------------------------------------------------------
        "mobilesam": {
            "pip": "git+https://github.com/ChaoningZhang/MobileSAM.git",
            "variants": {
                "vit_t": ("dhkim2810/MobileSAM", "mobile_sam.pt"),
            },
        },
        # -----------------------------------------------------------------
        "medsam2": {
            "pip": "git+https://github.com/bowang-lab/MedSAM2.git",
            "variants": {
                "latest": ("wanglab/MedSAM2", "MedSAM2_latest.pt"),
            },
        },
    }

    # -----------------------------------------------------------------
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    # ----------------------------- API pública ------------------------
    def install_repo(self, family: str) -> None:
        """Instala el repositorio oficial de una familia de modelos."""
        pkg = self._MODELS[family]["pip"]
        if not self._pkg_installed(pkg):
            print(f"Instalando {family}…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        else:
            print(f"{family} ya estaba instalado.")

    def download_variant(self, family: str, variant: str, force: bool = False) -> Path:
        """Descarga un *checkpoint* específico.

        Parameters
        ----------
        family:
            Grupo de modelos (sam2, mobilesam, medsam2).
        variant:
            Variante concreta dentro de la familia.
        force:
            Si ``True``, fuerza la descarga aunque el archivo exista.
        """
        repo_or_url, filename = self._MODELS[family]["variants"][variant]
        dest = self.root / Path(filename).name

        if dest.exists() and not force:
            print(f"{dest.name} ya existe → se omite descarga.")
            return dest

        if repo_or_url == "url" or repo_or_url.startswith("http"):
            import requests

            url = filename if repo_or_url == "url" else repo_or_url
            print(f"Descargando desde URL directa:\n  {url}")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        else:
            print(f"Descargando desde Hugging Face ({repo_or_url})…")
            dest_path = hf_hub_download(
                repo_id=repo_or_url,
                filename=filename,
                cache_dir=str(self.root),
                force_download=force,
            )
            dest = Path(dest_path)

        return dest

    def setup(self, family: str, variant: str, install: bool = True, force: bool = False) -> Path:
        """Instala dependencias y descarga el modelo."""
        if install:
            self.install_repo(family)
        return self.download_variant(family, variant, force)

    def list_supported(self) -> None:
        """Muestra las familias y variantes disponibles."""
        for fam, info in self._MODELS.items():
            print(f"{fam}: {', '.join(info['variants'])}")

    # ------------------------ utilidades internas ---------------------
    @staticmethod
    def _pkg_installed(pip_spec: str) -> bool:
        import pkg_resources

        name = re.sub(r".*/|\.git$", "", pip_spec)
        try:
            pkg_resources.get_distribution(name)
            return True
        except pkg_resources.DistributionNotFound:
            return False


__all__ = ["ModelManager"]

if __name__ == "__main__":
    mgr = ModelManager("checkpoints")
    mgr.list_supported()
