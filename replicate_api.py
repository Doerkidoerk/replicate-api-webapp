from __future__ import annotations

"""Hilfsfunktionen für Aufrufe der Replicate-API."""

import logging
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import replicate
import requests
import streamlit as st
from PIL import Image

HTTP_TIMEOUT = 45  # Sekunden

logger = logging.getLogger(__name__)


def get_replicate_client(api_token: Optional[str]) -> replicate.Client:
    """Initialisiert einen Replicate-Client oder stoppt die App bei fehlendem Token."""
    token = api_token or os.getenv("REPLICATE_API_TOKEN")
    if not token:
        st.error(
            "Fehlendes Replicate-API-Token. Bitte 'REPLICATE_API_TOKEN' in der Umgebung "
            "oder 'replicate_api_token' in st.secrets/config setzen."
        )
        st.stop()
    return replicate.Client(api_token=token)


@st.cache_data(show_spinner=False)
def list_images(folder: Path) -> List[str]:
    """Liest verfügbares Bild-Inventar im Speicherordner (einfach gecacht)."""
    supported = {".png", ".jpg", ".jpeg", ".webp"}
    if not folder.exists():
        return []
    return sorted([p.name for p in folder.iterdir() if p.suffix.lower() in supported])


def safe_unique_filename(suffix: str, prefix: str = "image") -> str:
    """Erzeugt einen eindeutigen, sicheren Dateinamen mit Zeitstempel."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return f"{prefix}_{ts}{suffix}"


def ensure_save_dir(path: Path) -> None:
    """Stellt sicher, dass der Zielordner existiert."""
    path.mkdir(parents=True, exist_ok=True)


def download_image(url: str, dest: Path) -> None:
    """Lädt ein Bild via HTTP herunter (mit Timeout & Stream) und speichert es."""
    with requests.Session() as s:
        with s.get(url, stream=True, timeout=HTTP_TIMEOUT) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)


def image_to_png_buffer(img: Image.Image) -> BytesIO:
    """Konvertiert ein PIL-Bild zu PNG in einen BytesIO-Buffer (verlustfrei, Alpha ok)."""
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf


def run_inference(
    client: replicate.Client,
    model_id: str,
    inputs: Dict[str, Union[str, int, float, bool, BytesIO]],
) -> List[str] | object:
    """Führt ein Replicate-Modell aus und behandelt Ausnahmen."""
    try:
        return client.run(model_id, input=inputs)
    except Exception as e:
        logger.exception("Replicate-Run fehlgeschlagen")
        st.error(f"Fehler beim Ausführen des Modells '{model_id}': {e}")
        return []


def save_and_show_images(urls: List[str], desired_ext: str, save_dir: Path) -> None:
    """Lädt Bilder herunter, speichert sie und zeigt sie in Streamlit an."""
    ensure_save_dir(save_dir)
    for idx, url in enumerate(urls, start=1):
        ext = f".{desired_ext.lower()}" if desired_ext else os.path.splitext(url.split("?")[0])[-1]
        if ext.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            ext = ".jpg"

        filename = safe_unique_filename(ext, prefix="generiertes_bild")
        dest = save_dir / filename

        try:
            download_image(url, dest)
            st.success(f"Bild {idx} gespeichert: '{dest.as_posix()}'")
            st.image(dest.as_posix(), caption=f"Generiertes Bild {idx}", use_column_width=True)
        except requests.HTTPError as http_err:
            st.error(f"HTTP-Fehler beim Herunterladen von Bild {idx}: {http_err}")
        except Exception as e:
            logger.exception("Download fehlgeschlagen")
            st.error(f"Fehler beim Speichern von Bild {idx}: {e}")


def save_bytes_as_image_show(
    data: bytes,
    save_dir: Path,
    prefix: str = "upscaled",
    ext: str = ".png",
) -> None:
    """Speichert Byte-Daten als Bild und zeigt das Ergebnis an."""
    ensure_save_dir(save_dir)
    try:
        img = Image.open(BytesIO(data))
        buf = image_to_png_buffer(img)
        filename = safe_unique_filename(".png", prefix=prefix)
        dest = save_dir / filename
        with dest.open("wb") as f:
            f.write(buf.getbuffer())
        st.success(f"Bild gespeichert: '{dest.as_posix()}'")
        st.image(dest.as_posix(), caption="Ergebnis", use_column_width=True)
    except Exception:
        filename = safe_unique_filename(ext, prefix=prefix)
        dest = save_dir / filename
        with dest.open("wb") as f:
            f.write(data)
        st.success(f"Datei gespeichert: '{dest.as_posix()}'")
        st.image(dest.as_posix(), caption="Ergebnis", use_column_width=True)
