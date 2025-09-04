# app.py
"""
Streamlit Bildgenerator, Upscaler & Galerie (Replicate + Auth, kompatibel zu streamlit-authenticator 0.4.x)
-----------------------------------------------------------------------------------------------------------
- Login nach 0.4.x-Pattern: authenticator.login() (ohne positionsbasierte Argumente) + Session-State
- Erst-Login-Flow: admin/admin mit Pflicht zum Passwortwechsel (YAML-Variante)
- Optionaler Bild-Upload / Galerie (Upscaler)
- Robustes Speichern, Anzeigen, Downloaden & Löschen
"""

from __future__ import annotations

import logging
import mimetypes
import os
import secrets as pysecrets
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import replicate
import streamlit as st
import streamlit_authenticator as stauth
import toml
import yaml
from PIL import Image
from yaml.loader import SafeLoader

# eigene Hilfsfunktionen (ohne get_replicate_client importieren!)
from replicate_api import (
    ensure_save_dir,
    image_to_png_buffer,
    list_images,
    run_inference,
    save_and_show_images,
    save_bytes_as_image_show,
)

# =========================
# Konstante Pfade & Limits
# =========================
APP_TITLE = "Flux – Bildgenerator & Upscaler"
# Basispfad der App (unabhängig vom aktuellen Arbeitsverzeichnis)
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "KI-Bilder"
CONFIG_PATH = BASE_DIR / ".streamlit" / "config.yaml"
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# Modelle
MODELS: Dict[str, str] = {
    "Flux Dev": "black-forest-labs/flux-dev",
    "Flux 1.1 Pro": "black-forest-labs/flux-1.1-pro",
}
UPSCALER_MODEL_ID = "google/upscaler"  # Replicate Upscaler

# Standard-Account für Erst-Login
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin"

# ==============
# Logging-Setup
# ==============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")


# =================
# Hilfs-Datamodelle
# =================
@dataclass
class AppConfig:
    credentials: dict
    cookie_name: str
    cookie_key: str
    cookie_expiry_days: int
    replicate_api_token: Optional[str] = None
    source: str = "yaml"  # "yaml" oder "secrets"


# ===========
# YAML I/O
# ===========
def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=SafeLoader) or {}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


# ============================================
# Passwort-Hashing (0.4.x-tauglich)
# ============================================
def hash_password(password: str) -> str:
    try:
        if hasattr(stauth.Hasher, "hash"):  # 0.4.x
            return stauth.Hasher.hash(password)
        return stauth.Hasher([password]).generate()[0]  # legacy
    except Exception:
        try:
            import bcrypt
            salt = bcrypt.gensalt(rounds=12)
            return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
        except Exception as e:
            st.error(
                "Passworthashing fehlgeschlagen. Installiere `streamlit-authenticator` oder `bcrypt`."
                f" Technisches Detail: {e}"
            )
            st.stop()
            raise


# ============================================
# Erst-Login-Bootstrap
# ============================================
def _load_secrets() -> Mapping[str, Any]:
    """Load secrets by merging Streamlit's secrets with the local file.

    Streamlit does not automatically reload ``st.secrets`` after writing to
    ``secrets.toml``. To pick up a newly saved token without restarting the
    server, we read the file directly and merge it on top of ``st.secrets``.
    """

    merged: Dict[str, Any] = {}

    # Start with what's already available in st.secrets (if any)
    try:
        merged.update(dict(st.secrets))  # type: ignore[arg-type]
    except Exception:
        pass

    # Overlay anything from the secrets file so recent saves take effect
    if SECRETS_PATH.exists():
        try:
            merged.update(toml.load(SECRETS_PATH))
        except Exception:
            pass

    return merged


def save_replicate_api_token(token: str) -> None:
    data: Dict[str, Any] = {}
    if SECRETS_PATH.exists():
        try:
            data = toml.load(SECRETS_PATH)
        except Exception:
            data = {}
    data["replicate_api_token"] = token
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SECRETS_PATH.open("w", encoding="utf-8") as f:
        toml.dump(data, f)


def ensure_bootstrap_files() -> None:
    if not SECRETS_PATH.exists():
        SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SECRETS_PATH.write_text("# hier können Secrets eingetragen werden\n", encoding="utf-8")

    if CONFIG_PATH.exists():
        return

    hashed = hash_password(DEFAULT_PASSWORD)
    cfg = {
        "cookie": {
            "name": "auth",
            "key": pysecrets.token_hex(16),
            "expiry_days": 30,
        },
        "credentials": {
            "usernames": {
                DEFAULT_USERNAME: {
                    "name": "Admin",
                    "email": f"{DEFAULT_USERNAME}@example.com",
                    "password": hashed,
                    "must_change_credentials": True,
                }
            }
        },
    }
    _write_yaml(CONFIG_PATH, cfg)
    logger.info("Bootstrap: admin/admin angelegt; must_change_credentials=True")


def must_change_credentials_from_yaml(username: str) -> bool:
    if not CONFIG_PATH.exists():
        return False
    cfg = _read_yaml(CONFIG_PATH)
    try:
        return bool(
            cfg["credentials"]["usernames"][username].get("must_change_credentials", False)
        )
    except Exception:
        return False


def set_user_password_in_yaml(username: str, new_password: str) -> None:
    cfg = _read_yaml(CONFIG_PATH)
    if "credentials" not in cfg:
        st.error("Ungültige YAML-Struktur: 'credentials' fehlt.")
        st.stop()
    usernames = cfg["credentials"].setdefault("usernames", {})
    user = usernames.get(username)
    if not user:
        st.error("Benutzer nicht gefunden.")
        st.stop()
    user["password"] = hash_password(new_password)
    user["must_change_credentials"] = False
    usernames[username] = user
    _write_yaml(CONFIG_PATH, cfg)


# ===========
# Utilities
# ===========
def load_config() -> AppConfig:
    secrets = _load_secrets()
    if "credentials" in secrets and "cookie" in secrets:
        cookie = secrets["cookie"]
        token = secrets.get("replicate_api_token") or os.getenv("REPLICATE_API_TOKEN")
        return AppConfig(
            credentials=dict(secrets["credentials"]),
            cookie_name=str(cookie.get("name", "app_auth")),
            cookie_key=str(cookie.get("key", "supersecret")),
            cookie_expiry_days=int(cookie.get("expiry_days", 30)),
            replicate_api_token=token,
            source="secrets",
        )

    if not CONFIG_PATH.exists():
        st.error(f"Konfigurationsdatei '{CONFIG_PATH.as_posix()}' nicht gefunden.")
        st.stop()

    try:
        cfg = _read_yaml(CONFIG_PATH)
    except yaml.YAMLError as e:
        st.error(f"Fehler beim Laden der Konfiguration: {e}")
        st.stop()
        raise

    try:
        cookie_cfg = cfg["cookie"]
        token = cfg.get("replicate_api_token") or os.getenv("REPLICATE_API_TOKEN")
        return AppConfig(
            credentials=cfg["credentials"],
            cookie_name=cookie_cfg["name"],
            cookie_key=cookie_cfg["key"],
            cookie_expiry_days=int(cookie_cfg["expiry_days"]),
            replicate_api_token=token,
            source="yaml",
        )
    except KeyError as e:
        st.error(f"Fehlendes Konfigurationsfeld: {e}")
        st.stop()
        raise


@st.cache_resource(show_spinner=False)
def get_replicate_client(api_token: Optional[str]) -> replicate.Client:
    token = api_token or os.getenv("REPLICATE_API_TOKEN")
    if not token:
        st.error(
            "Fehlendes Replicate-API-Token. Setze 'REPLICATE_API_TOKEN' in der Umgebung "
            "oder 'replicate_api_token' in st.secrets/config."
        )
        st.stop()
    return replicate.Client(api_token=token)


def guess_mime(filename: str) -> str:
    mt, _ = mimetypes.guess_type(filename)
    return mt or "application/octet-stream"


# =====================
# Authentifizierung (0.4.x-konformes Pattern)
# =====================
def do_authentication(cfg: AppConfig) -> Tuple[Optional[str], Optional[str], Optional[bool], Any]:
    """
    Gemäß 0.4.x-Doku:
      - login() ohne positionsbasierte Argumente (Standard location='main')
      - Status/Name/Username aus st.session_state lesen
      - logout() per Keywords
    Referenz: README & API-Doku.   [oai_citation:4‡GitHub](https://github.com/mkhorasani/Streamlit-Authenticator) [oai_citation:5‡streamlit-authenticator.readthedocs.io](https://streamlit-authenticator.readthedocs.io/?utm_source=chatgpt.com)
    """
    authenticator = stauth.Authenticate(
        credentials=cfg.credentials,
        cookie_name=cfg.cookie_name,
        cookie_key=cfg.cookie_key,
        cookie_expiry_days=cfg.cookie_expiry_days,
    )

    # Login-Widget rendern (Standard: main). KEIN Titel als 1. Argument!
    try:
        authenticator.login()  # location='main' (Default)
    except Exception as e:
        # Falls hier ein Fehler auftritt, ist fast immer noch irgendwo ein alter Aufruf im Projekt.
        st.error(f"Login fehlgeschlagen: {e}")
        return None, None, None, authenticator

    name = st.session_state.get("name")
    username = st.session_state.get("username")
    authentication_status = st.session_state.get("authentication_status")

    if authentication_status is False:
        st.error("Benutzername oder Passwort ist falsch.")
        st.session_state["authentication_status"] = None
        st.session_state.pop("username", None)
        st.session_state.pop("name", None)
    elif authentication_status is None:
        st.info("Bitte Benutzernamen und Passwort eingeben.")

    if authentication_status:
        # Logout-Button in der Sidebar
        authenticator.logout(button_name="Logout", location="sidebar", key="Logout")
        st.sidebar.success(f"Willkommen, {name or username}!")

    return name, username, authentication_status, authenticator


# =======================
# Generator-Formulare
# =======================
def render_flux_dev_form() -> Dict[str, Union[str, float, int, bool]]:
    st.header("Parameter für Flux Dev")

    prompt = st.text_area(
        "Prompt:",
        value='black forest gateau cake spelling out the words "FLUX DEV", tasty, food photography, dynamic shot',
        height=100,
        help="Kurze, präzise Beschreibung – je konkreter, desto konsistenter.",
    )
    if not prompt.strip():
        st.error("Der Prompt darf nicht leer sein.")
        st.stop()

    go_fast = st.checkbox("Go Fast", value=True)
    guidance = st.slider("Guidance Scale:", 0.0, 10.0, 3.5, 0.1)
    megapixels_str = st.selectbox("Megapixels:", options=["0.25", "1"], index=1)
    megapixels = float(megapixels_str)

    num_outputs = st.slider("Anzahl Ausgaben:", 1, 4, 1, 1)
    aspect_ratio = st.selectbox("Aspect Ratio:", ["1:1", "16:9", "4:3", "3:2", "2:3", "9:16"], index=0)
    output_format = st.selectbox("Output Format:", ["webp", "png", "jpg"], index=0)
    output_quality = st.slider("Output Quality:", 10, 100, 80, 1)
    prompt_strength = st.slider("Prompt Strength:", 0.0, 1.0, 0.8, 0.05)
    num_inference_steps = st.slider("Inference Schritte:", 1, 50, 28, 1)
    disable_safety_checker = st.checkbox("Sicherheitsprüfer deaktivieren", value=False)

    return {
        "prompt": prompt,
        "go_fast": bool(go_fast),
        "guidance": float(guidance),
        "megapixels": float(megapixels),
        "num_outputs": int(num_outputs),
        "aspect_ratio": aspect_ratio,
        "output_format": output_format,
        "output_quality": int(output_quality),
        "prompt_strength": float(prompt_strength),
        "num_inference_steps": int(num_inference_steps),
        "disable_safety_checker": bool(disable_safety_checker),
    }


def render_flux_pro_form() -> Dict[str, Union[str, int, bool]]:
    st.header("Parameter für Flux 1.1 Pro")

    prompt = st.text_area(
        "Prompt:",
        value='black forest gateau cake spelling out the words "FLUX 1.1 Pro", tasty, food photography',
        height=100,
    )
    if not prompt.strip():
        st.error("Der Prompt darf nicht leer sein.")
        st.stop()

    seed = st.number_input("Seed:", min_value=0, value=0, step=1)
    aspect_ratio = st.selectbox(
        "Aspect Ratio:", ["custom", "1:1", "16:9", "2:3", "3:2", "4:5", "5:4", "9:16", "3:4", "4:3"], index=1
    )

    width: Optional[int] = None
    height: Optional[int] = None
    if aspect_ratio == "custom":
        width = st.number_input("Width:", 256, 1440, 512, 16, help="Breite (Vielfaches von 16).")
        if width % 16 != 0:
            st.warning("Width wird auf das nächste Vielfache von 16 gerundet.")
            width = (width + 15) // 16 * 16

        height = st.number_input("Height:", 256, 1440, 512, 16, help="Höhe (Vielfaches von 16).")
        if height % 16 != 0:
            st.warning("Height wird auf das nächste Vielfache von 16 gerundet.")
            height = (height + 15) // 16 * 16

    output_format = st.selectbox("Output Format:", ["webp", "jpg", "png"], index=0)
    output_quality = st.slider("Output Quality:", 0, 100, 80, 1)
    safety_tolerance = st.slider("Safety Tolerance:", 1, 5, 2, 1)
    prompt_upsampling = st.checkbox("Prompt Upsampling", value=True)

    inputs: Dict[str, Union[str, int, bool]] = {
        "prompt": prompt,
        "seed": int(seed),
        "aspect_ratio": aspect_ratio,
        "output_format": output_format,
        "output_quality": int(output_quality),
        "safety_tolerance": int(safety_tolerance),
        "prompt_upsampling": bool(prompt_upsampling),
    }
    if aspect_ratio == "custom":
        inputs["width"] = int(width or 512)
        inputs["height"] = int(height or 512)
    return inputs


def handle_optional_image_upload() -> Optional[BytesIO]:
    uploaded = st.file_uploader("Optional: Bild hochladen", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        return None

    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("Die hochgeladene Datei überschreitet die maximale Größe von 50 MB.")
        st.stop()

    try:
        image = Image.open(uploaded)
        rgb = image.convert("RGB")
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=90, optimize=True)
        buf.seek(0)
        st.success(f"Bild '{uploaded.name}' wurde erfolgreich vorbereitet.")
        return buf
    except Exception as e:
        st.error(f"Fehler beim Verarbeiten der hochgeladenen Datei: {e}")
        return None


# ==========
# Seiten-UI
# ==========
def render_generator(client: replicate.Client) -> None:
    st.title("Bildgenerator")

    model_name = st.selectbox("Wähle das Bildgenerierungsmodell:", options=list(MODELS.keys()))
    model_id = MODELS[model_name]

    if model_name == "Flux Dev":
        inputs = render_flux_dev_form()
    else:
        inputs = render_flux_pro_form()

    uploaded = handle_optional_image_upload()
    if uploaded is not None:
        inputs["image"] = uploaded  # Replicate akzeptiert file-like Objekte

    if st.button("Bild generieren"):
        with st.spinner("Generiere Bild(er)…"):
            output = run_inference(client, model_id, inputs)
            if isinstance(output, list):
                urls = [str(u) for u in output]
                if urls:
                    desired_ext = str(inputs.get("output_format", "jpg"))
                    save_and_show_images(urls, desired_ext, SAVE_DIR)
            elif isinstance(output, str):
                save_and_show_images([output], str(inputs.get("output_format", "jpg")), SAVE_DIR)
            else:
                st.error("Unerwartetes Ausgabeformat der Replicate API für den Generator.")


def render_upscaler(client: replicate.Client) -> None:
    st.title("Upscaler (google/upscaler)")
    st.write("Skaliere Bilder hoch (x2/x4) mit Googles Upscaler über Replicate.")

    source = st.radio("Quelle wählen", ["Upload", "Aus Galerie"], horizontal=True)

    img_buffer: Optional[BytesIO] = None
    selected_name: Optional[str] = None

    if source == "Upload":
        st.caption("Lade ein Bild hoch (PNG/JPG/WebP, max. 50 MB).")
        uploaded = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg", "webp"], key="upscaler_uploader")
        if uploaded:
            if uploaded.size > MAX_UPLOAD_BYTES:
                st.error("Die hochgeladene Datei überschreitet die maximale Größe von 50 MB.")
                st.stop()
            try:
                img = Image.open(uploaded)
                img_buffer = image_to_png_buffer(img)
                selected_name = uploaded.name
                st.success(f"Bild '{uploaded.name}' wurde vorbereitet.")
                st.image(img, caption="Ausgewähltes Bild", use_column_width=True)
            except Exception as e:
                st.error(f"Fehler beim Verarbeiten des Uploads: {e}")
                return
    else:
        images = list_images(SAVE_DIR)
        if not images:
            st.info("Keine Bilder in der Galerie gefunden. Bitte zuerst ein Bild generieren oder hochladen.")
            return
        selected_name = st.selectbox("Bild aus Galerie wählen", options=images)
        if selected_name:
            path = (SAVE_DIR / selected_name).resolve()
            if not (path.exists() and path.is_file() and path.parent == SAVE_DIR.resolve()):
                st.error("Ungültige Bildauswahl.")
                return
            try:
                img = Image.open(path)
                img_buffer = image_to_png_buffer(img)
                st.image(img, caption=f"Ausgewählt: {selected_name}", use_column_width=True)
            except Exception as e:
                st.error(f"Fehler beim Laden des Bildes: {e}")
                return

    factor = st.selectbox("Upscale-Faktor", options=["x2", "x4"], index=1, help="Empfohlen: x4 für maximale Details.")

    if st.button("Upscale starten"):
        if img_buffer is None:
            st.warning("Bitte zunächst ein Bild auswählen oder hochladen.")
            return

        inputs = {"image": img_buffer, "upscale_factor": factor}

        with st.spinner(f"Upscaling läuft ({factor})…"):
            output = run_inference(client, UPSCALER_MODEL_ID, inputs)
            try:
                read_fn = getattr(output, "read", None)
                if callable(read_fn):
                    data: bytes = read_fn()
                    save_bytes_as_image_show(data, SAVE_DIR, prefix="upscaled", ext=".png")
                    return
                if isinstance(output, str):
                    save_and_show_images([output], "png", SAVE_DIR)
                    return
                if isinstance(output, list) and output:
                    first = output[0]
                    if isinstance(first, str):
                        save_and_show_images([first], "png", SAVE_DIR)
                        return
                    read_fn0 = getattr(first, "read", None)
                    if callable(read_fn0):
                        data0: bytes = read_fn0()
                        save_bytes_as_image_show(data0, SAVE_DIR, prefix="upscaled", ext=".png")
                        return
                url_fn = getattr(output, "url", None)
                if callable(url_fn):
                    url_val = url_fn()
                    if isinstance(url_val, str):
                        save_and_show_images([url_val], "png", SAVE_DIR)
                        return
                st.error("Unerwartetes Ausgabeformat der Replicate API für den Upscaler.")
            except Exception as e:
                logger.exception("Verarbeitung des Upscale-Ergebnisses fehlgeschlagen")
                st.error(f"Fehler beim Verarbeiten des Upscale-Ergebnisses: {e}")


def render_gallery() -> None:
    st.title("Galerie der gespeicherten Bilder")
    ensure_save_dir(SAVE_DIR)

    images = list_images(SAVE_DIR)
    if not images:
        st.info("Keine Bilder im Ordner 'KI-Bilder' gefunden.")
        return

    selected_image = st.session_state.get("selected_image")
    if selected_image:
        image_path = (SAVE_DIR / selected_image).resolve()
        if image_path.exists() and image_path.is_file() and image_path.parent == SAVE_DIR.resolve():
            st.markdown("---")
            st.header(f"Vorschau: {selected_image}")
            st.image(image_path.as_posix(), use_column_width=True)

            mime = guess_mime(selected_image)
            with image_path.open("rb") as f:
                data = f.read()
            st.download_button("Download", data=data, file_name=selected_image, mime=mime)

            if st.button("Löschen", key=f"delete_{selected_image}"):
                try:
                    image_path.unlink(missing_ok=False)
                    st.success(f"Bild '{selected_image}' wurde gelöscht.")
                    st.session_state.pop("selected_image", None)
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()  # type: ignore[attr-defined]
                except Exception as e:
                    st.error(f"Fehler beim Löschen: {e}")

    st.markdown("---")
    st.header("Gespeicherte Bilder")
    cols = st.columns(3)
    for idx, fname in enumerate(images):
        col = cols[idx % 3]
        with col:
            path = SAVE_DIR / fname
            st.image(path.as_posix(), caption=fname, use_column_width=True, clamp=True)
            if st.button("Ansehen", key=f"view_{fname}"):
                st.session_state["selected_image"] = fname
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()  # type: ignore[attr-defined]


# ========================================
# Zwingender Wechsel von Benutzer & Passwort
# ========================================
def enforce_initial_credentials_change_ui(cfg: AppConfig, username: Optional[str]) -> None:
    if not username:
        return

    must_change = False
    if cfg.source == "yaml":
        must_change = must_change_credentials_from_yaml(username)
    else:
        secrets = _load_secrets()
        try:
            must_change = bool(
                secrets["credentials"]["usernames"][username].get(
                    "must_change_credentials", False
                )
            )
        except Exception:
            must_change = False

    if not must_change:
        return

    st.markdown("## 🔐 Passwort ändern")
    st.warning("Erster Login mit Standard-Passwort erkannt. Bitte neues Passwort setzen.")

    new1 = st.text_input("Neues Passwort", type="password")
    new2 = st.text_input("Neues Passwort bestätigen", type="password")
    min_len = 8
    if st.button("Speichern"):
        if not new1 or not new2:
            st.error("Bitte alle Felder ausfüllen.")
        elif new1 != new2:
            st.error("Die Passwörter stimmen nicht überein.")
        elif len(new1) < min_len:
            st.error(f"Das Passwort muss mindestens {min_len} Zeichen haben.")
        else:
            if cfg.source == "yaml":
                try:
                    set_user_password_in_yaml(username, new1)
                    st.success("Passwort gespeichert. Bitte neu anmelden.")
                    st.balloons()
                    st.stop()
                except Exception as e:
                    st.error(f"Fehler beim Schreiben in die YAML: {e}")
            else:
                hashed = hash_password(new1)
                st.warning("Aktuell laufen die Logins über `st.secrets`. Bitte manuell aktualisieren & neu starten.")
                st.code(
                    f"""# In deinen Secrets/YAML einfügen:
credentials:
  usernames:
    {username}:
      name: "{username}"
      email: "{username}@example.com"
      password: "{hashed}"
      must_change_credentials: false
""",
                    language="yaml",
                )
                st.stop()

    st.stop()


# =====
# Main
# =====
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎨", layout="wide")

    logger.info("streamlit-authenticator version: %s", getattr(stauth, "__version__", "unknown"))

    ensure_bootstrap_files()
    cfg = load_config()

    # Auth
    name, username, authentication_status, authenticator = do_authentication(cfg)
    if not authentication_status:
        st.write("Bitte einloggen, um auf den Inhalt zuzugreifen.")
        st.stop()

    # Passwortwechsel erzwingen (nur YAML-Variante automatisiert)
    enforce_initial_credentials_change_ui(cfg, username or st.session_state.get("username"))

    # Replicate-Token sicherstellen
    if not cfg.replicate_api_token:
        st.markdown("## Replicate API Token erforderlich")
        st.info("Bitte gib deinen REPLICATE_API_TOKEN ein, um die App nutzen zu können.")
        token_val = st.text_input("REPLICATE_API_TOKEN", type="password")
        if st.button("Speichern"):
            if not token_val.strip():
                st.error("Token darf nicht leer sein.")
            else:
                save_replicate_api_token(token_val.strip())
                st.success("Token gespeichert. Seite wird neu geladen...")
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()  # type: ignore[attr-defined]
        st.stop()

    client = get_replicate_client(cfg.replicate_api_token)

    # Menü
    st.sidebar.markdown("## Menü")
    menu = st.sidebar.selectbox("Ansicht wählen", ["Bildgenerator", "Upscaler", "Galerie"], index=0)

    ensure_save_dir(SAVE_DIR)

    if menu == "Bildgenerator":
        render_generator(client)
    elif menu == "Upscaler":
        render_upscaler(client)
    else:
        render_gallery()


if __name__ == "__main__":
    main()
