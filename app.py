# app.py
"""
Streamlit Bildgenerator, Upscaler & Galerie (Replicate + Auth)
--------------------------------------------------------------
- Sichere Konfiguration (YAML oder st.secrets), robustes Login via streamlit_authenticator
- Erst-Login-Flow: flux/flux + erzwungene Änderung von Benutzername und Passwort (YAML-Variante)
- Optionaler Bild-Upload oder Auswahl aus Galerie (für Upscaler)
- Zuverlässiges Speichern, Anzeigen, Downloaden & Löschen
- Sauberes Logging, PEP8, Typ-Hints & Docstrings
"""

from __future__ import annotations

import logging
import mimetypes
import os
import secrets as pysecrets
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import replicate
import requests
import streamlit as st
import streamlit_authenticator as stauth
import toml
import yaml
from PIL import Image
from yaml.loader import SafeLoader

# =========================
# Konstante Pfade & Limits
# =========================
APP_TITLE = "Flux – Bildgenerator & Upscaler"
SAVE_DIR = Path("./KI-Bilder")
CONFIG_PATH = Path(".streamlit/config.yaml")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
HTTP_TIMEOUT = 45  # Sekunden

# Modelle
MODELS: Dict[str, str] = {
    "Flux Dev": "black-forest-labs/flux-dev",
    "Flux 1.1 Pro": "black-forest-labs/flux-1.1-pro",
}
UPSCALER_MODEL_ID = "google/upscaler"  # Replicate Upscaler

# Standard-Account für den Erst-Login
DEFAULT_USERNAME = "flux"
DEFAULT_PASSWORD = "flux"

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
    """Konfigurationscontainer für Auth & Co."""
    credentials: dict
    cookie_name: str
    cookie_key: str
    cookie_expiry_days: int
    preauthorized: Optional[dict] = None
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
# Versionstolerantes Passwort-Hashing
# ============================================
def hash_password(password: str) -> str:
    """
    Versionstolerantes Hashing:
    - bevorzugt streamlit_authenticator.Hasher (neue und alte API)
    - Fallback auf bcrypt
    """
    try:
        # streamlit-authenticator >=0.4: classmethod `hash`
        if hasattr(stauth.Hasher, "hash"):
            return stauth.Hasher.hash(password)  # type: ignore[arg-type]
        # Ältere Versionen: Instanziierung mit Liste und .generate()
        return stauth.Hasher([password]).generate()[0]  # pragma: no cover - legacy path
    except Exception:
        try:
            import bcrypt  # pip install bcrypt
            salt = bcrypt.gensalt(rounds=12)
            return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
        except Exception as e:
            st.error(
                "Passworthashing fehlgeschlagen. Bitte installiere entweder "
                "`streamlit-authenticator` oder `bcrypt` (pip install bcrypt). "
                f"Technisches Detail: {e}"
            )
            st.stop()
            raise


# ============================================
# Erst-Login-Bootstrap: flux/flux + Pflicht
# ============================================
SECRETS_PATH = Path(".streamlit/secrets.toml")


def _load_secrets() -> Mapping[str, Any]:
    """Lädt st.secrets sicher, gibt {} bei fehlender Datei zurück."""
    try:
        return st.secrets  # type: ignore[return-value]
    except FileNotFoundError:
        return {}


def save_replicate_api_token(token: str) -> None:
    """Persistiert den übergebenen Token in `.streamlit/secrets.toml`."""
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
    """Erstellt fehlende `config.yaml`/`secrets.toml` und legt User `flux/flux` an."""
    # secrets.toml anlegen, falls nicht vorhanden
    if not SECRETS_PATH.exists():
        SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SECRETS_PATH.write_text("# hier können Secrets eingetragen werden\n", encoding="utf-8")

    # Wenn mit st.secrets gearbeitet wird, kein YAML-Bootstrap durchführen
    secrets = _load_secrets()
    if "credentials" in secrets and "cookie" in secrets:
        return

    cfg = _read_yaml(CONFIG_PATH)

    changed = False

    cookie = cfg.get("cookie")
    if not cookie:
        cookie = {
            "name": "auth",
            "key": pysecrets.token_hex(16),
            "expiry_days": 30,
        }
        cfg["cookie"] = cookie
        changed = True

    credentials = cfg.get("credentials") or {}
    usernames = credentials.get("usernames") or {}

    default_added = False
    if DEFAULT_USERNAME not in usernames:
        hashed = hash_password(DEFAULT_PASSWORD)
        usernames[DEFAULT_USERNAME] = {
            "name": "Flux User",
            "email": f"{DEFAULT_USERNAME}@example.com",
            "password": hashed,
            "must_change_credentials": True,
        }
        credentials["usernames"] = usernames
        cfg["credentials"] = credentials
        changed = True
        default_added = True

    if changed:
        _write_yaml(CONFIG_PATH, cfg)
        if default_added:
            logger.info(
                "Bootstrap: flux/flux angelegt und must_change_credentials=True gesetzt."
            )


def must_change_credentials_from_yaml(username: str) -> bool:
    """Prüft, ob für den Benutzer ein Credential-Change erzwungen wird."""
    if not CONFIG_PATH.exists():
        return False
    cfg = _read_yaml(CONFIG_PATH)
    try:
        return bool(
            cfg["credentials"]["usernames"][username].get("must_change_credentials", False)
        )
    except Exception:
        return False


def set_user_credentials_in_yaml(
    old_username: str, new_username: str, new_password: str
) -> None:
    """Aktualisiert Benutzername & Passwort in YAML und entfernt Pflicht-Flag."""
    cfg = _read_yaml(CONFIG_PATH)
    if "credentials" not in cfg:
        st.error("Ungültige YAML-Struktur: 'credentials' fehlt.")
        st.stop()
    usernames = cfg["credentials"].setdefault("usernames", {})
    user = usernames.pop(old_username, None)
    if not user:
        st.error("Benutzer nicht gefunden.")
        st.stop()

    user["password"] = hash_password(new_password)
    user["must_change_credentials"] = False
    user["name"] = new_username
    usernames[new_username] = user
    _write_yaml(CONFIG_PATH, cfg)


# ===========
# Utilities
# ===========
def load_config() -> AppConfig:
    """
    Lädt die Konfiguration aus st.secrets (falls vorhanden) oder YAML.
    Erwartete Felder:
      - credentials
      - cookie: { name, key, expiry_days }
      - preauthorized (optional)
      - replicate_api_token (optional; empfohlen via Umgebungsvariable ODER st.secrets)
    """
    # st.secrets bevorzugen
    secrets = _load_secrets()
    if "credentials" in secrets and "cookie" in secrets:
        cookie = secrets["cookie"]
        token = secrets.get("replicate_api_token") or os.getenv("REPLICATE_API_TOKEN")
        return AppConfig(
            credentials=dict(secrets["credentials"]),
            cookie_name=str(cookie.get("name", "app_auth")),
            cookie_key=str(cookie.get("key", "supersecret")),
            cookie_expiry_days=int(cookie.get("expiry_days", 30)),
            preauthorized=secrets.get("preauthorized"),
            replicate_api_token=token,
            source="secrets",
        )

    # Fallback: YAML-Datei
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
            preauthorized=cfg.get("preauthorized"),
            replicate_api_token=token,
            source="yaml",
        )
    except KeyError as e:
        st.error(f"Fehlendes Konfigurationsfeld: {e}")
        st.stop()
        raise


@st.cache_resource(show_spinner=False)
def get_replicate_client(api_token: Optional[str]) -> replicate.Client:
    """
    Initialisiert einen Replicate-Client. Token-Priorität:
    - Explizit übergebenes Token (st.secrets/config)
    - Umgebungsvariable REPLICATE_API_TOKEN
    """
    token = api_token or os.getenv("REPLICATE_API_TOKEN")
    if not token:
        st.error(
            "Fehlendes Replicate-API-Token. "
            "Bitte 'REPLICATE_API_TOKEN' in der Umgebung oder 'replicate_api_token' in st.secrets/config setzen."
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


def guess_mime(filename: str) -> str:
    """Bestimme den MIME-Type eines Dateinamens."""
    mt, _ = mimetypes.guess_type(filename)
    return mt or "application/octet-stream"


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


# =====================
# Authentifizierung UI
# =====================
def do_authentication(cfg: AppConfig):
    """
    Kompatibles Login für verschiedene streamlit_authenticator-Versionen.
    Gibt (name, username, auth_status, authenticator) zurück.
    """
    authenticator = stauth.Authenticate(
        credentials=cfg.credentials,
        cookie_name=cfg.cookie_name,
        key=cfg.cookie_key,
        cookie_expiry_days=cfg.cookie_expiry_days,
        preauthorized=cfg.preauthorized,
    )

    try:
        # streamlit-authenticator >=0.4 nutzt einen positionsbasierten "location"-Parameter
        name, auth_status, username = authenticator.login("main")
    except TypeError:
        # Ältere Versionen erwarten (form_name, location)
        name, auth_status, username = authenticator.login("Login", "main")
    except Exception as e:
        st.error(f"Login-Fehler: {e}")
        return None, None, None, authenticator

    if auth_status is False:
        st.error("Benutzername oder Passwort ist falsch.")
        st.session_state["authentication_status"] = None
        st.session_state.pop("username", None)
        st.session_state.pop("name", None)
    elif auth_status is None:
        st.info("Bitte geben Sie Ihren Benutzernamen und Ihr Passwort ein.")

    if auth_status:
        try:
            authenticator.logout("sidebar")
        except TypeError:
            authenticator.logout("Logout", "sidebar")
        st.sidebar.success(f"Willkommen, {name or username}!")

    return name, username, auth_status, authenticator


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


def run_inference(
    client: replicate.Client,
    model_id: str,
    inputs: Dict[str, Union[str, int, float, bool, BytesIO]],
) -> List[str] | object:
    try:
        return client.run(model_id, input=inputs)
    except Exception as e:
        logger.exception("Replicate-Run fehlgeschlagen")
        st.error(f"Fehler beim Ausführen des Modells '{model_id}': {e}")
        return []


def save_and_show_images(urls: List[str], desired_ext: str) -> None:
    ensure_save_dir(SAVE_DIR)
    for idx, url in enumerate(urls, start=1):
        ext = f".{desired_ext.lower()}" if desired_ext else os.path.splitext(url.split("?")[0])[-1]
        if ext.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            ext = ".jpg"

        filename = safe_unique_filename(ext, prefix="generiertes_bild")
        dest = SAVE_DIR / filename

        try:
            download_image(url, dest)
            st.success(f"Bild {idx} gespeichert: '{dest.as_posix()}'")
            st.image(dest.as_posix(), caption=f"Generiertes Bild {idx}", use_column_width=True)
        except requests.HTTPError as http_err:
            st.error(f"HTTP-Fehler beim Herunterladen von Bild {idx}: {http_err}")
        except Exception as e:
            logger.exception("Download fehlgeschlagen")
            st.error(f"Fehler beim Speichern von Bild {idx}: {e}")


def save_bytes_as_image_show(data: bytes, prefix: str = "upscaled", ext: str = ".png") -> None:
    ensure_save_dir(SAVE_DIR)
    try:
        img = Image.open(BytesIO(data))
        buf = image_to_png_buffer(img)
        filename = safe_unique_filename(".png", prefix=prefix)
        dest = SAVE_DIR / filename
        with dest.open("wb") as f:
            f.write(buf.getbuffer())
        st.success(f"Bild gespeichert: '{dest.as_posix()}'")
        st.image(dest.as_posix(), caption="Ergebnis", use_column_width=True)
    except Exception:
        filename = safe_unique_filename(ext, prefix=prefix)
        dest = SAVE_DIR / filename
        with dest.open("wb") as f:
            f.write(data)
        st.success(f"Datei gespeichert: '{dest.as_posix()}'")
        st.image(dest.as_posix(), caption="Ergebnis", use_column_width=True)


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
                    save_and_show_images(urls, desired_ext)
            elif isinstance(output, str):
                save_and_show_images([output], str(inputs.get("output_format", "jpg")))
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
                    save_bytes_as_image_show(data, prefix="upscaled", ext=".png")
                    return
                if isinstance(output, str):
                    save_and_show_images([output], "png")
                    return
                if isinstance(output, list) and output:
                    first = output[0]
                    if isinstance(first, str):
                        save_and_show_images([first], "png")
                        return
                    read_fn0 = getattr(first, "read", None)
                    if callable(read_fn0):
                        data0: bytes = read_fn0()
                        save_bytes_as_image_show(data0, prefix="upscaled", ext=".png")
                        return
                url_fn = getattr(output, "url", None)
                if callable(url_fn):
                    url_val = url_fn()
                    if isinstance(url_val, str):
                        save_and_show_images([url_val], "png")
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
def enforce_initial_credentials_change_ui(
    cfg: AppConfig, username: Optional[str]
) -> None:
    """Erzwingt beim ersten Login einen Wechsel von Benutzername und Passwort."""
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

    st.markdown("## 🔐 Zugangsdaten aktualisieren")
    st.info(
        "Erster Login mit Standard-Zugangsdaten erkannt. Bitte wähle einen neuen Benutzername und ein neues Passwort."
    )

    new_user = st.text_input("Neuer Benutzername")
    new1 = st.text_input("Neues Passwort", type="password")
    new2 = st.text_input("Neues Passwort bestätigen", type="password")
    min_len = 8
    if st.button("Speichern"):
        if not new_user or not new1 or not new2:
            st.error("Bitte alle Felder ausfüllen.")
        elif new1 != new2:
            st.error("Die Passwörter stimmen nicht überein.")
        elif len(new1) < min_len:
            st.error(f"Das Passwort muss mindestens {min_len} Zeichen haben.")
        else:
            if cfg.source == "yaml":
                try:
                    set_user_credentials_in_yaml(username, new_user, new1)
                    st.success(
                        "Zugangsdaten gespeichert. Bitte neu anmelden, um fortzufahren."
                    )
                    st.balloons()
                    st.stop()
                except Exception as e:
                    st.error(f"Fehler beim Schreiben in die YAML: {e}")
            else:
                hashed = hash_password(new1)
                st.warning(
                    "Die App läuft aktuell mit `st.secrets`. Bitte aktualisiere deine Secrets manuell und starte neu."
                )
                st.code(
                    f"""# In deinen Secrets/YAML einfügen:
credentials:
  usernames:
    {new_user}:
      name: "{new_user}"
      email: "{new_user}@example.com"
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
    """Entry-point der Streamlit-App."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🎨", layout="wide")

    # 1) Bootstrap-Dateien & Default-User anlegen
    ensure_bootstrap_files()

    # 2) Konfiguration laden
    cfg = load_config()

    # 3) Auth
    name, username, auth_status, _auth = do_authentication(cfg)
    if not auth_status:
        st.write("Bitte einloggen, um auf den Inhalt zuzugreifen.")
        st.stop()

    # 4) Erzwinge Wechsel der Standard-Zugangsdaten
    enforce_initial_credentials_change_ui(cfg, username or getattr(_auth, "username", None))

    # 5) Replicate-Token sicherstellen
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

    # 6) Menü
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
