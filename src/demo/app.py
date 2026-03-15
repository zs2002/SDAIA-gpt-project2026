"""Streamlit demo app for GPT text generation.

Provides a web interface with:
- Model selector (pretrained vs fine-tuned checkpoint)
- Text prompt input
- Generation parameter sliders (temperature, top_k, top_p, max_new_tokens)
- Generated text display with timing
- RTL support for Arabic text
- Error handling for missing checkpoints, timeouts, and invalid params
"""

from __future__ import annotations

import glob
import os
import re
import time

import streamlit as st
import torch

from src.config import GPTConfig
from src.generation.generator import TextGenerator
from src.model.transformer import GPTModel
from src.tokenizer.bpe_tokenizer import BPETokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PRETRAINED_DIR = os.path.join(_PROJECT_ROOT, "checkpoints", "pretrained")
_FINETUNED_DIR = os.path.join(_PROJECT_ROOT, "checkpoints", "finetuned")
_GENERATION_TIMEOUT = 60  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scan_checkpoints() -> dict[str, str]:
    """Scan checkpoint directories and return a mapping of display name → path."""
    checkpoints: dict[str, str] = {}

    for ckpt_path in sorted(glob.glob(os.path.join(_PRETRAINED_DIR, "*.pt"))):
        name = f"pretrained / {os.path.basename(ckpt_path)}"
        checkpoints[name] = ckpt_path

    for ckpt_path in sorted(glob.glob(os.path.join(_FINETUNED_DIR, "*.pt"))):
        name = f"finetuned / {os.path.basename(ckpt_path)}"
        checkpoints[name] = ckpt_path

    return checkpoints


def _find_tokenizer_path() -> str | None:
    """Look for a saved tokenizer JSON in common locations."""
    candidates = [
        os.path.join(_PROJECT_ROOT, "tokenizer.json"),
        os.path.join(_PROJECT_ROOT, "data", "tokenizer.json"),
        os.path.join(_PROJECT_ROOT, "checkpoints", "tokenizer.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    # Also search recursively under checkpoints
    for path in glob.glob(os.path.join(_PROJECT_ROOT, "checkpoints", "**", "tokenizer.json"), recursive=True):
        return path
    return None


def _contains_arabic(text: str) -> bool:
    """Return True if *text* contains Arabic script characters."""
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", text))


def _render_text(text: str) -> None:
    """Render text with RTL support when Arabic characters are detected."""
    if _contains_arabic(text):
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = f'<div dir="rtl" style="text-align:right; font-size:1.1em; line-height:1.8;">{escaped}</div>'
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.text(text)


@st.cache_resource
def _load_model_and_tokenizer(ckpt_path: str) -> tuple[GPTModel, BPETokenizer]:
    """Load a checkpoint and tokenizer, returning (model, tokenizer).

    The checkpoint is expected to contain at minimum ``model_state_dict`` and
    ``config`` (a GPTConfig instance or dict).
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Reconstruct GPTConfig
    cfg = checkpoint.get("config")
    if isinstance(cfg, dict):
        config = GPTConfig(**cfg)
    elif isinstance(cfg, GPTConfig):
        config = cfg
    else:
        # Fallback to defaults
        config = GPTConfig()

    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load tokenizer
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tok_path = _find_tokenizer_path()
    if tok_path is not None:
        tokenizer.load(tok_path)
    else:
        # Train a minimal tokenizer so the app doesn't crash
        tokenizer.train("hello world مرحبا بالعالم")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit demo application."""
    st.set_page_config(page_title="GPT Demo", page_icon="🤖", layout="centered")
    st.title("🤖 GPT from Scratch — Text Generation Demo")
    st.markdown("Generate text using a decoder-only transformer trained from scratch.")

    # ------------------------------------------------------------------
    # Sidebar: model & generation parameters
    # ------------------------------------------------------------------
    st.sidebar.header("Model")

    checkpoints = _scan_checkpoints()

    if not checkpoints:
        st.error(
            "No checkpoint files found. Please train a model first and place "
            "`.pt` files in `checkpoints/pretrained/` or `checkpoints/finetuned/`."
        )
        generate_disabled = True
        selected_ckpt_path: str | None = None
    else:
        selected_name = st.sidebar.selectbox("Checkpoint", list(checkpoints.keys()))
        selected_ckpt_path = checkpoints[selected_name]
        generate_disabled = False

    st.sidebar.header("Generation Parameters")

    temperature = st.sidebar.slider(
        "Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.05,
        help="0 = greedy (deterministic). Higher values increase randomness.",
    )
    top_k = st.sidebar.slider(
        "Top-k", min_value=0, max_value=100, value=50, step=1,
        help="Keep only the top-k most probable tokens. 0 = disabled.",
    )
    top_p = st.sidebar.slider(
        "Top-p (nucleus)", min_value=0.0, max_value=1.0, value=0.9, step=0.05,
        help="Keep smallest set of tokens whose cumulative probability exceeds p.",
    )
    max_new_tokens = st.sidebar.slider(
        "Max new tokens", min_value=1, max_value=500, value=100, step=1,
    )

    # ------------------------------------------------------------------
    # Main area: prompt input & generation
    # ------------------------------------------------------------------
    prompt = st.text_area(
        "Enter your prompt",
        height=120,
        placeholder="Type something here… (English or Arabic)",
    )

    generate_clicked = st.button("Generate", disabled=generate_disabled, type="primary")

    if generate_clicked:
        if not prompt or not prompt.strip():
            st.warning("Please enter a non-empty prompt.")
            return

        if selected_ckpt_path is None:
            st.error("No checkpoint selected.")
            return

        try:
            with st.spinner("Loading model…"):
                model, tokenizer = _load_model_and_tokenizer(selected_ckpt_path)

            generator = TextGenerator(model, tokenizer)

            # Resolve effective parameters
            effective_top_k = top_k if top_k > 0 else None
            effective_top_p = top_p if top_p < 1.0 else None

            with st.spinner("Generating…"):
                start = time.time()
                generated_text, token_ids = generator.generate(
                    prompt=prompt.strip(),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=effective_top_k,
                    top_p=effective_top_p,
                )
                elapsed = time.time() - start

            if elapsed > _GENERATION_TIMEOUT:
                st.warning(
                    f"Generation took {elapsed:.1f}s which exceeds the "
                    f"{_GENERATION_TIMEOUT}s timeout threshold."
                )

            # Display results
            st.subheader("Generated Text")
            _render_text(generated_text)

            st.caption(f"⏱ Generated {len(token_ids)} tokens in {elapsed:.2f}s")

        except Exception as exc:  # noqa: BLE001
            st.error(f"Generation failed: {exc}")


if __name__ == "__main__":
    main()
