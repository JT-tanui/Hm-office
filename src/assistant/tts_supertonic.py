import json
import os
import time
import re
import logging
import numpy as np
import onnxruntime as ort
from typing import Optional, List, Tuple
from unicodedata import normalize
from .config import ASSETS_DIR

logger = logging.getLogger(__name__)

# --- Adapted from Supertonic helper.py ---

class UnicodeProcessor:
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, "r", encoding="utf-8") as f:
            self.indexer = json.load(f)

    def _preprocess_text(self, text: str) -> str:
        """Full preprocessing from Supertonic - critical for TTS reliability"""
        text = normalize("NFKD", text)
        
        # Remove emojis (wide Unicode range) - CRITICAL: prevents synthesis failures
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f700-\U0001f77f"
            "\U0001f780-\U0001f7ff"
            "\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff"
            "\U0001fa00-\U0001fa6f"
            "\U0001fa70-\U0001faff"
            "\u2600-\u26ff"
            "\u2700-\u27bf"
            "\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)
        
        # Replace various dashes and symbols
        replacements = {
            "–": "-",
            "‑": "-",
            "—": "-",
            "¯": " ",
            "_": " ",
            """: '"',
            """: '"',
            "'": "'",
            "'": "'",
            "´": "'",
            "`": "'",
            "[": " ",
            "]": " ",
            "|": " ",
            "/": " ",
            "#": " ",
            "→": " ",
            "←": " ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        
        # Remove combining diacritics
        text = re.sub(
            r"[\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u030A\u030B\u030C\u0327\u0328\u0329\u032A\u032B\u032C\u032D\u032E\u032F]",
            "",
            text,
        )
        
        # Remove special symbols
        text = re.sub(r"[♥☆♡©\\]", "", text)
        
        # Replace known expressions
        expr_replacements = {
            "@": " at ",
            "e.g.,": "for example, ",
            "i.e.,": "that is, ",
        }
        for k, v in expr_replacements.items():
            text = text.replace(k, v)
        
        # Fix spacing around punctuation
        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)
        text = re.sub(r" ;", ";", text)
        text = re.sub(r" :", ":", text)
        text = re.sub(r" '", "'", text)
        
        # Remove duplicate quotes
        while '""' in text:
            text = text.replace('""', '"')
        while "''" in text:
            text = text.replace("''", "'")
        while "``" in text:
            text = text.replace("``", "`")
        
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        # If text doesn't end with punctuation, quotes, or closing brackets, add a period
        if not re.search(r"[.!?;:,'\"')\\]}…。」』】〉》›»]$", text):
            text += "."
        
        return text

    def _get_text_mask(self, text_ids_lengths: np.ndarray) -> np.ndarray:
        max_len = text_ids_lengths.max()
        ids = np.arange(0, max_len)
        mask = (ids < np.expand_dims(text_ids_lengths, axis=1)).astype(np.float32)
        return mask.reshape(-1, 1, max_len)

    def _text_to_unicode_values(self, text: str) -> np.ndarray:
        return np.array([ord(char) for char in text], dtype=np.uint16)

    def __call__(self, text_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        text_list = [self._preprocess_text(t) for t in text_list]
        text_ids_lengths = np.array([len(text) for text in text_list], dtype=np.int64)
        text_ids = np.zeros((len(text_list), text_ids_lengths.max()), dtype=np.int64)
        for i, text in enumerate(text_list):
            unicode_vals = self._text_to_unicode_values(text)
            
            for j, val in enumerate(unicode_vals):
                if val < len(self.indexer):
                    id_ = self.indexer[val]
                    if id_ != -1:
                        text_ids[i, j] = id_
                    else:
                        text_ids[i, j] = 0 # Fallback/Unknown
                else:
                    text_ids[i, j] = 0
        text_mask = self._get_text_mask(text_ids_lengths)
        return text_ids, text_mask

class Style:
    def __init__(self, style_ttl_onnx: np.ndarray, style_dp_onnx: np.ndarray):
        self.ttl = style_ttl_onnx
        self.dp = style_dp_onnx

class SupertonicTTS:
    def __init__(self, assets_dir: str = str(ASSETS_DIR)):
        self.assets_dir = assets_dir
        self.onnx_dir = os.path.join(assets_dir, "onnx") # Assuming assets/onnx structure
        
        # Check if onnx dir exists, if not, maybe it's directly in assets?
        if not os.path.exists(self.onnx_dir):
             self.onnx_dir = assets_dir

        self.cfgs = self._load_cfgs()
        self.text_processor = self._load_text_processor()
        self.models = self._load_models()
        
        self.sample_rate = self.cfgs["ae"]["sample_rate"]
        self.base_chunk_size = self.cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = self.cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = self.cfgs["ttl"]["latent_dim"]

    def _load_cfgs(self) -> dict:
        with open(os.path.join(self.onnx_dir, "tts.json"), "r") as f:
            return json.load(f)

    def _load_text_processor(self) -> UnicodeProcessor:
        return UnicodeProcessor(os.path.join(self.onnx_dir, "unicode_indexer.json"))

    def _load_models(self):
        opts = ort.SessionOptions()
        providers = ["CPUExecutionProvider"]
        
        return {
            "dp": ort.InferenceSession(os.path.join(self.onnx_dir, "duration_predictor.onnx"), opts, providers=providers),
            "text_enc": ort.InferenceSession(os.path.join(self.onnx_dir, "text_encoder.onnx"), opts, providers=providers),
            "vector_est": ort.InferenceSession(os.path.join(self.onnx_dir, "vector_estimator.onnx"), opts, providers=providers),
            "vocoder": ort.InferenceSession(os.path.join(self.onnx_dir, "vocoder.onnx"), opts, providers=providers)
        }

    def load_voice_style(self, style_path: str) -> Style:
        with open(style_path, "r") as f:
            style_data = json.load(f)
        
        ttl_dims = style_data["style_ttl"]["dims"]
        dp_dims = style_data["style_dp"]["dims"]
        
        ttl_style = np.array(style_data["style_ttl"]["data"], dtype=np.float32).reshape(1, ttl_dims[1], ttl_dims[2])
        dp_style = np.array(style_data["style_dp"]["data"], dtype=np.float32).reshape(1, dp_dims[1], dp_dims[2])
        
        return Style(ttl_style, dp_style)

    def synthesize(self, text: str, style_path: str = None, speed: float = 1.0, steps: int = 5) -> Tuple[np.ndarray, int]:
        if style_path is None:
            # Default to first style found in assets/voice_styles if exists
            style_dir = os.path.join(self.assets_dir, "voice_styles")
            if os.path.exists(style_dir):
                styles = [f for f in os.listdir(style_dir) if f.endswith(".json")]
                if styles:
                    style_path = os.path.join(style_dir, styles[0])
        
        if not style_path or not os.path.exists(style_path):
             raise FileNotFoundError("No voice style found.")

        style = self.load_voice_style(style_path)
        
        # Chunk text (simplified)
        wav_cat = None
        # For simplicity, treating whole text as one chunk if short, or basic split
        # In production, use the robust chunk_text from helper.py
        
        wav, dur = self._infer([text], style, steps, speed)
        return wav.flatten(), self.sample_rate

    def _infer(self, text_list: List[str], style: Style, total_step: int, speed: float) -> Tuple[np.ndarray, np.ndarray]:
        bsz = len(text_list)
        text_ids, text_mask = self.text_processor(text_list)
        
        dp_ort = self.models["dp"]
        text_enc_ort = self.models["text_enc"]
        vector_est_ort = self.models["vector_est"]
        vocoder_ort = self.models["vocoder"]

        dur_onnx, *_ = dp_ort.run(None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask})
        dur_onnx = dur_onnx / speed
        
        text_emb_onnx, *_ = text_enc_ort.run(None, {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask})
        
        xt, latent_mask = self._sample_noisy_latent(dur_onnx)
        
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)
        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            xt, *_ = vector_est_ort.run(None, {
                "noisy_latent": xt,
                "text_emb": text_emb_onnx,
                "style_ttl": style.ttl,
                "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": current_step,
                "total_step": total_step_np,
            })
            
        wav, *_ = vocoder_ort.run(None, {"latent": xt})
        return wav, dur_onnx

    def _sample_noisy_latent(self, duration: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = ((wav_len_max + chunk_size - 1) / chunk_size).astype(np.int32)
        latent_dim = self.ldim * self.chunk_compress_factor
        
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        
        # Latent mask logic from helper.py
        latent_size = self.base_chunk_size * self.chunk_compress_factor
        latent_lengths = (wav_lengths + latent_size - 1) // latent_size
        
        # length_to_mask logic
        max_len = latent_len # Use the calculated max latent len
        ids = np.arange(0, max_len)
        mask = (ids < np.expand_dims(latent_lengths, axis=1)).astype(np.float32)
        latent_mask = mask.reshape(-1, 1, max_len)
        
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask
