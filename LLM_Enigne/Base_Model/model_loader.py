import os
import shutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ============================================================
# MODEL REGISTRY (Instruct models for reasoning)
# ============================================================

ALL_MODELS = {
    # alias              HF model id                                vram_4bit (GB)
    "qwen2.5-1.5b":  {"id": "Qwen/Qwen2.5-1.5B-Instruct",         "vram_4bit": 1.5},
    "qwen2.5-3b":    {"id": "Qwen/Qwen2.5-3B-Instruct",           "vram_4bit": 2.5},
    "qwen2.5-7b":    {"id": "Qwen/Qwen2.5-7B-Instruct",           "vram_4bit": 4.5},
    "phi3-mini":     {"id": "microsoft/Phi-3-mini-4k-instruct",   "vram_4bit": 3.0},
    "mistral-7b":    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "vram_4bit": 4.5},
}

GATED_MODELS = set()

# ============================================================
# LLMLoader
# ============================================================

class LLMLoader:

    def __init__(
        self,
        model_name: str = "qwen2.5-1.5b",   # 🔥 Default model
        model_path: str | None = None,
    ):
        if model_name not in ALL_MODELS:
            raise ValueError(
                f"Unknown model alias '{model_name}'. "
                f"Available: {sorted(ALL_MODELS.keys())}"
            )

        entry            = ALL_MODELS[model_name]
        self.model_alias = model_name
        self.hf_model_id = entry["id"]
        self.vram_4bit   = entry["vram_4bit"]

        default_path     = os.path.join(
            "LLM_engine", "base_model",
            model_name.replace("/", "_")
        )
        self.model_path  = model_path or default_path

        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self.model       = None
        self.tokenizer   = None

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _is_cached(self) -> bool:
        return (
            os.path.isdir(self.model_path)
            and "config.json" in os.listdir(self.model_path)
        )

    def _clean_broken_cache(self):
        if os.path.isdir(self.model_path) and not self._is_cached():
            print(f"⚠️ Broken cache detected at {self.model_path} — removing…")
            shutil.rmtree(self.model_path)

    # ----------------------------------------------------------
    # Load Model
    # ----------------------------------------------------------

    def load_model(self, quantization: str = "4bit"):

        self._clean_broken_cache()
        source = self.model_path if self._is_cached() else self.hf_model_id

        print(f"📂 Loading {self.model_alias}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None

        if quantization == "4bit" and self.device == "cuda":
            print("⚡ Using 4-bit quantization")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            source,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✅ {self.model_alias} loaded on {self.device}")
        return self.model, self.tokenizer

    # ----------------------------------------------------------
    # Simple Chat
    # ----------------------------------------------------------

    def chat(self, prompt: str, max_new_tokens: int = 300) -> str:

        if self.model is None:
            raise RuntimeError("Model not loaded.")

        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        if prompt in decoded:
            decoded = decoded.split(prompt)[-1].strip()

        return decoded

    # ----------------------------------------------------------
    # Info
    # ----------------------------------------------------------

    def get_model_info(self):

        if self.model is None:
            return {"status": "not loaded"}

        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "alias": self.model_alias,
            "hf_id": self.hf_model_id,
            "device": self.device,
            "parameters": f"{total_params/1e9:.2f}B",
            "dtype": str(self.model.dtype),
        }


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":

    loader = LLMLoader(model_name="qwen2.5-1.5b")
    model, tokenizer = loader.load_model(quantization="4bit")

    print(loader.get_model_info())

    print("\nTest Chat:")
    print(loader.chat("Assess a restricted area breach at night."))