"""
Modal inference service for fine-tuned adapters.

Serves the fine-tuned Gemma models (base model + PEFT adapter) on a Modal
GPU using the exact same stack that trained them (transformers + PEFT +
bitsandbytes NF4). This avoids two problems with running the adapters
locally via MLX:

1. mlx-lm cannot load HuggingFace PEFT-format adapters at all.
2. Even after conversion, the locally quantized base model differs from
   the bnb-NF4 base the adapter was trained against, which can shift
   behavior and muddy the fine-tuned-vs-baseline comparison.

Here, the model evaluated is bit-for-bit the model that was trained.

Architecture:
- A parameterized Modal class loads the base model in 4-bit and applies
  the requested adapter from the ``mischar-adapters`` volume on container
  start. Setting ``adapter_name=""`` serves the plain base model (useful
  for a prompted baseline on identical infrastructure).
- The container stays warm for ``SCALEDOWN_WINDOW_SECONDS`` after the
  last call, so sequential eval requests don't reload the model.
- HuggingFace downloads are cached in a volume so cold starts after the
  first don't re-download the ~50GB base model.
- The local side (``mischar.models.modal_inference.ModalInferenceClient``)
  looks this class up by name and calls ``generate`` remotely.

Usage:
    # Deploy (required before the local client can connect):
    modal deploy src/mischar/scripts/inference/serve_adapter.py

    # Smoke test without deploying:
    modal run src/mischar/scripts/inference/serve_adapter.py
"""

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure (app, image, volumes, GPU config)
#
# Defined inline rather than imported from a shared module — Modal copies
# each script into the container detached from the local directory
# structure, so cross-file imports break inside the container.
# ---------------------------------------------------------------------------

app = modal.App("mischar-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.46",
        "peft>=0.13",
        "bitsandbytes>=0.44",
        "accelerate>=1.0",
        # sentencepiece / protobuf needed for Gemma tokenizer
        "sentencepiece",
        "protobuf",
    )
)

# Trained adapters (written by train_primary.py / train_secondary.py).
adapter_volume = modal.Volume.from_name(
    "mischar-adapters",
    create_if_missing=True,
)

# HuggingFace download cache. Persists base model weights across cold
# starts so only the first-ever container pays the download time.
hf_cache_volume = modal.Volume.from_name(
    "mischar-hf-cache",
    create_if_missing=True,
)

ADAPTER_PATH = "/adapters"
HF_CACHE_PATH = "/root/.cache/huggingface"

GPU_TYPE = "H100"

# Keep the container warm between eval calls. Long enough to ride out
# local-side retrieval work between classify calls; short enough not to
# burn idle GPU dollars after the eval finishes.
SCALEDOWN_WINDOW_SECONDS = 300

# Generous per-call timeout: the first call on a cold container includes
# model load (and on the very first run, the base-model download).
CALL_TIMEOUT_SECONDS = 3600

DEFAULT_BASE_MODEL_ID = "google/gemma-3-27b-it"
DEFAULT_ADAPTER_NAME = "gemma3-27b-mischar-v2"


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        ADAPTER_PATH: adapter_volume,
        HF_CACHE_PATH: hf_cache_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    timeout=CALL_TIMEOUT_SECONDS,
)
class MischarClassifier:
    """
    Serves a (base model + adapter) pair for classification inference.

    Parameterized so one deployment serves any trained adapter: each
    distinct (base_model_id, adapter_name) pair gets its own container
    pool. ``adapter_name=""`` serves the unadapted base model.
    """

    base_model_id: str = modal.parameter(default=DEFAULT_BASE_MODEL_ID)
    adapter_name: str = modal.parameter(default=DEFAULT_ADAPTER_NAME)

    @modal.enter()
    def load(self) -> None:
        """
        Load the base model in 4-bit and apply the adapter.

        Runs once per container start. Uses the same quantization config
        as training (bnb NF4, bf16 compute, double quant) so the adapter
        sees the identical base it was trained against.
        """
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        hf_token = os.environ.get("HF_TOKEN")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"Loading base model {self.base_model_id} in 4-bit...")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        tokenizer_source = self.base_model_id

        if self.adapter_name:
            from peft import PeftModel

            adapter_dir = os.path.join(ADAPTER_PATH, self.adapter_name, "best")
            if not os.path.isdir(adapter_dir):
                raise FileNotFoundError(
                    f"Adapter directory not found: {adapter_dir}. "
                    "Check the adapter volume contents with: "
                    "modal volume ls mischar-adapters"
                )

            print(f"Applying adapter from {adapter_dir}...")
            model = PeftModel.from_pretrained(model, adapter_dir)
            # The training scripts save the tokenizer alongside the adapter.
            tokenizer_source = adapter_dir

        model.eval()

        self._model = model
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            token=hf_token,
            trust_remote_code=True,
        )

        print(
            f"Model ready: base={self.base_model_id}, "
            f"adapter={self.adapter_name or '(none)'}"
        )

    @modal.method()
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a completion for a single prompt.

        Applies Gemma's chat template (user turn → generation prompt),
        matching the format the adapter was trained on via SFTTrainer.

        Args:
            prompt: The full prompt text (e.g. a classification prompt).
            temperature: Sampling temperature. 0.0 = greedy decoding.
            max_tokens: Maximum new tokens to generate.

        Returns:
            The generated text, with prompt and special tokens stripped.
        """
        import torch

        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        generate_kwargs: dict = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **generate_kwargs)

        # Slice off the prompt tokens so only the completion is decoded.
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = output_ids[0][prompt_length:]

        return self._tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Smoke test entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    """
    Smoke test: ``modal run src/mischar/scripts/inference/serve_adapter.py``.

    Spins up a container with the default base model and adapter and runs
    a single trivial generation to verify the load path works end to end.
    """
    classifier = MischarClassifier()
    result = classifier.generate.remote(
        prompt='Respond with a JSON object: {"label": "accurate"}',
        max_tokens=32,
    )
    print(f"Response: {result}")
