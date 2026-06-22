from functools import lru_cache


@lru_cache(maxsize=None)
def _get_tokenizer(tokenizer_name: str):
    """Load (and cache) the HF tokenizer for a model so that we can apply its
    own chat template. We try the fast/standard load first and fall back to
    ``trust_remote_code=True`` for models that ship a custom tokenizer class
    (e.g. ChatGLM, InternLM)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    except Exception:
        return AutoTokenizer.from_pretrained(
            tokenizer_name, padding_side="left", trust_remote_code=True
        )


def apply_hf_chat_template(
    tokenizer_name: str, chat_messages: list[dict[str, str]]
) -> str:
    """Render ``chat_messages`` into a single prompt string using the model's
    own chat template shipped with its tokenizer.

    This lets us support arbitrary instruction-tuned models (Llama-3, Qwen/Yi
    ChatML, Mistral ``[INST]``, Granite, GLM-4, InternLM, Arcee, ...) without
    hand-writing a template for every family. ``tokenizer_name`` is the HF repo
    id (or a local path) of the model being served.
    """
    tokenizer = _get_tokenizer(tokenizer_name)
    return tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        truncation=False,
        padding=False,
    )
