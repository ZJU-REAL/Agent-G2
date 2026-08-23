from .alfworld import CosineHintAlfworldPrefixRuntime
from .webshop import CosineHintWebshopPrefixRuntime

__all__ = [
    "CosineHintAlfworldPrefixRuntime",
    "CosineHintWebshopPrefixRuntime",
    "build_cosine_hint_runtime",
]


def build_cosine_hint_runtime(tokenizer, config):
    if not bool(config.get("cosine_hint", {}).get("enable", False)):
        return None

    env_name = str(config.env.env_name).lower()
    if "alfworld" in env_name:
        return CosineHintAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
    if "webshop" in env_name:
        return CosineHintWebshopPrefixRuntime(tokenizer=tokenizer, config=config)
    return None
