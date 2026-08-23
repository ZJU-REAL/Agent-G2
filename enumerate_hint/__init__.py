from .runtime import EnumerateHintAlfworldPrefixRuntime
from .search import run_enumerate_hint_multi_turn_loop
from .webshop import EnumerateHintWebshopPrefixRuntime

__all__ = [
    "EnumerateHintAlfworldPrefixRuntime",
    "EnumerateHintWebshopPrefixRuntime",
    "build_enumerate_hint_runtime",
    "run_enumerate_hint_multi_turn_loop",
]


def build_enumerate_hint_runtime(tokenizer, config):
    if not bool(config.get("enumerate_hint", {}).get("enable", False)):
        return None

    env_name = str(config.env.env_name).lower()
    if "alfworld" in env_name:
        return EnumerateHintAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
    if "webshop" in env_name:
        return EnumerateHintWebshopPrefixRuntime(tokenizer=tokenizer, config=config)
    return None
