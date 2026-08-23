from .alfworld import LinerHintAlfworldPrefixRuntime
from .webshop import LinerHintWebshopPrefixRuntime

__all__ = [
    "LinerHintAlfworldPrefixRuntime",
    "LinerHintWebshopPrefixRuntime",
    "build_liner_hint_runtime",
]


def build_liner_hint_runtime(tokenizer, config):
    if not bool(config.get("liner_hint", {}).get("enable", False)):
        return None

    env_name = str(config.env.env_name).lower()
    if "alfworld" in env_name:
        return LinerHintAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
    if "webshop" in env_name:
        return LinerHintWebshopPrefixRuntime(tokenizer=tokenizer, config=config)
    return None
