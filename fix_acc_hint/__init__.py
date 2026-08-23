from .alfworld import FixAccHintAlfworldPrefixRuntime
from .webshop import FixAccHintWebshopPrefixRuntime

__all__ = [
    "FixAccHintAlfworldPrefixRuntime",
    "FixAccHintWebshopPrefixRuntime",
    "build_fix_acc_hint_runtime",
]


def build_fix_acc_hint_runtime(tokenizer, config):
    if not bool(config.get("fix_acc_hint", {}).get("enable", False)):
        return None

    env_name = str(config.env.env_name).lower()
    if "alfworld" in env_name:
        return FixAccHintAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
    if "webshop" in env_name:
        return FixAccHintWebshopPrefixRuntime(tokenizer=tokenizer, config=config)
    return None
