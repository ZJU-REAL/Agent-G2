from .alfworld import FixStepHintAlfworldPrefixRuntime
from .webshop import FixStepHintWebshopPrefixRuntime

__all__ = [
    "FixStepHintAlfworldPrefixRuntime",
    "FixStepHintWebshopPrefixRuntime",
    "build_fix_step_hint_runtime",
]


def build_fix_step_hint_runtime(tokenizer, config):
    if not bool(config.get("fix_step_hint", {}).get("enable", False)):
        return None

    env_name = str(config.env.env_name).lower()
    if "alfworld" in env_name:
        return FixStepHintAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
    if "webshop" in env_name:
        return FixStepHintWebshopPrefixRuntime(tokenizer=tokenizer, config=config)
    return None
