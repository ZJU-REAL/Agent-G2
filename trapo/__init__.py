from .runtime import TRAPOAlfworldPrefixRuntime, TRAPOWebshopPrefixRuntime
from .search import run_trapo_multi_turn_loop

__all__ = [
    "TRAPOAlfworldPrefixRuntime",
    "TRAPOWebshopPrefixRuntime",
    "build_trapo_runtime",
    "run_trapo_multi_turn_loop",
]


def build_trapo_runtime(tokenizer, config):
    if not bool(config.get("trapo", {}).get("enable", False)):
        return None

    env_name = str(config.env.env_name).lower()
    if "alfworld" in env_name:
        return TRAPOAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
    if "webshop" in env_name:
        return TRAPOWebshopPrefixRuntime(tokenizer=tokenizer, config=config)
    return None
