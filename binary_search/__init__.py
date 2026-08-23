from .runtime import BinarySearchAlfworldPrefixRuntime
from .search import run_binary_search_multi_turn_loop

__all__ = [
    "BinarySearchAlfworldPrefixRuntime",
    "BinarySearchWebshopPrefixRuntime",
    "build_binary_search_runtime",
    "run_binary_search_multi_turn_loop",
]


def __getattr__(name):
    if name == "BinarySearchWebshopPrefixRuntime":
        from .webshop import BinarySearchWebshopPrefixRuntime

        return BinarySearchWebshopPrefixRuntime
    raise AttributeError(name)


def build_binary_search_runtime(tokenizer, config):
    if not bool(config.get("binary_search", {}).get("enable", False)):
        return None

    env_name = str(config.env.env_name).lower()
    if "alfworld" in env_name:
        return BinarySearchAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
    if "webshop" in env_name:
        from .webshop import BinarySearchWebshopPrefixRuntime

        return BinarySearchWebshopPrefixRuntime(tokenizer=tokenizer, config=config)
    return None
