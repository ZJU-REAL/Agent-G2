from .alfworld import GMSVAlfworldPrefixRuntime
from .webshop import GMSVWebshopPrefixRuntime

__all__ = [
    "GMSVAlfworldPrefixRuntime",
    "GMSVWebshopPrefixRuntime",
    "build_gmsv_runtime",
]


def build_gmsv_runtime(tokenizer, config):
    if not bool(config.get("gmsv", {}).get("enable", False)):
        return None

    env_name = str(config.env.env_name).lower()
    if "alfworld" in env_name:
        return GMSVAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
    if "webshop" in env_name:
        return GMSVWebshopPrefixRuntime(tokenizer=tokenizer, config=config)
    return None
