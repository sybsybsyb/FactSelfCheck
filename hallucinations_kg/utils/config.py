from omegaconf import DictConfig, OmegaConf


def resolve_config(config: DictConfig, resolve: bool = True) -> dict:
    omegaconf_register_resolvers()
    config_primitive = OmegaConf.to_container(config, resolve=resolve)
    assert isinstance(config_primitive, dict)
    return config_primitive


def omegaconf_register_resolvers() -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    if not OmegaConf.has_resolver("if"):
        OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)
