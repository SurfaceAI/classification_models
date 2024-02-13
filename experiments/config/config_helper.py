def format_sweep_config(config):
    params = {
        key: {"value": value}
        for key, value in config.items()
        if key not in ["transform", "augment", "search_params"]
    }

    return {
        **params,
        "transform": {
            "parameters": {
                key: {"value": value} for key, value in config.get("transform").items()
            }
        },
        "augment": {
            "parameters": {
                key: {"value": value} for key, value in config.get("augment").items()
            }
        },
        **config.get("search_params"),
    }
