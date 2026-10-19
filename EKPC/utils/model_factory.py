def get_model(model_name, args):
    if model_name.lower() not in {"adapter", "ekpc"}:
        raise ValueError(f"Unsupported model: {model_name}")
    from models.EKPC import Learner

    return Learner(args)
