import fnmatch


registry = {}


def make(id, call=True, *args, **kwargs):
    if id not in registry:
        matches = match(id)
        if len(matches) > 0:
            return {k: make(k, *args, **kwargs) for k in matches}

        # no matches found, try suggesting closest match
        import difflib
        suggestion = difflib.get_close_matches(id, registry.keys(), n=1)

        if len(suggestion) == 0:
            raise ValueError(f"`{id}` not found in registry.")
        else:
            raise ValueError(f"`{id}` not found in registry. Did you mean: `{suggestion[0]}`?")

    if call:
        return registry[id](*args, **kwargs)
    else:
        return registry[id]


def register(id, obj=None):
    if obj is None:
        # decorator
        def decorator(obj):
            registry[id] = obj
            return obj
        return decorator
    else:
        registry[id] = obj


def is_registered(id: str) -> bool:
    assert isinstance(id, str), f"Expected type str, got {type(id)}"
    return id in registry


def match(id: str) -> list[str]:
    assert isinstance(id, str), f"Expected type str, got {type(id)}"
    res = [k for k in registry.keys() if fnmatch.fnmatch(k, id)]
    return res

