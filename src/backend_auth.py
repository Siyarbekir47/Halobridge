"""Resolve backend credentials without mixing them with the router's login token."""
from pathlib import Path


class BackendAuthError(ValueError):
    pass


def first_key(value):
    keys = [key.strip() for key in value.replace("\n", ",").split(",") if key.strip()]
    if len(value) > 65536 or not keys or not all(
            len(key) <= 512 and all(0x21 <= ord(c) <= 0x7e for c in key) for key in keys):
        raise BackendAuthError("Backend API key is empty or invalid")
    return keys[0]


def file_key(path):
    try:
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            value = handle.read(65537)
    except (OSError, UnicodeError) as exc:
        raise BackendAuthError("Backend API key file cannot be read") from exc
    return first_key(value)


class BackendAuth:
    def __init__(self, config, profiles):
        self.config, self.profiles = config, profiles

    def headers(self, model):
        profile = next((p for p in self.profiles() if p.model_id == model), None)
        # Prefer the target profile: credentials may differ after a model switch.
        if profile and profile.env.get("HALOGEN_API_KEY"):
            key = first_key(profile.env["HALOGEN_API_KEY"])
        elif profile and profile.env.get("HALOGEN_API_KEY_FILE"):
            from shared_assets import host_path
            path = host_path(profile, profile.env["HALOGEN_API_KEY_FILE"])
            if path is None:
                raise BackendAuthError("Backend API key file has no host mount")
            key = file_key(path)
        elif self.config.backend_api_key_file:
            key = file_key(self.config.backend_api_key_file)
        elif self.config.backend_api_key:
            key = first_key(self.config.backend_api_key)
        else:
            return {}
        return {"Authorization": "Bearer " + key}
