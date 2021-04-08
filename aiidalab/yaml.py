# -*- coding: utf-8 -*-
from pathlib import Path
from urllib.parse import urlsplit

import cachecontrol
import requests
import jsonref
from ruamel.yaml import YAML


REQUESTS = cachecontrol.CacheControl(requests.Session())


class JsonYamlLoader(jsonref.JsonLoader):

    safe_yaml = YAML(typ="safe")

    def __call__(self, uri, **kwargs):
        uri_split = urlsplit(uri)
        if Path(uri_split.path).suffix in (".yml", ".yaml"):
            if uri_split.scheme == "file":
                content = Path(uri_split.path).read_bytes()
            else:
                response = REQUESTS.get(uri)
                response.raise_for_status()
                content = response.content
            return self.safe_yaml.load(content)
        else:
            return super().__call__(uri, **kwargs)


json_yaml_loader = JsonYamlLoader()


def replace_refs(obj):
    """Dereference all references in obj.

    References may point to JSON or YAML files.
    """
    return jsonref.JsonRef.replace_refs(obj, loader=json_yaml_loader)


def loads(s):
    """Deserialize serialized YAML file 's' to a Python object and dereference all references."""
    return replace_refs(YAML(typ="safe").load(s))


def load(path):
    """Deserialize YAML file at path to a Python object and dereference all references."""
    return loads(Path(path).read_text())


__all__ = [
    "load",
    "loads",
    "replace_refs",
]
