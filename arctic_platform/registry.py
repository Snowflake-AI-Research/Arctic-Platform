# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Automatic class registration for Arctic Platform extension points."""

from __future__ import annotations

import inspect
from abc import ABC
from abc import ABCMeta
from typing import Callable
from typing import TypeVar


class RegistryError(LookupError):
    """Raised when a requested registered class is unavailable."""


class RegistryValidationError(TypeError):
    """Raised when a registered class does not satisfy its base contract."""


_T = TypeVar("_T", bound=type)


class RegistryMeta(ABCMeta):
    """Register concrete subclasses under the nearest registry base class."""

    _registry: dict[str, dict[str, type]] = {}

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict) -> type:
        cls = super().__new__(mcs, name, bases, namespace)
        _validate_class_method(cls, "_validate_subclass", ["cls"])

        # A direct ABC child establishes a registry family but is not an entry.
        if any(base is ABC for base in bases):
            return cls
        if namespace.get("_skip_registry_registration", False):
            return cls

        roots = [
            ancestor
            for ancestor in cls.__mro__[1:]
            if isinstance(ancestor, RegistryMeta) and any(base is ABC for base in ancestor.__bases__)
        ]
        if not roots:
            return cls
        root = roots[0]

        # Names are deliberately not inherited: every registered subclass must
        # make its public identity explicit.
        if "name" not in namespace:
            raise RegistryValidationError(f"{cls.__name__} must define a 'name' attribute.")
        registry_name = namespace["name"]
        if not isinstance(registry_name, str) or not registry_name:
            raise RegistryValidationError(f"{cls.__name__}.name must be a non-empty string.")

        cls._validate_subclass()
        family = mcs._registry.setdefault(root.__name__, {})
        if registry_name in family:
            raise RegistryValidationError(f"{registry_name} is already registered as a {root.__name__}.")
        family[registry_name] = cls
        return cls


def get_registered_class(class_type: str, name: str) -> type:
    """Return a registered class and name the available entries on failure."""
    available = RegistryMeta._registry.get(class_type)
    if available is None:
        raise RegistryError(
            f"No classes of type {class_type} have been registered. "
            f"Available class families: {sorted(RegistryMeta._registry)}"
        )
    if name not in available:
        raise RegistryError(
            f"{name!r} is not a registered {class_type}. Available registered classes: {sorted(available)}"
        )
    return available[name]


def _validate_class_method(cls: type, method_name: str, expected_args: list[str] | None = None) -> None:
    expected_args = expected_args or []
    if not hasattr(cls, method_name):
        raise RegistryValidationError(f"{cls.__name__} must define a '{method_name}' method.")
    method: Callable = getattr(cls, method_name)
    if not callable(method):
        raise RegistryValidationError(f"{cls.__name__}.{method_name} must be callable.")
    if inspect.ismethod(method):
        method = method.__func__
    actual_args = set(inspect.signature(method).parameters)
    if actual_args != set(expected_args):
        raise RegistryValidationError(
            f"{cls.__name__}.{method_name} must accept exactly {set(expected_args)}, got {actual_args}."
        )


def _validate_class_attribute_set(cls: type, attribute: str) -> None:
    if not hasattr(cls, attribute):
        raise RegistryValidationError(f"{cls.__name__} must define a '{attribute}' attribute.")
