# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ast
import difflib
import functools
import inspect
import os
import sys
import textwrap
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NamedTuple

import torch

from aklr.core.contract.closure import compile_closure, fill_expr, fill_statements
from aklr.core.contract.fastkey import (
    AllocOutputsFnGenerator,
    CallArgs,
    FastKeyCodec,
    FastKeyFnGenerator,
)
from aklr.core.contract.locate import import_entry, op_module
from aklr.core.contract.op import OpContract, api_name
from aklr.core.contract.registry import ContractRegistry, UnknownContractError
from aklr.core.contract.spec import AttrSpec, attr_value, contract_name
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason

__all__ = [
    "OUT_MODES",
    "BackendCall",
    "EnvAttr",
    "NoBaselineError",
    "Proxy",
    "ProxyRoute",
    "RouteDeclinedError",
    "RouteError",
    "aklr_proxy",
    "aklr_route",
    "apply_env",
    "attr_from_env",
    "backend_bind",
    "backend_call",
    "baseline_of",
    "check_signature",
    "contract_routes",
    "custom_baseline",
    "must_default",
    "none_if_falsy",
    "route_for",
    "routes",
    "unused",
]

# How an entry point hands its result back.
OUT_MODES = ("value", "out", "dps")


class EnvAttr(NamedTuple):
    """One parsed ``attr_from_env(...)`` argument."""

    # Api param whose value selects the rule.
    selector: str
    # ``attr_value(call[selector])`` at which the environment decides.
    when: str | int | bool
    # ``(variable, attr value)`` in the backend's priority order.
    env: tuple[tuple[str, str | int | bool], ...]


class RouteError(ValueError):
    pass


class RouteDeclinedError(ValueError):
    """This call is one AKLR does not serve, or the route cannot spell back."""

    def __init__(self, api: str, name: str, value: Any, expected: Any) -> None:
        self.api = api
        self.name = name
        self.value = value
        self.expected = expected
        super().__init__(api, name, value, expected)  # .args rebuilds it

    def __str__(self) -> str:
        if isinstance(self.expected, EnvAttr):
            names = ", ".join(var for var, _ in self.expected.env)
            return (
                f"{self.api}: '{self.name}' comes from {names}, none of which is '1'; aiter serves this call"
            )
        value = self.value
        if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device"):
            described = f"{type(value).__name__}({tuple(value.shape)}, {value.dtype}, {value.device})"
        else:
            described = repr(value)
        return f"{self.api}: '{self.name}'={described}, expected {self.expected!r}; aiter serves this call"


@dataclass(frozen=True)
class ProxyRoute:
    """One entry point's declaration that it proxies to an op."""

    api: str
    op: str
    # The op's api function; out of equality like ``declared``.
    api_fn: Callable = field(compare=False)
    # Backend parameter name -> api param name; the rest keep their own.
    params: dict[str, str]
    # Api param name -> literal the entry point pins rather than passes.
    consts: dict[str, Any]
    out_mode: str
    must_default: tuple[str, ...]
    # Forwarded backend parameters whose falsy values reach the op as None.
    none_if_falsy: tuple[str, ...]
    # Out of equality: a re-import builds a new function for the same route.
    declared: Callable = field(compare=False)
    # Api attr param -> the environment rule that sets it.
    env_attrs: dict[str, EnvAttr]

    def __post_init__(self) -> None:
        if not self.api:
            raise RouteError("a route needs the entry point's dotted name")
        if self.out_mode not in OUT_MODES:
            raise RouteError(
                f"route for '{self.api}': out_mode must be one of {OUT_MODES}, got {self.out_mode!r}"
            )
        # Own the dicts: a caller's literal must not be edited underneath us.
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "consts", dict(self.consts))
        object.__setattr__(self, "env_attrs", dict(self.env_attrs))
        object.__setattr__(self, "must_default", tuple(self.must_default))
        object.__setattr__(self, "none_if_falsy", tuple(self.none_if_falsy))

    def backend_bind(self, contract: OpContract, call: CallArgs) -> BackendCall:
        """``call`` for this route's raw backend body."""
        api = self.api
        try:
            entry = import_entry(api)
        except (ImportError, AttributeError) as exc:
            raise RouteError(f"{api}: backend not loaded") from exc
        proxy = getattr(entry, "aklr_proxy", None)
        backend_fn = entry if proxy is None else proxy.backend_fn
        bind_fn = self.bind_fn(backend_fn)
        parameters = inspect.signature(bind_fn).parameters.values()
        outputs = contract.outputs

        values = {}
        for p in parameters:
            name = self.params.get(p.name)
            if name is None:
                if p.default is p.empty:
                    values[p.name] = None
            elif name not in outputs:
                value = call[name]
                members = getattr(type(p.default), "__members__", None)
                if members is not None and isinstance(value, str):
                    for member in members.values():
                        if attr_value(member) == value:
                            value = member
                            break
                    else:
                        spellings = tuple(attr_value(m) for m in members.values())
                        raise RouteDeclinedError(api, name, value, spellings)
                values[p.name] = value

        # What the route pins, and what it cannot pass, must be what the call says.
        for name, const in self.consts.items():
            if not _same(call[name], const):
                raise RouteDeclinedError(api, name, call[name], const)
        firing = {
            name: rule
            for name, rule in self.env_attrs.items()
            if attr_value(call[rule.selector]) == rule.when
        }
        passed = {*self.params.values(), *self.consts, *firing, *outputs}
        if "out" in self.params.values():
            passed.add("out_dtype")  # the forwarded out tensor carries the dtype
        for p in contract.api_params:
            value = call[p.name]
            if p.name not in passed and value is not None and not _same(value, p.default):
                raise RouteDeclinedError(api, p.name, value, p.default)

        # The environment the backend reads the attr from; earlier variables win.
        env: dict[str, str | None] = {}
        for name, rule in firing.items():
            for i, (var, value) in enumerate(rule.env):
                if _same(value, call[name]):
                    env.update((earlier, None) for earlier, _ in rule.env[:i])
                    env[var] = "1"
                    break
            else:
                expected = tuple(value for _, value in rule.env)
                raise RouteDeclinedError(api, name, call[name], expected)

        if self.out_mode != "value":
            missing = [o for o in outputs if call[o] is None]
            if missing:
                device = torch.device("cuda", torch.cuda.current_device())
                codec = FastKeyCodec.from_contract(contract)
                key = FastKeyFnGenerator(codec, device).generate()(call)
                if key is None:
                    raise RouteDeclinedError(api, missing[0], None, "a call inside the contract")
                AllocOutputsFnGenerator(codec, device).generate()(key, call)
            values.update((b, call[c]) for b, c in self.params.items() if c in outputs)

        args, kwargs = [], {}
        for p in parameters:
            if p.name in values:
                if p.kind is p.POSITIONAL_ONLY:
                    args.append(values[p.name])
                else:
                    kwargs[p.name] = values[p.name]
        return BackendCall(backend_fn, tuple(args), kwargs, MappingProxyType(env), self.out_mode, api)

    def bind_fn(self, entry: Callable) -> Callable:
        """What a call of ``entry`` binds by: ``entry``, checked against the route, or
        the route itself, checked against the torch op, when torch_compile_guard hides
        aiter's params."""
        kinds = [p.kind for p in inspect.signature(entry).parameters.values()]
        if kinds == [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]:
            # The guard registers the hidden function as torch.ops.aiter.<name>.
            op = getattr(torch.ops.aiter, self.api.rpartition(".")[2], None)
            if op is not None:
                hidden = [a.name for a in op.default._schema.arguments if a.name != "dummy"]
                declared = list(inspect.signature(self.declared).parameters)
                if hidden != declared:
                    raise RouteError(
                        f"{self.api}: the route no longer describes the entry point.\n"
                        f"  entry point: {hidden}\n  route:       {declared}\n"
                        f"  update {_where(self.declared)} to match, then decide what "
                        "the difference means for the op."
                    )
            return self.declared
        check_signature(self.api, entry, self.declared)
        return entry

    def proxy(self, entry: Callable) -> Callable:
        """``entry``'s stand-in, which offers each call to the op."""
        proxy = Proxy(self, entry)

        @functools.wraps(entry)
        def stand_in(*args, **kwargs):
            return proxy.call(args, kwargs)

        stand_in.aklr_proxy = proxy  # type: ignore[attr-defined]
        return stand_in


# The seam must not import `aklr.ops`, so op api modules register on import.
_ROUTES: dict[str, ProxyRoute] = {}


def route_for(api: str) -> ProxyRoute | None:
    """The route declared for this entry point, or None if nobody claims it."""
    return _ROUTES.get(api)


def routes() -> tuple[ProxyRoute, ...]:
    """Every declared route, in declaration order."""
    return tuple(_ROUTES.values())


def unused(*_: Any) -> None:
    """Declare that these entry-point parameters reach no op, whatever their value."""


def must_default(*_: Any) -> None:
    """Declare that AKLR serves this entry point only at these parameters' defaults."""


def none_if_falsy(*_: Any) -> None:
    """Declare that the backend reads these forwarded scalars by truthiness, so a falsy
    one (0, 0.0, False) reaches the op as None, which the contract reads as unset."""


def attr_from_env(selector: Any, *, when: str | int | bool, env: dict[str, str | int | bool]) -> Any:
    """Declare that an attr comes from the backend's environment when ``selector == when``."""


def aklr_route(api: str) -> Callable[[Callable], Callable]:
    """Declare that ``api`` reaches an op through this function's body."""
    if not api:
        raise RouteError("@aklr_route needs the entry point's dotted name")

    def decorate(fn: Callable) -> Callable:
        try:
            source = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError) as exc:  # pragma: no cover - REPL-defined route
            raise RouteError(f"{_where(fn)}: source is not available to read: {exc}") from exc
        route = RouteSyntax.parse(ast.parse(source).body[0], fn).route(api)
        existing = _ROUTES.get(route.api)
        if existing is not None and existing != route:
            raise RouteError(f"two routes for '{route.api}': op '{existing.op}' and op '{route.op}'")
        _ROUTES[route.api] = route
        return fn

    return decorate


# --- reading a route ----------------------------------------------------------


class RouteSyntax(NamedTuple):
    unused_names: tuple[str, ...]
    must_default_names: tuple[str, ...]
    none_if_falsy_names: tuple[str, ...]
    op_call: ast.Call  # its func is an ast.Name
    is_returned: bool
    fn: Callable

    @classmethod
    def parse(cls, node: ast.stmt, fn: Callable) -> RouteSyntax:
        """The syntax of ``fn``'s route ``node``; ``RouteError(f"{where}: ...")`` if malformed."""
        where = _where(fn)
        fail = lambda message: RouteError(f"{where}: {message}")
        if not isinstance(node, ast.FunctionDef):
            raise fail("a route must be a plain def")
        statements = list(node.body)
        if (
            isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)
        ):
            statements = statements[1:]
        names: dict[str, list[str]] = {
            unused.__name__: [],
            must_default.__name__: [],
            none_if_falsy.__name__: [],
        }
        call: ast.Call | None = None
        is_returned = False
        for statement in statements:
            if isinstance(statement, ast.Expr | ast.Return) and isinstance(statement.value, ast.Call):
                called = statement.value
                verb = called.func.id if isinstance(called.func, ast.Name) else ""
                if isinstance(statement, ast.Expr) and verb in names:
                    for arg in called.args:
                        if not isinstance(arg, ast.Name):
                            raise fail(f"{verb}() takes parameter names, nothing else")
                        names[verb].append(arg.id)
                    continue
                if call is not None:
                    raise fail("a route makes exactly one call to its op")
                call, is_returned = called, isinstance(statement, ast.Return)
                continue
            raise fail(
                "a route body is one call to its op, plus unused() / must_default() "
                f"for what it drops -- found {type(statement).__name__.lower()}. "
                "Anything that computes belongs in a kernel."
            )
        if call is None:
            raise fail("a route body must call its op")
        if not isinstance(call.func, ast.Name):
            raise fail("call the op by a plain name: import it into the route module")
        return cls(
            tuple(names[unused.__name__]),
            tuple(names[must_default.__name__]),
            tuple(names[none_if_falsy.__name__]),
            call,
            is_returned,
            fn,
        )

    def route(self, api: str) -> ProxyRoute:
        """The route this syntax declares for the entry point ``api``."""
        fn, op_call = self.fn, self.op_call
        api_fn = (fn.__globals__ or {}).get(op_call.func.id)
        if api_fn is None:
            raise self._fail(f"'{op_call.func.id}' is not defined in the route's module")
        try:
            contract = ContractRegistry.default().for_api(api_fn)
        except UnknownContractError:
            raise self._fail(
                f"'{op_call.func.id}' is not an op: only the api function of a contract can be bound to",
            ) from None
        unused_names, must_default_names = self.unused_names, self.must_default_names
        op = contract_name(contract.contract_id)
        attrs = {a.param_name: a for a in contract.attrs.values() if a.param_name}
        params, consts, env_attrs = self._keywords(contract, attrs)

        # Every argument the entry point takes was decided about.
        accounted = set(params) | set(unused_names) | set(must_default_names)
        declared_params = inspect.signature(self.fn).parameters
        missing = [p for p in declared_params if p not in accounted]
        if missing:
            raise self._fail(
                f"{', '.join(repr(p) for p in missing)} reach nothing. Forward each "
                f"to '{op}', name it in unused() if dropping it cannot change the "
                "answer, or in must_default() if setting it means AKLR should stand "
                "aside.",
            )
        unknown = [p for p in unused_names + must_default_names if p not in declared_params]
        if unknown:
            raise self._fail(f"{', '.join(unknown)}: not a parameter of this route")
        both = [p for p in must_default_names if p in unused_names]
        if both:
            raise self._fail(
                f"{', '.join(repr(p) for p in both)} in both unused() and "
                "must_default(): AKLR either drops it or stands aside for it",
            )
        no_default = [p for p in must_default_names if declared_params[p].default is inspect.Parameter.empty]
        if no_default:
            raise self._fail(
                f"must_default({', '.join(no_default)}): a parameter without a default "
                "has no default to be at",
            )
        unforwarded = [p for p in self.none_if_falsy_names if p not in params]
        if unforwarded:
            raise self._fail(
                f"none_if_falsy({', '.join(unforwarded)}): only a parameter forwarded to '{op}' reaches it",
            )

        outputs = contract.outputs
        unwritten = [o for o in outputs if o not in params.values()]
        if len(unwritten) < len(outputs):
            if unwritten:
                raise self._fail(
                    f"forward {', '.join(map(repr, unwritten))} too: the baseline hands "
                    "back every output, and the backend writes only those passed",
                )
            out_mode = "out" if self.is_returned else "dps"
        elif contract.required_outputs:
            raise self._fail(
                f"'{op}' needs its caller's {', '.join(map(repr, contract.required_outputs))}: "
                "forward them, since a route that returns its result has nowhere to put them",
            )
        elif self.is_returned:
            out_mode = "value"
        else:
            raise self._fail(
                "a route that forwards no output must return the op's result, or "
                "the entry point's caller gets None",
            )
        return ProxyRoute(
            api=api,
            op=op,
            api_fn=api_fn,
            params=params,
            consts=consts,
            out_mode=out_mode,
            must_default=must_default_names,
            none_if_falsy=self.none_if_falsy_names,
            declared=fn,
            env_attrs=env_attrs,
        )

    def _keywords(
        self, contract: OpContract, attrs: dict
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, EnvAttr]]:
        op_call = self.op_call
        op = contract_name(contract.contract_id)
        if op_call.args:
            raise self._fail(
                "pass the op's arguments by keyword: the correspondence is the "
                "whole content of a route, and position hides it",
            )
        known = [p.name for p in contract.api_params]
        backend_params = list(inspect.signature(self.fn).parameters)

        params: dict[str, str] = {}
        consts: dict[str, Any] = {}
        from_env: dict[str, ast.Call] = {}
        for keyword in op_call.keywords:
            name = keyword.arg
            if name is None:
                raise self._fail("a route cannot splat **kwargs into its op")
            if name not in known:
                raise self._fail(f"'{op}' has no parameter '{name}'{_suggest(name, known)}")
            value = keyword.value
            if isinstance(value, ast.Name):
                if value.id not in backend_params:
                    raise self._fail(
                        f"'{value.id}' is not a parameter of this route{_suggest(value.id, backend_params)}",
                    )
                if value.id in params:
                    raise self._fail(
                        f"'{value.id}' is forwarded twice, as '{params[value.id]}' and "
                        f"'{name}'; one backend argument is one api argument",
                    )
                params[value.id] = name
            elif isinstance(value, ast.Constant):
                consts[name] = value.value
            elif (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == attr_from_env.__name__
            ):
                from_env[name] = value  # read once every forward is known
            else:
                raise self._fail(
                    f"'{name}=' must be a parameter of this route, a literal or "
                    f"{attr_from_env.__name__}(), not {type(value).__name__.lower()}",
                )
        env_attrs = {name: self._env_attr(name, node, params, attrs) for name, node in from_env.items()}
        return params, consts, env_attrs

    def _env_attr(
        self,
        name: str,
        node: ast.Call,
        params: dict[str, str],
        attrs: dict[str, AttrSpec],
    ) -> EnvAttr:
        verb = attr_from_env.__name__
        keywords = {k.arg: k.value for k in node.keywords}
        when, env = keywords.get("when"), keywords.get("env")
        if not (
            len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and set(keywords) == {"when", "env"}
            and isinstance(when, ast.Constant)
            and isinstance(env, ast.Dict)
            and env.keys
            and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in env.keys)
            and all(isinstance(v, ast.Constant) for v in env.values)
        ):
            raise self._fail(
                f"write '{name}={verb}(<parameter>, when=<literal>, env={{<str>: <literal>}})'",
            )
        if name not in attrs:
            raise self._fail(f"{verb}() sets an attr, and '{name}' is not one")
        selector = params.get(node.args[0].id)
        if selector is None:
            raise self._fail(f"{verb}({node.args[0].id}): not a parameter this call forwards")
        if selector not in attrs:
            raise self._fail(f"{verb}({node.args[0].id}): '{selector}' is not an attr")
        # ``when`` is a value of the selector attr, every ``env`` value one of ``name``'s values.
        for value, of in [
            (when.value, selector),
            *((v.value, name) for v in env.values),
        ]:
            # Type-exact, as contracts compare attrs: ``True`` is not ``1``.
            if (type(value) is bool, value) not in {(type(v) is bool, v) for v in attrs[of].values}:
                raise self._fail(f"{value!r} is not a value of '{of}': {attrs[of].values!r}")
        return EnvAttr(
            selector=selector,
            when=when.value,
            env=tuple((k.value, v.value) for k, v in zip(env.keys, env.values)),
        )

    def _fail(self, message: str) -> RouteError:
        return RouteError(f"{_where(self.fn)}: {message}")


def _where(fn: Callable) -> str:
    code = getattr(fn, "__code__", None)
    if code is None:
        return getattr(fn, "__qualname__", "<route>")
    return f"{code.co_filename}:{code.co_firstlineno}"


def _suggest(name: str, known: list[str]) -> str:
    close = difflib.get_close_matches(name, known, n=1)
    return f" -- did you mean '{close[0]}'?" if close else ""


def check_signature(api: str, entry_point: Callable, declared: Callable) -> None:
    """Raise :class:`RouteError` unless the route's parameters are the entry point's."""
    expected = [
        f"'{p.name}'/{p.kind.name.lower()}" for p in inspect.signature(entry_point).parameters.values()
    ]
    actual = [f"'{p.name}'/{p.kind.name.lower()}" for p in inspect.signature(declared).parameters.values()]
    if expected == actual:
        return

    differences = []
    for backend, bound in zip(expected, actual):
        if backend != bound:
            differences.append(f"  - entry point has {backend}, route has {bound}")
    for extra in expected[len(actual) :]:
        differences.append(f"  - entry point has {extra}, route has nothing")
    for extra in actual[len(expected) :]:
        differences.append(f"  - route has {extra}, entry point has nothing")

    raise RouteError(
        f"{api}: the route no longer describes the entry point.\n"
        f"  entry point: {inspect.signature(entry_point)}\n"
        f"  route:       {inspect.signature(declared)}\n"
        + "\n".join(differences)
        + f"\n  update {_where(declared)} to match, then decide what the "
        "difference means for the op."
    )


# --- forward: the proxy -------------------------------------------------------


class Proxy:
    """AKLR's stand-in for one entry point: offer it to the op."""

    __slots__ = (
        "_check_defaults",
        "_defaults",
        "_dispatcher",
        "_to_api_kwargs",
        "backend_fn",
        "route",
    )

    def __init__(self, route: ProxyRoute, backend_fn: Callable) -> None:
        self.route = route
        self.backend_fn = backend_fn
        contract = ContractRegistry.default().for_api(route.api_fn)
        # The dispatcher the api function calls: the default registry is never replaced.
        self._dispatcher = DispatcherRegistry.default().dispatcher(contract)
        self._defaults = {p.name: p.default for p in contract.api_params}
        bind_fn = route.bind_fn(backend_fn)
        backend_params = inspect.signature(bind_fn).parameters
        api = route.api

        self._check_defaults = None
        if route.must_default:
            defaults = {name: backend_params[name].default for name in route.must_default}
            read = _make_binder(bind_fn, None, list(route.must_default))

            def check(args: tuple, kwargs: dict) -> None:
                for name, value in read(args, kwargs).items():
                    default = defaults[name]
                    # A tensor never reaches `!=`: it is either the default object or the default is None.
                    if value is not default and (default is None or value != default):
                        raise RouteDeclinedError(api, name, value, default)

            self._check_defaults = check

        wanted = [route.params[p] for p in backend_params if p in route.params]
        wanted.extend(name for name in route.consts if name not in wanted)
        binder = _make_binder(bind_fn, route.params, wanted, route.consts)
        if route.none_if_falsy:
            falsy_names = tuple(route.params[p] for p in route.none_if_falsy)
            bind = binder

            def bind_none_if_falsy(args: tuple, kwargs: dict) -> dict[str, Any]:
                api_kwargs = bind(args, kwargs)
                for name in falsy_names:
                    if name in api_kwargs and not api_kwargs[name]:
                        api_kwargs[name] = None
                return api_kwargs

            binder = bind_none_if_falsy
        self._to_api_kwargs = binder
        if route.env_attrs:
            rules = tuple(route.env_attrs.items())

            def bind_with_env(args: tuple, kwargs: dict) -> dict[str, Any]:
                api_kwargs = binder(args, kwargs)
                for name, rule in rules:
                    if attr_value(api_kwargs.get(rule.selector)) != rule.when:
                        continue  # absent: the contract default applies
                    for var, value in rule.env:
                        if os.environ.get(var) == "1":
                            api_kwargs[name] = value
                            break
                    else:
                        raise RouteDeclinedError(api, name, None, rule)
                return api_kwargs

            self._to_api_kwargs = bind_with_env

    def __repr__(self) -> str:
        return f"<Proxy {self.route.api} -> {self.route.op}>"

    def call(self, args: tuple, kwargs: dict) -> Any:
        """Serve from AKLR if the op runs a kernel, else the entry point's own body."""
        # Closed: dispatch would only return its reason.
        if self._dispatcher.table is None:
            return self.backend_fn(*args, **kwargs)
        try:
            if self._check_defaults is not None:
                self._check_defaults(args, kwargs)
            api_kwargs = self._to_api_kwargs(args, kwargs)
        except RouteDeclinedError:
            return self.backend_fn(*args, **kwargs)
        result = self._dispatcher.dispatch(self._defaults | api_kwargs)
        if isinstance(result, FallbackReason):
            return self.backend_fn(*args, **kwargs)
        # The op hands back the destination buffer; out_mode says who sees it.
        return None if self.route.out_mode == "dps" else result


def _make_binder(
    fn: Callable,
    params: dict[str, str] | None,
    wanted: list[str],
    consts: dict[str, Any] | None = None,
) -> Callable[[tuple, dict], dict[str, Any]]:
    # A name the entry point lacks is absent, leaving the contract default.
    consts = consts or {}
    backend_of = {api_param: backend_param for backend_param, api_param in (params or {}).items()}
    sig_params = list(inspect.signature(fn).parameters.values())
    position = {p.name: i for i, p in enumerate(sig_params)}
    defaults = {p.name: p.default for p in sig_params if p.default is not inspect.Parameter.empty}
    known = {p.name for p in sig_params}

    # Defaults go through scope, not as constants: an enum is no constant.
    scope: dict[str, Any] = {"_default": {}}

    bound_keys, bound_values = [], []
    for api_param in wanted:
        if api_param in consts:
            bound_keys.append(ast.Constant(api_param))
            bound_values.append(ast.Constant(consts[api_param]))
            continue
        backend_param = backend_of.get(api_param, api_param)
        if backend_param not in known:
            continue
        scope["_default"][backend_param] = defaults.get(backend_param, None)
        key = ast.Constant(backend_param)
        value = fill_expr("kwargs.get(P, _default[P])", P=key)
        if (index := position.get(backend_param)) is not None:
            value = fill_expr("args[I] if _n > I else V", I=index, V=value)
        bound_keys.append(ast.Constant(api_param))
        bound_values.append(value)
    body = fill_statements(
        "def _bind(args, kwargs):\n    _n = len(args)\n    return BOUND",
        BOUND=ast.Dict(bound_keys, bound_values),
    )
    return compile_closure(
        ast.Module(body, type_ignores=[]),
        "_bind",
        scope,
        f"<binder:{getattr(fn, '__name__', '?')}>",
    )


def aklr_proxy(package: str | None = None) -> None:
    """Replace each routed entry point under ``package`` (all if None) that imports with
    its proxy, under every module-level name it is bound to."""
    stand_ins = {}
    for route in routes():
        if package is not None and not route.api.startswith(f"{package}."):
            continue
        try:
            entry = import_entry(route.api)
        except (ImportError, AttributeError):  # not in this environment
            continue
        if not hasattr(entry, "aklr_proxy"):
            stand_ins[id(entry)] = (
                entry,
                route.proxy(entry),
            )  # entry kept: ids stay unique
    # The names `from x import entry` bound before now, including the entry module's own.
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if type(namespace) is not dict:
            continue
        for name, value in list(namespace.items()):
            if id(value) in stand_ins:
                namespace[name] = stand_ins[id(value)][1]


# --- reverse: the bench's baseline --------------------------------------------


class BackendCall(NamedTuple):
    """An api call spelled for the raw backend body."""

    # the Proxy's backend_fn (never the stand-in), or baseline.py's
    backend_fn: Callable
    args: tuple
    kwargs: dict[str, Any]
    env: Mapping[str, str | None]  # None: unset during the call
    out_mode: str  # the chosen route's; "value" for baseline.py's
    name: str  # what runs, dotted: the chosen route's entry point, or baseline.py's


class NoBaselineError(RouteError):
    """Neither a custom baseline nor a route: the new case is its own baseline."""


def custom_baseline(contract: OpContract) -> Callable | None:
    """The op's ``baseline.py`` function ``contract.baseline_name``, if it has one; it
    takes the api's params and returns the outputs, as the api does."""
    module_name = f"{contract.op_package}.baseline"
    try:
        module = op_module(contract.op_package, "baseline")
    except ModuleNotFoundError as e:
        if e.name != module_name:  # baseline.py exists but its own import failed
            raise
        return None
    return getattr(module, contract.baseline_name, None)


def contract_routes(  # structure: allow S2 reads the module route registry; OpContract cannot own it without an import cycle
    contract: OpContract,
) -> tuple[ProxyRoute, ...]:
    """The contract's routes in declaration order; importing its api registers them."""
    op_module(contract.op_package, "api")
    return tuple(r for r in _ROUTES.values() if api_name(r.api_fn) == contract.api_name)


def baseline_of(contract: OpContract) -> str | None:
    """The defined baseline's dotted name: the custom baseline, else the first route's
    entry point; ``None`` if neither."""
    fn = custom_baseline(contract)
    if fn is not None:
        return f"{fn.__module__}.{fn.__qualname__}"
    found = contract_routes(contract)
    return found[0].api if found else None


def backend_bind(contract: OpContract, call: CallArgs) -> BackendCall:
    """``call`` (api param names and spellings) for the defined baseline: the custom
    baseline (called with the api's params), else the first route that binds."""
    fn = custom_baseline(contract)
    if fn is not None:
        name = f"{fn.__module__}.{fn.__qualname__}"
        return BackendCall(fn, (), dict(call), {}, "value", name)
    found = contract_routes(contract)
    if not found:
        raise NoBaselineError(f"{contract.contract_id}: no baseline")
    declined = None
    for route in found:
        try:
            return route.backend_bind(contract, call)
        except RouteDeclinedError as exc:
            declined = declined or exc
    raise declined


def backend_call(  # structure: allow S2 reads the module route registry; OpContract cannot own it without an import cycle
    contract: OpContract, call: CallArgs
) -> Any:
    """Run ``call`` on the raw backend body under its ``env``; return the api's result."""
    bound = backend_bind(contract, call)
    with apply_env(bound.env):
        result = bound.backend_fn(*bound.args, **bound.kwargs)
    outputs = tuple(call[o] for o in contract.outputs)
    if bound.out_mode == "value":
        results = result if isinstance(result, (tuple, list)) else (result,)
        if len(results) != len(outputs):
            raise RouteError(
                f"{contract.contract_id}: the backend returned {len(results)} "
                f"results for {len(outputs)} outputs"
            )
        outputs = tuple(r if o is None else o.copy_(r) for o, r in zip(outputs, results))
    return outputs[0] if len(outputs) == 1 else outputs


def _same(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return a is b
    if a is b or type(a) is type(b) and a == b:
        return True
    try:
        return attr_value(a) == attr_value(b)
    except (AttributeError, TypeError):
        return False


@contextmanager
def apply_env(env: Mapping[str, str | None]) -> Iterator[None]:
    """Apply ``env`` to ``os.environ`` and restore every touched variable on exit."""
    saved = {var: os.environ.get(var) for var in env}
    try:
        _setenv(env)
        yield
    finally:
        _setenv(saved)


def _setenv(env: Mapping[str, str | None]) -> None:
    for var, value in env.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
