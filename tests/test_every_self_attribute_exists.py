"""Every ``self.<name>`` a class uses exists on that class.

WHY: the review window once called ``self._bundle_needs_review`` after the
method had been deleted, and raised AttributeError the moment it built its
queue. Nothing caught it: no test drove that path, and Python only finds a
missing attribute when the line runs. This test finds the whole class of bug
without running anything.

For every class defined in the review package and in the watcher's
orchestrator, it reads (ast) each ``self.<name>`` the class's methods use --
calls and plain references alike -- and checks that the name is provided:
  * defined in the class body (method, property, class attribute), or
    assigned to ``self`` anywhere in the class (``self.x = ...``,
    ``setattr(self, "x", ...)``), in this class or any of its project bases
    (their source is read the same way); or
  * provided by a base outside the project (a Qt widget's own methods),
    looked up on the imported class.
``getattr(self, "x", default)`` is not a ``self.x`` use and is not checked;
a name the class tests with ``hasattr(self, "x")`` is one it knows may be
absent (a mixin reading its host's attribute), and counts as provided.
"""
import ast
import importlib
import inspect
import textwrap
from pathlib import Path

import pytest

import mousereach

SRC = Path(mousereach.__file__).resolve().parent
MODULES = sorted(
    ["mousereach.review." + p.stem for p in (SRC / "review").glob("*.py")
     if p.stem != "__init__"]
    + ["mousereach.watcher.orchestrator"])


def _is_self(node):
    return isinstance(node, ast.Name) and node.id == "self"


def _provided_in_source(cls) -> set:
    """Names a project class provides by its own source."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    except (OSError, TypeError, SyntaxError):
        return set()
    body = tree.body[0]
    names = set()
    for stmt in body.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
    for node in ast.walk(body):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            targets = [node.optional_vars]
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Attribute) and _is_self(n.value):
                    names.add(n.attr)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("setattr", "hasattr") and len(node.args) >= 2
                and _is_self(node.args[0]) and isinstance(node.args[1], ast.Constant)):
            names.add(node.args[1].value)
    return names


def _used(cls) -> dict:
    """{name: first line} of every ``self.<name>`` load in the class's own methods."""
    src, start = inspect.getsourcelines(cls)
    tree = ast.parse(textwrap.dedent("".join(src)))
    out = {}
    for node in ast.walk(tree.body[0]):
        if (isinstance(node, ast.Attribute) and _is_self(node.value)
                and isinstance(node.ctx, ast.Load)):
            out.setdefault(node.attr, start + node.lineno - 1)
    return out


def _project_class(c) -> bool:
    return (getattr(c, "__module__", "") or "").startswith("mousereach.")


def _classes():
    for name in MODULES:
        mod = importlib.import_module(name)
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ == name:
                yield cls


def _missing(cls) -> list:
    provided = set()
    for base in cls.__mro__:
        if base is cls or _project_class(base):
            provided |= _provided_in_source(base)
    out = []
    for attr, line in sorted(_used(cls).items()):
        if attr in provided or hasattr(cls, attr):
            continue
        out.append("%s.%s (line %d of %s)" % (cls.__name__, attr, line,
                                              Path(inspect.getsourcefile(cls)).name))
    return out


def test_every_self_attribute_a_class_uses_exists():
    missing = [m for cls in _classes() for m in _missing(cls)]
    assert not missing, "self.<name> used but never provided:\n  " + "\n  ".join(missing)


def test_the_checker_catches_a_missing_method():
    class Probe:
        def a(self):
            return self.b()           # b is defined nowhere

        def c(self):
            self.d = 1
            return self.d + self.e    # e is defined nowhere

    assert sorted(m.split(" ")[0] for m in _missing(Probe)) == ["Probe.b", "Probe.e"]
