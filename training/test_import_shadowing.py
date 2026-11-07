"""
CPU regression test: no function may import a module whose top-level name the same function also
uses from module scope. No GPU, no training, no pytest. Run with:
    python training/test_import_shadowing.py

Why: `import torch._dynamo` inside train_dinov2 (commit e31a6b7) bound the name `torch` as a LOCAL
of the whole function, so the first `torch.*` expression in the function raised
UnboundLocalError on every run (baseline, v1 and v2 alike). Python decides local names when it
compiles the function, not when the branch runs, so py_compile cannot catch this. This test walks
the AST of every Python file in the training code and fails on that pattern. It also checks that
the fixed trainer passes and that the broken trainer from commit e31a6b7 fails (when git is
available), so the test is known to detect the bug.
"""
import ast
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = ('training', 'models', 'typicality', 'data', 'configs', 'losses')
SCAN_FILES = ('utils.py', 'run_with_submitit.py')


def _names_used(node):
    """Every bare Name read in a node (not attributes' tails, not stores)."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
    return out


def shadowing_imports(source, filename='<src>'):
    """Return [(function, line, bound name)] for every import inside a function that binds a name
    which is also bound at module scope by an import, which the function reads, and which can be
    read before the import has run: a read on an earlier line, or an import nested in a branch,
    loop, try or with block (then a read after it may execute without it). An import at the top
    of the function body, before every read, is harmless and is not flagged."""
    tree = ast.parse(source, filename)
    module_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                module_imports.add((a.asname or a.name).split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                module_imports.add(a.asname or a.name)
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        reads = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
        top_level = set(id(s) for s in fn.body)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Import):
                continue
            nested = id(node) not in top_level
            for a in node.names:
                bound = (a.asname or a.name).split('.')[0]
                if bound not in module_imports:
                    continue
                my_reads = [r for r in reads if r.id == bound]
                if not my_reads:
                    continue
                read_before = any(r.lineno < node.lineno for r in my_reads)
                if nested or read_before:
                    hits.append((fn.name, node.lineno, bound))
    return hits


def _scan_paths():
    paths = [os.path.join(ROOT, f) for f in SCAN_FILES if os.path.exists(os.path.join(ROOT, f))]
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, _, files in os.walk(base):
            if '__pycache__' in dirpath:
                continue
            for f in files:
                if f.endswith('.py'):
                    paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def test_no_function_local_import_shadows_module_import():
    bad = []
    for p in _scan_paths():
        with open(p, 'r', encoding='utf-8') as fh:
            src = fh.read()
        try:
            hits = shadowing_imports(src, p)
        except SyntaxError as e:
            bad.append((os.path.relpath(p, ROOT), 'SyntaxError', str(e)))
            continue
        for fn, line, name in hits:
            bad.append((os.path.relpath(p, ROOT), line, f"{fn}: 'import {name}...' shadows the module-level '{name}'"))
    assert not bad, "function-local imports shadow module-level names:\n  " + \
        "\n  ".join(f"{p}:{line} {msg}" for p, line, msg in bad)


def test_detector_catches_the_e31a6b7_trainer():
    """The trainer of commit e31a6b7 had `import torch._dynamo` inside train_dinov2. The detector
    must flag it. Skipped (passes) when git or that commit is not available."""
    try:
        src = subprocess.check_output(['git', 'show', 'e31a6b7:training/trainer.py'], cwd=ROOT,
                                      stderr=subprocess.DEVNULL).decode('utf-8')
    except Exception:  # noqa: BLE001 - no git history here
        print("  (git history not available: detector check skipped)")
        return
    hits = shadowing_imports(src, 'e31a6b7:training/trainer.py')
    assert any(fn == 'train_dinov2' and name == 'torch' for fn, _, name in hits), hits


def test_detector_ignores_from_imports_and_unused_names():
    src = (
        "import torch\n"
        "import os\n"
        "def ok_from():\n"
        "    from torch import nn\n"
        "    return torch.zeros(1), nn\n"
        "def ok_unused():\n"
        "    import json\n"
        "    return json.dumps({})\n"
        "def ok_top_of_body():\n"
        "    import torch._dynamo\n"
        "    return torch._dynamo.config\n"
        "def bad_read_before():\n"
        "    x = torch.zeros(1)\n"
        "    import torch._dynamo\n"
        "    return x\n"
        "def bad_nested(flag):\n"
        "    if flag:\n"
        "        import torch._dynamo\n"
        "    return torch.zeros(1)\n"
    )
    hits = shadowing_imports(src)
    assert hits == [('bad_read_before', 14, 'torch'), ('bad_nested', 18, 'torch')], hits


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
