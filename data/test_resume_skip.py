"""
CPU unit test for the resume fast-forward in MemoryEfficientShardedPathologyDataset.
No GPU, no real data, no pytest -- run with:  python data/test_resume_skip.py

Verifies the resume skip (a) emits exactly the post-skip tail of the un-skipped stream
(correctness) and (b) does NOT read/decode the skipped images -- the fix. The previous
order (decode, then discard if within the skip) re-decoded ~iter*batch/num_workers images
per worker on every checkpoint resume (~2.9M at iter 112k), a multi-hour silent stall.
"""
import io, os, sys, zipfile, tempfile, shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data.datasets as D
from PIL import Image


def _make_zip(path, colours):
    with zipfile.ZipFile(path, 'w') as zf:
        for i, c in enumerate(colours):
            buf = io.BytesIO()
            Image.new('RGB', (4, 4), c).save(buf, format='WEBP', lossless=True)
            zf.writestr(f'img_{i:04d}.webp', buf.getvalue())


def _bare_dataset(zip_paths, resume_global):
    """A MemoryEfficientShardedPathologyDataset with only the state __iter__ needs, so we
    exercise the shard/skip loop without the heavy __init__ (index scan, real transforms)."""
    ds = D.MemoryEfficientShardedPathologyDataset.__new__(D.MemoryEfficientShardedPathologyDataset)
    ds.rank = 0
    ds.world_size = 1
    ds.zip_files = list(zip_paths)
    ds.corrupted_zip_files = set()
    ds._epoch = 0
    ds.seed = 42
    ds._resume_global = resume_global
    ds.zip_interleave = 4
    ds.last_error_time = 0.0
    ds.error_count = 0
    ds.max_errors_per_minute = 10
    ds._log_corrupt_file = lambda *a, **k: None
    ds.transforms = lambda img: img.getpixel((0, 0))     # identity -> recover the source colour
    return ds


def _run(zips, skip):
    """Return (emitted_colours, decode_count) for a fresh single-worker pass with `skip`.
    Counts PIL decodes by wrapping Image.open (the only decode site)."""
    calls = []
    orig = D.Image.open
    D.Image.open = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        emitted = list(iter(_bare_dataset(zips, skip)))
    finally:
        D.Image.open = orig
    return emitted, len(calls)


def test_resume_skip_cheap_and_correct():
    tmp = tempfile.mkdtemp(prefix='resume_skip_')
    try:
        N_ZIPS, PER = 5, 20
        total = N_ZIPS * PER
        colours = [(i + 1, 0, 0) for i in range(total)]  # unique by R channel
        zips = []
        for z in range(N_ZIPS):
            p = os.path.join(tmp, f'shard_{z}.zip')
            _make_zip(p, colours[z * PER:(z + 1) * PER])
            zips.append(p)

        full, full_decodes = _run(zips, 0)
        assert len(full) == total, f"full pass emitted {len(full)} != {total}"
        assert full_decodes == total, f"full pass should decode all {total}, got {full_decodes}"
        assert len(set(full)) == total, "colours must be unique for identity checks"

        SKIP = 30
        tail, tail_decodes = _run(zips, SKIP)
        # (a) correctness: the resumed run is exactly the post-skip tail of the full stream
        assert tail == full[SKIP:], "resume skip did not reproduce the post-skip stream tail"
        assert len(tail) == total - SKIP, f"expected {total - SKIP} emitted, got {len(tail)}"
        # (b) cheapness (the fix): only emitted images are decoded; the skipped ones are NOT read
        assert tail_decodes == total - SKIP, \
            f"resume skip decoded {tail_decodes}; must decode only {total - SKIP} (skipped not read)"


        # a larger skip that spans multiple shards still lands correctly and stays cheap
        SKIP2 = 85
        tail2, dec2 = _run(zips, SKIP2)
        assert tail2 == full[SKIP2:], "multi-shard resume skip tail mismatch"
        assert dec2 == total - SKIP2, f"decoded {dec2}, expected {total - SKIP2}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1; print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
