"""
CPU unit tests for checkpoint-resume in the pathology dataloader. No GPU, no real data, no
pytest -- run with:  python data/test_resume_skip.py

Covers:
  1. The resume fast-forward emits the post-skip stream tail exactly AND does not decode the
     skipped images (the old order decoded then discarded ~iter*batch/num_workers per worker
     -- a multi-hour silent stall on every resume).
  2. The skip wraps within one pass, so a shard can never be skipped empty.
  3. ProportionalMultiDatasetWrapper hands each sub-dataset only its proportional share of the
     global count, and survives an (over-)skipped-empty sub-dataset instead of dying with
     'generator raised StopIteration' (PEP 479).
"""
import io, os, sys, zipfile, tempfile, shutil, itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data.datasets as D
from PIL import Image


def _make_zip(path, colours):
    with zipfile.ZipFile(path, 'w') as zf:
        for i, c in enumerate(colours):
            buf = io.BytesIO()
            Image.new('RGB', (4, 4), c).save(buf, format='WEBP', lossless=True)
            zf.writestr(f'img_{i:04d}.webp', buf.getvalue())


def _bare_dataset(zip_paths, resume_global, total_images=0):
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
    ds.total_images = total_images                       # 0 -> skip-wrap disabled
    ds.zip_interleave = 4
    ds.last_error_time = 0.0
    ds.error_count = 0
    ds.max_errors_per_minute = 10
    ds._log_corrupt_file = lambda *a, **k: None
    ds.transforms = lambda img: img.getpixel((0, 0))     # identity -> recover the source colour
    return ds


def _make_shards(tmp, n_zips, per):
    colours = [(i + 1, 0, 0) for i in range(n_zips * per)]   # unique by R channel
    zips = []
    for z in range(n_zips):
        p = os.path.join(tmp, f'shard_{z}.zip')
        _make_zip(p, colours[z * per:(z + 1) * per])
        zips.append(p)
    return zips, colours


def _run(zips, skip, total_images=0):
    """Return (emitted_colours, decode_count) for a fresh single-worker pass. Counts PIL
    decodes by wrapping Image.open (the only decode site)."""
    calls = []
    orig = D.Image.open
    D.Image.open = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        emitted = list(iter(_bare_dataset(zips, skip, total_images)))
    finally:
        D.Image.open = orig
    return emitted, len(calls)


# ----------------------------------------------------------------- tests
def test_resume_skip_cheap_and_correct():
    tmp = tempfile.mkdtemp(prefix='resume_skip_')
    try:
        zips, colours = _make_shards(tmp, 5, 20)
        total = len(colours)

        full, full_decodes = _run(zips, 0, total)
        assert len(full) == total and full_decodes == total, (len(full), full_decodes)
        assert len(set(full)) == total, "colours must be unique for identity checks"

        for SKIP in (30, 85):
            tail, decodes = _run(zips, SKIP, total)
            assert tail == full[SKIP:], f"skip {SKIP}: did not reproduce the post-skip tail"
            # cheapness (the core fix): only emitted images are decoded; skipped ones are NOT read
            assert decodes == total - SKIP, \
                f"skip {SKIP}: decoded {decodes}, must decode only {total - SKIP} (skipped not read)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_skip_wraps_within_pass():
    """A skip larger than the shard wraps (skip % pass) instead of skipping everything empty."""
    tmp = tempfile.mkdtemp(prefix='resume_wrap_')
    try:
        zips, colours = _make_shards(tmp, 5, 20)
        total = len(colours)                             # 100
        full, _ = _run(zips, 0, total)
        # skip=130 with a 100-image shard -> wraps to 30 -> emits the same tail as skip=30
        tail, decodes = _run(zips, total + 30, total)
        assert tail == full[30:], "over-long skip did not wrap within one pass"
        assert len(tail) == total - 30 and decodes == total - 30, (len(tail), decodes)
        # with wrap OFF (total_images=0) the same skip would empty the shard -> nothing emitted
        empty, _ = _run(zips, total + 30, 0)
        assert empty == [], "sanity: without the wrap an over-long skip empties the shard"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _bare_wrapper(datasets, names, per_dataset):
    w = D.ProportionalMultiDatasetWrapper.__new__(D.ProportionalMultiDatasetWrapper)
    w.datasets = list(datasets)
    w.dataset_names = list(names)
    w.samples_per_dataset = list(per_dataset)
    w.batch_size_per_gpu = sum(per_dataset)
    w.seed = 42
    w.rank = 0
    return w


def test_wrapper_resume_position_is_proportional():
    """Each sub-dataset is told to skip only its share of the global count, not the full count."""
    d0 = _bare_dataset([], 0)
    d1 = _bare_dataset([], 0)
    w = _bare_wrapper([d0, d1], ['A', 'B'], [10, 30])    # batch 40
    w.set_resume_position(4000)
    assert d0._resume_global == 1000, d0._resume_global   # 4000 * 10/40
    assert d1._resume_global == 3000, d1._resume_global   # 4000 * 30/40


def test_wrapper_survives_overskipped_empty_dataset():
    """An over-skipped (empty-on-restart) sub-dataset must not crash the wrapper generator
    with 'generator raised StopIteration' (PEP 479); the wrapper skips that slot and keeps
    yielding from the healthy dataset."""
    tmp = tempfile.mkdtemp(prefix='resume_wrap_ds_')
    try:
        z_empty, _ = _make_shards(tmp, 1, 5)             # 5 images
        # give it a huge skip with wrap OFF -> it yields nothing, every pass
        ds_empty = _bare_dataset(z_empty, resume_global=10_000, total_images=0)

        htmp = tempfile.mkdtemp(prefix='resume_wrap_h_')
        z_healthy, hcolours = _make_shards(htmp, 2, 10)  # 20 images, skip 0 -> healthy
        ds_healthy = _bare_dataset(z_healthy, resume_global=0, total_images=0)

        w = _bare_wrapper([ds_empty, ds_healthy], ['EMPTY', 'HEALTHY'], [1, 1])
        got = list(itertools.islice(iter(w), 12))        # must not raise
        assert len(got) == 12, f"wrapper stalled: only {len(got)} samples"
        assert set(got) <= set(hcolours), "all samples must come from the healthy dataset"
        shutil.rmtree(htmp, ignore_errors=True)
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
