"""Eval-mode context for the v2 scout forward (--keep_rule v2): no stochastic depth, exact
restore of every submodule's `training` flag.

The scout measures tile density with the STUDENT under torch.no_grad (trainer._scout_and_bank).
Only the EMA teacher is ever put in eval mode (trainer `teacher.eval()`). Stochastic depth
(DropPath) reads self.training, so a train-mode scout drops residual branches at random and
scales the survivors by 1/(1 - p). This context puts the whole student in eval mode for the scout
forward and then writes the previous `training` flag of EVERY submodule back from a snapshot.
"""
import contextlib

import torch.nn as nn


def training_flags(model):
    """{qualified name: training flag} for every submodule. Used by the restore check and T2(d)."""
    return {name: m.training for name, m in model.named_modules()}


@contextlib.contextmanager
def scout_eval_mode(model):
    """Eval mode for `model` and every submodule; exact restore on exit.

    The restore writes the snapshot back. It does NOT call .train(), so a submodule that was
    already in eval mode before the scout stays in eval mode after it. `model.modules()` yields
    each module object once, so a shared submodule is restored once.
    """
    snapshot = [(m, m.training) for m in model.modules()]
    try:
        model.eval()                  # nn.Module.train(False): one attribute write per module
        yield model
    finally:
        for m, flag in snapshot:
            m.training = flag         # the same attribute write nn.Module.train() performs
        bad = [type(m).__name__ for m, flag in snapshot if m.training is not flag]
        assert not bad, "scout_eval_mode: training flag not restored on %r" % (bad[:5],)
