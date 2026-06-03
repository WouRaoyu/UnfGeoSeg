"""Fast, dependency-light correctness tests for the pure-Python components.

Run with: ``python -m pytest tests/`` (or execute directly).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from segment.data.splits import blocked_chainage_split, kfold_cases, leave_one_tunnel_out
from segment.experiments.eval_metrics import (
    boundary_error_1d,
    classwise_metrics,
    reliability_metrics,
)
from segment.fine.loss import ConfidenceConstrainedLoss
from segment.fine.trainer import _DropProbfgChannel
from segment.fine.transunet_wrapper import build_transunet


def test_blocked_chainage_no_overlap_and_buffer():
    ch = np.arange(0, 100, 1.0)
    s = blocked_chainage_split(ch, block_length=20, buffer=3, test_block_stride=2)
    assert set(s["train"]).isdisjoint(s["test"])
    # no training record sits within the buffer distance of a test record
    tc = ch[s["test"]]
    for tr in s["train"]:
        assert np.abs(ch[tr] - tc).min() > 3


def test_leave_one_tunnel_out():
    folds = leave_one_tunnel_out({"A_1": "A", "A_2": "A", "B_1": "B"})
    assert len(folds) == 2
    assert folds[0]["val"] == ["A_1", "A_2"]
    assert "A_1" not in folds[0]["train"]


def test_kfold_cases_balances_single_project_folds():
    cases = [f"case_{i:03d}" for i in range(137)]
    folds = kfold_cases(cases, n_splits=5)
    assert len(folds) == 5
    assert [len(f["val"]) for f in folds] == [28, 28, 27, 27, 27]
    assert [len(f["train"]) for f in folds] == [109, 109, 110, 110, 110]
    assert sorted(c for f in folds for c in f["val"]) == cases


def test_classwise_and_boundary_metrics():
    yt = np.array([0, 1, 1, 2, 2, 0])
    yp = np.array([0, 1, 2, 2, 2, 0])
    m = classwise_metrics(yt, yp, ["a", "b"])
    assert 0.0 <= m["b"]["f1"] <= 1.0
    be = boundary_error_1d(np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 1, 1]), spacing=2.0)
    assert be["mean_boundary_error"] == 2.0


def test_reliability_metrics():
    proba = np.array([[0.9, 0.1], [0.1, 0.9]]).reshape(2, 2, 1, 1)
    r = reliability_metrics(proba)
    assert abs(r["mean_softmax"] - 0.9) < 1e-6


def test_confidence_loss_reduces_to_base_at_lambda0():
    ak = {"features_per_stage": [8, 16], "strides": [[1, 1, 1], [2, 2, 2]]}
    net = build_transunet(2, 2, ak)
    x = torch.randn(1, 2, 16, 16, 16)
    y = net(x)
    assert y.shape == (1, 2, 16, 16, 16)
    target = torch.randint(0, 2, (1, 1, 16, 16, 16))
    probfg = torch.rand(1, 1, 16, 16, 16)
    l0 = ConfidenceConstrainedLoss(2, lambda_kl=0.0)
    assert abs(float(l0(y, target, probfg)) - float(l0.base_loss(y, target))) < 1e-6


def test_drop_probfg_channel_exposes_wrapped_decoder():
    ak = {"features_per_stage": [8, 16], "strides": [[1, 1, 1], [2, 2, 2]]}
    net = build_transunet(3, 2, ak)
    wrapped = _DropProbfgChannel(net, image_channels=3)
    assert wrapped.decoder is net.decoder
    wrapped.decoder.deep_supervision = False

    x = torch.randn(1, 4, 16, 16, 16)
    y = wrapped(x)
    assert y.shape == (1, 2, 16, 16, 16)


def test_soft_target_uses_foreground_probability():
    L = ConfidenceConstrainedLoss(num_classes=2)
    probfg = torch.tensor([[[[[0.1, 0.7, 0.5]]]]]).float()
    P = L.soft_target(probfg)
    assert torch.allclose(P.sum(1), torch.ones_like(P.sum(1)), atol=1e-5)
    assert abs(float(P[0, 0, 0, 0, 0]) - 0.9) < 1e-5
    assert abs(float(P[0, 1, 0, 0, 1]) - 0.7) < 1e-5


def test_resolve_label_path_strict_per_class():
    io = pytest.importorskip("segment.io")
    with TemporaryDirectory() as tmp:
        labels = Path(tmp)
        fallback = labels / "case001.nii.gz"
        fallback.touch()
        assert io.resolve_label_path(labels, "case001", "fracture_zone") == fallback
        try:
            io.resolve_label_path(
                labels,
                "case001",
                "fracture_zone",
                strict_per_class=True,
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("strict per-class lookup should reject fallback labels")


def test_list_predicted_cases_skips_probability_and_probfg_maps():
    predictions = pytest.importorskip("segment.experiments.predictions")
    with TemporaryDirectory() as tmp:
        pred = Path(tmp)
        for name in (
            "case001.nii.gz",
            "prob_case001.nii.gz",
            "probfg_case001.nii.gz",
            "conf_case001.nii.gz",
        ):
            (pred / name).touch()
        assert predictions.list_predicted_cases(pred) == ["case001"]


def test_pseudolabel_probability_is_foreground_probability():
    pseudolabel = pytest.importorskip("segment.coarse.pseudolabel")

    class DummyClassifier:
        num_classes = 1

        def predict_proba(self, X):
            out = np.zeros((X.shape[0], 2), dtype=np.float32)
            out[:, 0] = 0.7
            out[:, 1] = 0.3
            return out

    vol = np.ones((1, 1, 2), dtype=np.float32)
    pl = pseudolabel.generate_pseudolabels(
        [vol],
        DummyClassifier(),
        half_window=(0, 0, 0),
        statistics=("mean",),
    )
    assert np.all(pl.hard == 0)
    assert np.allclose(pl.foreground_probability, 0.3)
    assert np.allclose(pl.confidence, 0.7)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
