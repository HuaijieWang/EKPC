import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from easydict import EasyDict
from PIL import Image
from timm.models.vision_transformer import VisionTransformer as TimmViT
from torch import nn
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EKPC"))

import trainer
from data.data_manager import DataManager
from models.base import BaseLearner
from models.EKPC import Learner
from models.Imp import HiddenImp
from network.vision_transformer_adapter import VisionTransformer, _load_pretrained
from utils import inc_net
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import accuracy


def tuning():
    return EasyDict(
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=2,
        d_model=8,
    )


def tiny_backbone():
    model = VisionTransformer(
        img_size=16,
        patch_size=8,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_ratio=2,
        num_classes=0,
        tuning_config=tuning(),
    )
    model.out_dim = 8
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".adaptmlp." in name)
    return model


def test_pretrained_conversion_and_freezing(tmp_path):
    torch.manual_seed(7)
    source = TimmViT(
        img_size=16,
        patch_size=8,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_ratio=2,
        num_classes=0,
    ).eval()
    checkpoint = tmp_path / "backbone.pth"
    torch.save(source.state_dict(), checkpoint)
    target = _load_pretrained(tiny_backbone(), "unused", True, checkpoint).eval()
    images = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(target(images), source(images), atol=2e-6, rtol=2e-5)
    assert all(
        parameter.requires_grad == (".adaptmlp." in name)
        for name, parameter in target.named_parameters()
    )
    bad = source.state_dict()
    bad.pop("norm.weight")
    torch.save(bad, checkpoint)
    with pytest.raises(ValueError, match="Incompatible"):
        _load_pretrained(tiny_backbone(), "unused", True, checkpoint)


def test_local_importance_task_averages_and_accumulation():
    adapter = nn.Module()
    adapter.down_proj = nn.Linear(2, 2)
    adapter.up_proj = nn.Linear(2, 3)
    with torch.no_grad():
        adapter.up_proj.weight.fill_(2)
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [2.0, 0.0]]])
    stats = HiddenImp(nn.ModuleList([adapter]), torch.device("cpu"))
    stats.obain_imp([hidden[:1]])
    stats.obain_imp([hidden[1:]])
    stats.finalize_task()
    torch.testing.assert_close(stats.mean_feature_mag[0], torch.tensor([2.5, 0.0]))
    torch.testing.assert_close(stats.util[0], torch.tensor([15.0, 0.0]))
    stats.begin_task()
    stats.obain_imp([hidden])
    stats.finalize_task()
    torch.testing.assert_close(stats.mean_feature_mag[0], torch.tensor([5.0, 0.0]))
    torch.testing.assert_close(stats.util[0], torch.tensor([30.0, 0.0]))


def test_drift_compensates_translation_and_backpropagates():
    learner = BaseLearner({"device": [torch.device("cpu")]})
    old = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    shift = torch.tensor([0.2, -0.5])
    new = (old + shift).requires_grad_()
    means = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    drift = learner.displacement_w(old, new, means, 4.0)
    torch.testing.assert_close(drift, shift.expand_as(drift))
    drift.square().sum().backward()
    assert new.grad is not None and new.grad.norm() > 0
    estimated = learner.displacement(
        old.numpy(), new.detach().numpy(), means.numpy(), 4.0
    )
    np.testing.assert_allclose(
        means.numpy() + estimated, (means + shift).numpy(), atol=1e-6
    )


def test_cosine_margin_loss_is_finite_and_has_gradient():
    logits = torch.tensor([[0.99, -0.9], [-0.9, 0.99]], requires_grad=True)
    value = AngularPenaltySMLoss(s=1000.0, m=0.01)(logits, torch.tensor([1, 0]))
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(logits.grad).all()


def test_task_metrics_follow_actual_boundaries():
    result = accuracy(np.array([2, 3, 0]), np.array([1, 2, 0]), 2, [2, 2])
    assert result["task_id"] == 66.67
    assert result["old"] == 50.0
    assert result["new"] == 0.0


@pytest.fixture
def image_data(tmp_path):
    root = tmp_path / "images"
    rng = np.random.default_rng(9)
    for split in ("train", "test"):
        for label in range(4):
            directory = root / split / f"class_{label}"
            directory.mkdir(parents=True)
            for index in range(3):
                pixels = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(directory / f"{index}.png")
    return root


def manager(root, seed=3):
    result = DataManager("cub", True, seed, 2, 2, data_root=str(root))
    result._train_trsf = [transforms.ToTensor()]
    result._test_trsf = [transforms.ToTensor()]
    return result


def config():
    path = Path(__file__).resolve().parents[1] / "EKPC/exps/adapter_cifar224.json"
    result = json.loads(path.read_text())
    result.update(
        device=[torch.device("cpu")],
        batch_size=3,
        init_epochs=1,
        inc_epochs=1,
        ca_epochs=1,
        num_workers=0,
        samples_per_class=4,
    )
    return result


def test_two_tasks_train_preserve_backbone_and_update_importance(
    monkeypatch, image_data
):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    monkeypatch.setattr(inc_net, "get_convnet", lambda *args: tiny_backbone())
    data = manager(image_data)
    learner = Learner(config())
    frozen = {
        name: value.detach().clone()
        for name, value in learner._network.convnet.named_parameters()
        if not value.requires_grad
    }
    learner.incremental_train(data)
    learner.after_task()
    first_global = learner.global_importance.clone()
    assert learner.eval_task()["top5"] == 100.0
    old = learner._network.copy().freeze()
    with torch.no_grad():
        learner._network.convnet.blocks[0].adaptmlp.down_proj.weight.add_(0.1)
    assert learner.IPR_loss_adapters_with_similarity(old).item() > 0
    original_get_dataset = data.get_dataset

    def current_task_only(indices, source, mode, **kwargs):
        if source == "train":
            assert min(indices) >= learner._known_classes
        return original_get_dataset(indices, source, mode, **kwargs)

    monkeypatch.setattr(data, "get_dataset", current_task_only)
    learner.incremental_train(data)
    assert learner.eval_task()["top5"] == 100.0
    assert learner._class_means.shape == (4, 8)
    assert learner._class_covs.shape == (4, 8, 8)
    assert not torch.equal(first_global, learner.global_importance)
    for name, parameter in learner._network.convnet.named_parameters():
        assert torch.isfinite(parameter).all()
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    assert all(
        (entry["down_W"] >= 0).all() and (entry["up_W"] >= 0).all()
        for entry in learner._importance
    )


def test_two_seeds_have_independent_logs_and_results(monkeypatch, image_data, tmp_path):
    monkeypatch.setattr(inc_net, "get_convnet", lambda *args: tiny_backbone())
    monkeypatch.setattr(
        trainer,
        "DataManager",
        lambda *args, **kwargs: manager(image_data, seed=args[2]),
    )
    args = config()
    args.update(
        seed=[3, 4],
        device=[-1],
        output_dir=str(tmp_path / "logs"),
        init_cls=2,
        increment=2,
    )
    result = trainer.train(copy.deepcopy(args))
    assert len(result["runs"]) == 2
    assert all(len(run["top1"]) == 2 for run in result["runs"])
    logs = list((tmp_path / "logs").rglob("*.log"))
    assert len(logs) == 2
    assert all("Task 1, Epoch" in path.read_text() for path in logs)
    assert not list((tmp_path / "logs").rglob("*.pkl"))
    assert args["device"] == [-1]
    assert (tmp_path / "logs/adapter/cifar224/summary.json").is_file()


@pytest.mark.skipif(
    os.environ.get("EKPC_FULL_VIT_CHECK") != "1",
    reason="Set EKPC_FULL_VIT_CHECK=1 to check the full ViT-B/16 architecture",
)
def test_full_vit_b16_mapping_and_backward(monkeypatch):
    import timm

    torch.set_num_threads(2)
    torch.manual_seed(11)
    source = timm.create_model(
        "vit_base_patch16_224_in21k", pretrained=False, num_classes=0
    ).eval()
    monkeypatch.setattr(
        "network.vision_transformer_adapter.timm.create_model",
        lambda *args, **kwargs: source,
    )
    target = inc_net.SimpleVitNet(
        {"convnet_type": "pretrained_vit_b16_224_in21k_adapter", "ffn_num": 64},
        pretrained=False,
    ).eval()
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(
            target.extract_vector(images), source(images), atol=3e-6, rtol=3e-5
        )
    target.update_fc(10)
    target.train()
    value = AngularPenaltySMLoss()(target(images)["logits"], torch.tensor([3]))
    value.backward()
    assert torch.isfinite(value)
    assert all(
        parameter.grad is None
        for name, parameter in target.convnet.named_parameters()
        if ".adaptmlp." not in name
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in target.convnet.named_parameters()
        if ".adaptmlp." in name
    )
