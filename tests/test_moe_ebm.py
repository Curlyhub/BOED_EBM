import torch
import torch.nn.functional as F

from boedx.models import MoEChangeOfMeasureEnergyNet


def _net(experts=("identity", "standard", "cross")):
    torch.manual_seed(0)
    return MoEChangeOfMeasureEnergyNet(
        hist_dim=5,
        theta_dim=3,
        hidden=8,
        experts=experts,
        router_hidden=6,
    )


def _inputs():
    torch.manual_seed(1)
    hist = F.softmax(torch.randn(4, 5), dim=-1)
    theta = torch.randn(7, 3)
    base_logits = torch.randn(4, 7)
    return hist, theta, base_logits


def _force_router(net, expert_name):
    with torch.no_grad():
        for p in net.router.parameters():
            p.zero_()
        bias = torch.full((len(net.expert_names),), -30.0)
        bias[net.expert_names.index(expert_name)] = 30.0
        net.router[-1].bias.copy_(bias)


def test_moe_log_weight_shapes_and_normalization():
    hist, theta, base_logits = _inputs()
    net = _net()

    log_q, diag = net.forward_log_weights(hist, theta, base_logits)

    assert log_q.shape == (4, 7)
    assert diag["alpha"].shape == (4, 3)
    torch.testing.assert_close(torch.exp(log_q).sum(dim=-1), torch.ones(4))


def test_identity_only_reproduces_base_bank_posterior():
    hist, theta, base_logits = _inputs()
    net = _net(("identity",))

    log_q, _ = net.forward_log_weights(hist, theta, base_logits)

    torch.testing.assert_close(log_q, F.log_softmax(base_logits, dim=-1))


def test_one_hot_standard_router_matches_standard_tilt():
    hist, theta, base_logits = _inputs()
    net = _net()
    _force_router(net, "standard")

    log_q, _ = net.forward_log_weights(hist, theta, base_logits)
    e = net.experts["standard"](hist, theta)
    expected = F.log_softmax(F.log_softmax(base_logits, dim=-1) - e, dim=-1)

    torch.testing.assert_close(log_q, expected)


def test_one_hot_cross_router_matches_cross_tilt():
    hist, theta, base_logits = _inputs()
    net = _net()
    _force_router(net, "cross")

    log_q, _ = net.forward_log_weights(hist, theta, base_logits)
    e = net.experts["cross"](hist, theta)
    expected = F.log_softmax(F.log_softmax(base_logits, dim=-1) - e, dim=-1)

    torch.testing.assert_close(log_q, expected)


def test_moe_forward_energy_is_finite():
    hist, theta, base_logits = _inputs()
    net = _net()

    energy = net(hist, theta, base_logits)

    assert energy.shape == (4, 7)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(net.last_diagnostics["alpha"]).all()
