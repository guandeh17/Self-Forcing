import pytest
import torch

from utils.scheduler import FlowMatchScheduler


@pytest.mark.parametrize("reverse", [False, True])
def test_step_handles_mixed_terminal_samples_independently(reverse):
    scheduler = FlowMatchScheduler(
        num_inference_steps=4,
        shift=1.0,
        sigma_max=1.0,
        sigma_min=0.0,
        reverse_sigmas=reverse,
    )
    sample = torch.zeros(2, 1, 1, 1)
    model_output = torch.ones_like(sample)
    timestep = torch.stack((scheduler.timesteps[0], scheduler.timesteps[-1]))

    result = scheduler.step(model_output, timestep, sample)

    terminal = 1.0 if reverse else 0.0
    expected_next = torch.tensor(
        [scheduler.sigmas[1].item(), terminal], dtype=sample.dtype
    ).view(2, 1, 1, 1)
    expected = expected_next - scheduler.sigmas[[0, -1]].view(2, 1, 1, 1)
    torch.testing.assert_close(result, expected)


def test_step_to_final_explicitly_finishes_every_sample():
    scheduler = FlowMatchScheduler(
        num_inference_steps=4,
        shift=1.0,
        sigma_max=1.0,
        sigma_min=0.0,
    )
    sample = torch.randn(2, 3, 2, 2)
    model_output = torch.randn_like(sample)
    timestep = torch.stack((scheduler.timesteps[0], scheduler.timesteps[1]))

    result = scheduler.step(model_output, timestep, sample, to_final=True)

    expected_sigma = torch.zeros(2, 1, 1, 1)
    expected = sample + model_output * (
        expected_sigma - scheduler.sigmas[:2].view(2, 1, 1, 1)
    )
    torch.testing.assert_close(result, expected)


def test_step_preserves_gradient_for_mixed_schedule():
    scheduler = FlowMatchScheduler(
        num_inference_steps=3,
        shift=1.0,
        sigma_max=1.0,
        sigma_min=0.0,
    )
    sample = torch.zeros(2, 1, 1, 1, requires_grad=True)
    model_output = torch.ones_like(sample)
    timestep = torch.stack((scheduler.timesteps[1], scheduler.timesteps[-1]))

    scheduler.step(model_output, timestep, sample).sum().backward()

    assert sample.grad is not None
    torch.testing.assert_close(sample.grad, torch.ones_like(sample))
