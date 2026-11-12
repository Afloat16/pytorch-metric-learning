import pytest
import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import FastAPLoss


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("size", [16, 300])
def test_fastap_all_positive_batch_has_zero_loss(dtype, size):
    embeddings = torch.tensor([[1.0, 0.0]], dtype=dtype).repeat(size, 1)
    embeddings.requires_grad_()
    loss = FastAPLoss()(embeddings, torch.zeros(size, dtype=torch.long))
    assert loss.dtype == dtype
    torch.testing.assert_close(loss, loss.new_zeros(()))
    loss.backward()
    torch.testing.assert_close(embeddings.grad, torch.zeros_like(embeddings))


def reference_fastap(embeddings, labels, bins):
    normalized = F.normalize(embeddings, dim=1)
    distances = (normalized[:, None] - normalized[None, :]).square().sum(-1)
    positive = labels[:, None] == labels[None, :]
    positive.fill_diagonal_(False)
    negative = labels[:, None] != labels[None, :]
    positive_count = positive.sum(-1)
    cumulative_positive = torch.zeros_like(positive_count, dtype=embeddings.dtype)
    cumulative_total = torch.zeros_like(cumulative_positive)
    average_precision = torch.zeros_like(cumulative_positive)
    for edge in torch.linspace(0, 4, bins + 1, dtype=embeddings.dtype):
        kernel = (1 - (distances - edge).abs() / (4 / bins)).clamp_min(0)
        positive_histogram = (kernel * positive).sum(-1)
        cumulative_positive = cumulative_positive + positive_histogram
        cumulative_total = cumulative_total + (kernel * (positive | negative)).sum(-1)
        denominator = cumulative_total.clamp_min(torch.finfo(embeddings.dtype).tiny)
        average_precision = (
            average_precision + positive_histogram * cumulative_positive / denominator
        )
    return (1 - average_precision / positive_count).mean()


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_fastap_large_mixed_batch_matches_histogram_reference(dtype):
    generator = torch.Generator().manual_seed(43)
    values = torch.randn(600, 3, generator=generator) * 0.03
    values[:, 0] += 1
    embeddings = values.to(dtype).requires_grad_()
    labels = torch.arange(len(values)) % 2
    reference_input = embeddings.detach().double().requires_grad_()
    expected = reference_fastap(reference_input, labels, 5)
    expected_gradient = torch.autograd.grad(expected, reference_input)[0]
    actual = FastAPLoss(num_bins=5)(embeddings, labels)
    gradient = torch.autograd.grad(actual * 128, embeddings)[0].double() / 128
    assert actual.dtype == dtype
    tolerance = (
        2e-2 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 1e-5
    )
    torch.testing.assert_close(
        actual.double(), expected, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        gradient.double(), expected_gradient, rtol=tolerance, atol=1e-6
    )
