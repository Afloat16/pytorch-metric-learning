import unittest

import pytest
import torch

from pytorch_metric_learning.losses import FastAPLoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord


def softBinning(D, mid, Delta):
    y = 1 - torch.abs(D - mid) / Delta
    return torch.max(torch.tensor([0], dtype=D.dtype).to(D.device), y)


def original_implementation_fast_ap_loss(input, target, num_bins):
    N = target.size()[0]
    assert input.size()[0] == N, "Batch size donesn't match!"

    Y = target.unsqueeze(1)
    Aff = 2 * (Y.t() == Y).type(input.dtype) - 1
    Aff.masked_fill_(torch.eye(N, N).bool().to(input.device), 0)

    I_pos = (Aff > 0).type(input.dtype).to(input.device)
    I_neg = (Aff < 0).type(input.dtype).to(input.device)
    N_pos = torch.sum(I_pos, 1)

    dist2 = 2 - 2 * torch.mm(input, input.t())

    Delta = torch.tensor(4.0 / num_bins).to(input.device)
    Z = torch.linspace(0.0, 4.0, steps=num_bins + 1).to(input.device)
    L = Z.size()[0]
    h_pos = torch.zeros((N, L), dtype=input.dtype).to(input.device)
    h_neg = torch.zeros((N, L), dtype=input.dtype).to(input.device)
    for bin_idx in range(L):
        pulse = softBinning(dist2, Z[bin_idx], Delta)
        h_pos[:, bin_idx] = torch.sum(pulse * I_pos, 1)
        h_neg[:, bin_idx] = torch.sum(pulse * I_neg, 1)

    H_pos = torch.cumsum(h_pos, 1)
    h = h_pos + h_neg
    H = torch.cumsum(h, 1)

    FastAP = h_pos * H_pos / H
    FastAP[torch.isnan(FastAP) | torch.isinf(FastAP)] = 0
    FastAP = torch.sum(FastAP, 1) / N_pos
    FastAP = FastAP[~torch.isnan(FastAP)]
    loss = 1 - torch.mean(FastAP)
    return loss


class TestFastAPLoss(unittest.TestCase):
    def test_fast_ap_loss(self):
        num_bins = 5
        loss_func = FastAPLoss(num_bins)
        ref_emb = torch.randn(32, 32)
        ref_labels = torch.randint(0, 10, (32,))

        for dtype in TEST_DTYPES:
            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

            loss = loss_func(embeddings, labels)
            loss.backward()
            original_loss = original_implementation_fast_ap_loss(
                torch.nn.functional.normalize(embeddings), labels, num_bins
            )
            rtol = helpers.rtol(dtype)
            self.assertTrue(
                torch.isclose(loss.double(), original_loss.double(), rtol=rtol)
            )

            with self.assertRaises(ValueError):
                loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("per_class", [6, 300])
def test_fast_ap_histogram_statistics(dtype, per_class):
    labels = torch.arange(2, device=TEST_DEVICE).repeat_interleave(per_class)
    positive = labels[:, None] == labels[None, :]
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=TEST_DEVICE)
    positive &= ~diagonal
    negative = labels[:, None] != labels[None, :]
    matrix = torch.where(positive, 0.125, 1.25).to(device=TEST_DEVICE, dtype=dtype)
    matrix = matrix.masked_fill(diagonal, 0).requires_grad_()
    embeddings = torch.zeros((len(labels), 2), device=TEST_DEVICE, dtype=dtype)

    loss = FastAPLoss(num_bins=4)(embeddings, labels, mat=matrix)
    (gradient,) = torch.autograd.grad(loss, matrix)
    positives, negatives = per_class - 1, per_class
    denominator = positives + 0.75 * negatives
    expected = 0.125 * 0.75 * negatives / denominator
    torch.testing.assert_close(
        loss.cpu().double(),
        torch.tensor(expected, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-8,
    )
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[diagonal]) == 0
    tolerance = 0.03 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    positive_gradient = 0.75 * negatives / denominator
    negative_gradient = -0.125 * negatives * positives / denominator**2
    positive_count = 2 * per_class * positives
    negative_count = 2 * per_class * negatives
    expected_positive = torch.tensor(positive_gradient / positive_count, dtype=dtype)
    expected_negative = torch.tensor(negative_gradient / negative_count, dtype=dtype)
    torch.testing.assert_close(
        gradient[positive].cpu().double().sum(),
        expected_positive.double() * positive_count,
        rtol=tolerance,
        atol=1e-8,
    )
    torch.testing.assert_close(
        gradient[negative].cpu().double().sum(),
        expected_negative.double() * negative_count,
        rtol=tolerance,
        atol=1e-8,
    )


@pytest.mark.parametrize("separated", [False, True])
def test_fast_ap_half_precision_large_embedding_batch(separated):
    labels = torch.arange(2, device=TEST_DEVICE).repeat_interleave(300)
    embeddings = torch.zeros((600, 2), dtype=torch.float16, device=TEST_DEVICE)
    embeddings[:, 0] = 1
    if separated:
        embeddings[300:, 0] = -1
    embeddings.requires_grad_()
    loss = FastAPLoss(num_bins=4)(embeddings, labels)
    (gradient,) = torch.autograd.grad(loss, embeddings)
    expected = 0 if separated else 300 / 599
    torch.testing.assert_close(
        loss.cpu().double(),
        torch.tensor(expected, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-8,
    )
    assert torch.isfinite(gradient).all()
