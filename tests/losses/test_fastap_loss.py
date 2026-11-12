import unittest

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
    def test_precomputed_matrix_preserves_precision(self):
        embeddings = torch.eye(4, dtype=torch.float32, device=TEST_DEVICE)
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        mat = torch.tensor(
            [
                [0, 0.25, 0.5, 1],
                [0.25, 0, 1, 0.5],
                [0.5, 1, 0, 0.75],
                [1, 0.5, 0.75, 0],
            ],
            dtype=torch.float64,
            device=TEST_DEVICE,
            requires_grad=True,
        )
        loss = FastAPLoss()(embeddings, labels, mat=mat)
        expected = FastAPLoss()(embeddings.double(), labels, mat=mat)
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(
            torch.autograd.grad(loss, mat)[0], torch.autograd.grad(expected, mat)[0]
        )

    def test_large_positive_histograms(self):
        labels = (torch.arange(1024) // 512).to(TEST_DEVICE)
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                embeddings = torch.eye(2, dtype=dtype, device=TEST_DEVICE)
                embeddings = embeddings.repeat_interleave(512, dim=0).requires_grad_()
                loss = FastAPLoss()(embeddings, labels)
                torch.testing.assert_close(loss, loss.new_zeros(()))
                loss.backward()
                torch.testing.assert_close(
                    embeddings.grad, torch.zeros_like(embeddings)
                )

    def test_large_histogram_gradients_match_float64(self):
        labels = (torch.arange(1024) // 512).to(TEST_DEVICE)
        rows = [[1, 0], [0.9, 0.45], [0.4, 0.95], [0, 1]]
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                embeddings = torch.tensor(rows, dtype=dtype, device=TEST_DEVICE)
                embeddings = embeddings.repeat_interleave(256, dim=0).requires_grad_()
                reference = embeddings.detach().double().requires_grad_()
                loss = FastAPLoss()(embeddings, labels)
                expected = FastAPLoss()(reference, labels)
                original = original_implementation_fast_ap_loss(
                    torch.nn.functional.normalize(reference), labels, 10
                )
                torch.testing.assert_close(expected, original)
                torch.testing.assert_close(
                    loss.double(), expected, rtol=0.02, atol=0.002
                )
                loss.backward()
                expected.backward()
                assert embeddings.grad is not None and reference.grad is not None
                self.assertTrue(torch.isfinite(embeddings.grad).all())
                torch.testing.assert_close(
                    embeddings.grad.double(), reference.grad, rtol=0.05, atol=3e-6
                )

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
            self.assertTrue(torch.isclose(loss, original_loss, rtol=rtol))

            with self.assertRaises(ValueError):
                loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
