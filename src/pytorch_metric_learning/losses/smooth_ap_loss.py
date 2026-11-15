from typing import Unpack

import torch
import torch.nn.functional as F
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class SmoothAPLoss(BaseMetricLossFunction):
    supports_mat = True

    def __init__(
        self, temperature: float = 0.01, **options: Unpack[LossOptions]
    ) -> None:
        super().__init__(**options)
        _common.check_distance_type(self, CosineSimilarity)
        self.temperature = temperature

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
        *,
        mat: Tensor | None = None,
    ) -> LossDict:
        labels = _common.labels_required(labels)
        _common.ref_not_supported(ref_emb)
        sims = _common.resolve_matrix(self.distance, embeddings, None, mat)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        counts = same.sum(dim=1)
        rows = torch.where(counts > 1)[0]
        anchors, positives = torch.where(same & (counts > 1).unsqueeze(1))
        differences = sims[anchors] - sims[anchors, positives].unsqueeze(1)
        others = torch.arange(len(labels), device=labels.device) != positives.unsqueeze(
            1
        )
        above = F.sigmoid(differences / self.temperature) * others
        all_ranks = 1 + above.sum(dim=1)
        positive_ranks = 1 + (above * same[anchors]).sum(dim=1)
        precision = positive_ranks / all_ranks
        dtype = _common.accumulation_dtype(sims.dtype)
        sums = torch.zeros(len(labels), device=sims.device, dtype=dtype)
        sums = sums.index_add(0, anchors, precision.to(dtype))
        ap = (sums / counts).to(sims.dtype)
        miner_weights = indexing.convert_to_weights(
            indices_tuple, labels, dtype=ap.dtype
        )
        loss = (1 - ap[rows]) * miner_weights[rows]
        return {"loss": {"losses": loss, "indices": rows}}
