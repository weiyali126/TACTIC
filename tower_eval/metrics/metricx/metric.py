# -*- coding: utf-8 -*-
from tower_eval.metrics.base.metricx import RefMetricX


class MetricX(RefMetricX):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            model="google/metricx-23-xl-v2p0", tokenizer="google/mt5-xl", **kwargs
        )

    @staticmethod
    def metric_name():
        return "metricx"
