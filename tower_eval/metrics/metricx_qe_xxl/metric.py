# -*- coding: utf-8 -*-
from tower_eval.metrics.base.metricx import QEMetricX


class MetricXQEXXL(QEMetricX):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            model="google/metricx-23-qe-xxl-v2p0", tokenizer="google/mt5-xxl", **kwargs
        )

    @staticmethod
    def metric_name():
        return "metricx_qe_xxl"
