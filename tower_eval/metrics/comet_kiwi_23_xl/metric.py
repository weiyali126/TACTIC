# -*- coding: utf-8 -*-
from tower_eval.metrics.base.comet import QECOMET


class COMETKiwi23XL(QECOMET):
    def __init__(self, **kwargs) -> None:
        super().__init__(model="Unbabel/wmt23-cometkiwi-da-xl", **kwargs)

    @staticmethod
    def metric_name():
        return "comet_kiwi_23_xl"
