from tower_eval.metrics.accuracy.metric import ACCURACY
from tower_eval.metrics.bleu.metric import BLEU
from tower_eval.metrics.bleurt.metric import BLEURT
from tower_eval.metrics.chrf.metric import CHRF
from tower_eval.metrics.comet.metric import COMET
from tower_eval.metrics.comet_kiwi.metric import COMETKiwi
from tower_eval.metrics.comet_kiwi_23_xl.metric import COMETKiwi23XL
from tower_eval.metrics.comet_kiwi_23_xxl.metric import COMETKiwi23XXL
from tower_eval.metrics.errant.metric import ERRANT
from tower_eval.metrics.error_span_detection_f1.metric import ErrorSpanDetectionF1
from tower_eval.metrics.error_span_detection_precision.metric import (
    ErrorSpanDetectionPrecision,
)
from tower_eval.metrics.error_span_detection_recall.metric import (
    ErrorSpanDetectionRecall,
)
from tower_eval.metrics.f1.metric import F1
from tower_eval.metrics.f1_sequence.metric import F1SEQUENCE
from tower_eval.metrics.metricx.metric import MetricX
from tower_eval.metrics.metricx_24.metric import (
    MetricX_24_Large,
    MetricX_24_QE_Large,
    MetricX_24_QE_XL,
    MetricX_24_QE_XXL,
    MetricX_24_XL,
    MetricX_24_XXL,
)
from tower_eval.metrics.metricx_large.metric import MetricXLarge
from tower_eval.metrics.metricx_qe.metric import MetricXQE
from tower_eval.metrics.metricx_qe_large.metric import MetricXQELarge
from tower_eval.metrics.metricx_qe_xxl.metric import MetricXQEXXL
from tower_eval.metrics.metricx_xxl.metric import MetricXXXL
from tower_eval.metrics.pearson.metric import PEARSON
from tower_eval.metrics.perplexity.metric import Perplexity
from tower_eval.metrics.spearman.metric import SPEARMAN
from tower_eval.metrics.ter.metric import TER
from tower_eval.metrics.xcomet_qe_xl.metric import XCOMETQEXL
from tower_eval.metrics.xcomet_qe_xxl.metric import XCOMETQEXXL
from tower_eval.metrics.xcomet_xl.metric import XCOMETXL
from tower_eval.metrics.xcomet_xxl.metric import XCOMETXXL
from tower_eval.metrics.xml_chrf.metric import XML_CHRF
from tower_eval.metrics.xml_match.metric import XML_MATCH

__all__ = [
    TER,
    BLEU,
    XCOMETXL,
    XCOMETQEXL,
    XCOMETXXL,
    XCOMETQEXXL,
    COMET,
    COMETKiwi,
    COMETKiwi23XL,
    COMETKiwi23XXL,
    BLEURT,
    CHRF,
    ERRANT,
    F1,
    F1SEQUENCE,
    ACCURACY,
    PEARSON,
    SPEARMAN,
    ErrorSpanDetectionF1,
    ErrorSpanDetectionRecall,
    ErrorSpanDetectionPrecision,
    Perplexity,
    MetricXLarge,
    MetricXQELarge,
    MetricX,
    MetricXQE,
    MetricXQEXXL,
    MetricXXXL,
    MetricX_24_Large,
    MetricX_24_XL,
    MetricX_24_XXL,
    MetricX_24_QE_Large,
    MetricX_24_QE_XL,
    MetricX_24_QE_XXL,
    XML_CHRF,
    XML_MATCH,
]


available_metrics = {metric.metric_name(): metric for metric in __all__}
