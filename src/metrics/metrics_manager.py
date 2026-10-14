from .mediapipe_metrics import BaseMetrics, MediapipeMetrics

strategy_metrics = {
    "MediaPipeEstimationStrategy": MediapipeMetrics()
}

class MetricsManager:
    @classmethod
    def get_metrics_for(cls, strategy, *args, **kwargs) -> BaseMetrics: 
        return strategy_metrics.get(strategy.__class__.__name__)
