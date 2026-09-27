from src import GUI

from src.strategies.mediapipe import MediaPipeEstimationStrategy
from src.strategies.estimation_strategy_manager import EstimationStrategyManager, PoseEstimationStrategies

strategy_manager = EstimationStrategyManager()
strategy_manager.register_strategy(PoseEstimationStrategies.MEDIAPIPE, MediaPipeEstimationStrategy)

gui = GUI()
gui.register_pose_strategy_manager(strategy_manager)

gui.mainloop()
