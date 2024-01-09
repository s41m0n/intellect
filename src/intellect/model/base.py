from abc import ABC, abstractmethod

from river.base import DriftDetector


class BaseModel(ABC):
    """BaseModel class to define methods that should be required by
    subclasses in order to run the entire methodology.
    """
    @abstractmethod
    def __init__(self, drift_detector: DriftDetector = None) -> None:
        if drift_detector is not None and hasattr(drift_detector, 'clone'):
            drift_detector = drift_detector.clone()
        self.drift_detector = drift_detector

    @abstractmethod
    def learn(self, X: list, y: list, *args, **kwargs) -> tuple[list[float], float]:
        """Method to perform a single learning step

        Args:
            X (list): input data
            y (list): target labels

        Returns:
            tuple[list[float], float]: tuple with predictions and loss value
        """

    @abstractmethod
    def predict(self, X: list, *args, **kwargs) -> list[int]:
        """Function to perform prediction on provided data.

        Args:
            x (list): provided data to predict

        Returns:
            list[int]: list of prediction targets
        """

    @abstractmethod
    def predict_proba(self, X: list, *args, **kwargs) -> list[float]:
        """Function to perform predictions of probabilities

        Args:
            X (list): input data

        Returns:
            list[float]: list of probabilities
        """

    @abstractmethod
    def fit(self, *args, **kwargs) -> dict[str, list[float]] | None:
        """Function to fit the Model.

        Returns:
            dict[str, list[float]] | None: None or potentially the history dictionary
        """

    @abstractmethod
    def clone(self, init: bool = True) -> 'BaseModel':
        """Function to clone the model.

        Args:
            init (bool, optional): true whether to initialize a new model.
                Defaults to True.

        Returns:
            BaseModel: the new model
        """

    @abstractmethod
    def continuous_learning(self, *args, **kwargs) -> tuple[list[int], list[int], list[int]]:
        """Function to perform continuous learning on the provided data.

        Returns:
            tuple[list[int], list[int], list[int]]: tuple containing list of predictions, true values
                and the list of drifts, if any.
        """

    @property
    @abstractmethod
    def is_concept_drift(self) -> bool:
        """Property to check if concept drift in the model has been detected.

        Returns:
            bool: whether a concept drift has been detected
        """

    @abstractmethod
    def concept_react(self, *args, **kwarg) -> None:
        """Function to react to a concept drift"""

    @property
    @abstractmethod
    def prunable(self) -> list[str] | list[object]:
        """Property providing the list of the prunable layers in the model

        Returns:
            list[str] | list[object]: list of prunable layers.
        """

    def learn_one(self, x: list, y: list | float, *args, **kwargs) -> tuple[float, float]:
        """Function to perform one step in the learning process of a single sample

        Args:
            x (list): sample data
            y (list | float): sample label

        Returns:
            tuple[float, float]: predicted value and loss value
        """
        predictions, loss = self.learn(x, y, *args, **kwargs)
        return predictions[0], loss

    def predict_one(self, X: list) -> int:
        """Function to predict provided sample

        Args:
            x (list): sample

        Returns:
            int: inferred label
        """
        return self.predict(X)[0].item()

    def predict_proba_one(self, X: list) -> float | list[float]:
        """Function to predict probabilities of a single sample

        Args:
            x (list): sample to predict

        Returns:
            float: probability/ies
        """
        return self.predict_proba(X)[0]
