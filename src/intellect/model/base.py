from abc import ABC, abstractmethod


class BaseModel(ABC):
    """BaseModel class to define methods that should be required by
    subclasses in order to run the entire methodology.
    """

    @abstractmethod
    def learn(self, X: list, y: list, *args, **kwargs) -> tuple[list[float], float]:
        """Method to perform a single learning step

        Args:
            X (list): input data
            y (list): target labels

        Returns:
            tuple[list[float], float]: tuple with predictions and loss value
        """
        pass

    @abstractmethod
    def predict(self, x: list, *args, **kwargs) -> list[int]:
        """Function to perform prediction on provided data.

        Args:
            x (list): provided data to predict

        Returns:
            list[int]: list of prediction targets
        """
        pass

    @abstractmethod
    def predict_proba(self, x: list, *args, **kwargs) -> list[float]:
        """Function to perform predictions of probabilities

        Args:
            x (list): input data

        Returns:
            list[float]: list of probabilities
        """
        pass

    @abstractmethod
    def fit(self, *args, **kwargs) -> dict[str, list[float]] | None:
        """Function to fit the Model.

        Returns:
            dict[str, list[float]] | None: None or potentially the history dictionary
        """
        pass

    @abstractmethod
    def clone(self, init: bool = True) -> "BaseModel":
        """Function to clone the model.

        Args:
            init (bool, optional): true whether to initialize a new model.
                Defaults to True.

        Returns:
            BaseModel: the new model
        """
        pass

    @abstractmethod
    def continuous_learning(self, *args, **kwargs) -> tuple[list[int], list[int]]:
        """Function to perform continuous learning on the provided data.

        Returns:
            tuple[list[int], list[int]]: tuple containing list of predictions and true values
        """
        pass

    @property
    @abstractmethod
    def is_concept_drift(self) -> bool:
        """Property to check if concept drift in the model has been detected.

        Returns:
            bool: whether a concept drift has been detected
        """
        pass

    @abstractmethod
    def concept_react(self, *args, **kwarg) -> None:
        """Function to react to a concept drift"""
        pass

    @property
    @abstractmethod
    def prunable(self) -> list[str] | list[object]:
        """Property providing the list of the prunable layers in the model

        Returns:
            list[str] | list[object]: list of prunable layers.
        """
        pass

    def learn_one(self, x: list, y: list | float, **kwargs) -> tuple[float, float]:
        """Function to perform one step in the learning process of a single sample

        Args:
            x (list): sample data
            y (list | float): sample label

        Returns:
            tuple[float, float]: predicted value and loss value
        """
        predictions, loss = self.learn(x, y)
        return predictions[0], loss

    def predict_one(self, x: list) -> int:
        """Function to predict provided sample

        Args:
            x (list): sample

        Returns:
            int: inferred label
        """
        return self.predict(x)[0].item()

    def predict_proba_one(self, x: list) -> float | list[float]:
        """Function to predict probabilities of a single sample

        Args:
            x (list): sample to predict

        Returns:
            float: probability/ies
        """
        return self.predict_proba(x)[0]
