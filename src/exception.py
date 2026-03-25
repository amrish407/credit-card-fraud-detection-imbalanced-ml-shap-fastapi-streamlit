class FraudDetectionException(Exception):
    """Project level custom exception with optional context."""

    def __init__(self, message: str, context: str | None = None):
        self.message = message
        self.context = context
        super().__init__(self.__str__())

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | Context: {self.context}"
        return self.message
