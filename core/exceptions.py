class FaceAIException(Exception):
    """Base exception for Face AI service."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


# Face Exceptions
class NoFaceDetectedException(FaceAIException):
    pass


class MultipleFacesDetectedException(FaceAIException):
    pass


class FaceMismatchException(FaceAIException):
    pass


class InvalidImageException(FaceAIException):
    pass


# Employee Exceptions
class EmployeeNotFoundException(FaceAIException):
    pass


class EmployeeAlreadyExistsException(FaceAIException):
    pass


# Storage Exceptions
class StorageException(FaceAIException):
    pass


class EmbeddingNotFoundException(StorageException):
    pass


class MetadataNotFoundException(StorageException):
    pass