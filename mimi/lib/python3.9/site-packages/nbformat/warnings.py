"""
Warnings that can be emitted by nbformat.
"""


class MissingIDFieldWarning(FutureWarning):
    """

    This warning is emitted in the validation step of nbformat as we used to
    mutate the structure which is cause signature issues.

    This will be turned into an error at later point.

    We subclass FutureWarning as we will change the behavior in the future.

    """

    pass


class DuplicateCellId(FutureWarning):
    """

    This warning is emitted in the validation step of nbformat as we used to
    mutate the structure which is cause signature issues.

    This will be turned into an error at later point.

    We subclass FutureWarning as we will change the behavior in the future.
    """

    pass
