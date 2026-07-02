"""Simple busy-error for the chat concurrency lock."""


class ActorBusyError(RuntimeError):
    pass
