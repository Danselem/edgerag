import os


def silence_stderr() -> None:
    if os.environ.get("EDGE_RAG_VERBOSE") != "1":
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
