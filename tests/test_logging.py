from ssa.train.logging import NoopLogger


def test_noop_logger_accepts_calls():
    logger = NoopLogger()
    logger.log({"loss": 1.0}, step=0)
    logger.finish()


def test_noop_logger_accepts_log_figures():
    from ssa.train.logging import NoopLogger

    logger = NoopLogger()
    logger.log_figures({"panel": object()}, step=0)  # must not raise
