from ssa.train.logging import NoopLogger


def test_noop_logger_accepts_calls():
    logger = NoopLogger()
    logger.log({"loss": 1.0}, step=0)
    logger.finish()
