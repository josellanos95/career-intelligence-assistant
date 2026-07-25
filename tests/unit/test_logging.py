import json

from app.infra.observability.logging import configure_logging, get_logger


def test_configure_logging_emits_valid_json(capsys):
    configure_logging("INFO")
    logger = get_logger("test")

    logger.info("something_happened", key="value")

    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "something_happened"
    assert payload["key"] == "value"
    assert payload["level"] == "info"


def test_configure_logging_respects_level(capsys):
    configure_logging("ERROR")
    logger = get_logger("test")

    logger.info("should_be_filtered")
    logger.error("should_appear")

    output = capsys.readouterr().out
    assert "should_be_filtered" not in output
    assert "should_appear" in output
