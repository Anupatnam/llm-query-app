from loguru import logger
import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_logger():
    config = load_config()
    log_conf = config["logging"]
    if log_conf.get("log_to_file"):
        logger.add(log_conf["file_path"], format=log_conf["format"], level=log_conf["level"])
    else:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), format=log_conf["format"], level=log_conf["level"])
    return logger
