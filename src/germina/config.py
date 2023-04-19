from configparser import ConfigParser
from pathlib import Path

config = ConfigParser()
config.read(f"{Path.home()}/.cache.config")
try:  # pragma: no cover
    local_cache_uri = config.get("Storage", "local")
    remote_cache_uri = config.get("Storage", "remote")
except Exception as e:
    print(
        "Please create a config file '.cache.config' in your home folder following the template:\n"
        """[Storage]
local = sqlite+pysqlite:////home/davi/.hdict.cache.db
remote = mysql+pymysql://username:xxxxxxxxxxxxxxxxxxxxx@url/database

[Path]
images_dir = /tmp/"""
    )
