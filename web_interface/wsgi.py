# fmt: off
import os
import sys

from web_interface.app import app as application  # noqa

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))


if __name__ == "__main__":
    application.run()
