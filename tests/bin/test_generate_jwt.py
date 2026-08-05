import os
import subprocess
import sys
from pathlib import Path

import jwt

PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT_PATH = PROJECT_ROOT / "bin" / "generate_jwt.py"
JWT_SECRET_KEY = "test-secret-key-must-be-at-least-32-characters"


def test_generate_jwt_script_outputs_a_signed_token() -> None:
    environment = os.environ | {"JWT_SECRET_KEY": JWT_SECRET_KEY}

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--subject",
            "test-user",
            "--expires-in-hours",
            "2",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = jwt.decode(
        result.stdout.strip(), JWT_SECRET_KEY, algorithms=["HS256"]
    )

    assert payload["sub"] == "test-user"
    assert "exp" in payload
