#!/usr/bin/env python3
"""Generate a local JWT for authenticated API requests."""

import argparse
import os
from datetime import UTC, datetime, timedelta

import jwt
from dotenv import load_dotenv


def positive_integer(value: str) -> int:
    """Parses a positive integer command-line argument."""
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return integer


def parse_arguments() -> argparse.Namespace:
    """Parses JWT generation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="local-dev")
    parser.add_argument(
        "--expires-in-hours", type=positive_integer, default=1
    )
    return parser.parse_args()


def main() -> None:
    """Prints a JWT signed with the configured application secret."""
    load_dotenv()
    secret_key = os.environ.get("JWT_SECRET_KEY")
    if not secret_key:
        raise RuntimeError("JWT_SECRET_KEY must be set in the environment or .env")

    arguments = parse_arguments()
    expires_at = datetime.now(UTC) + timedelta(
        hours=arguments.expires_in_hours
    )
    token = jwt.encode(
        {"sub": arguments.subject, "exp": expires_at},
        secret_key,
        algorithm="HS256",
    )
    print(token)


if __name__ == "__main__":
    main()
