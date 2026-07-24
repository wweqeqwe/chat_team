"""CLI entry point for ``chat-team-admin``.

Sub-commands:
  add-user <name>    Prompt for a password twice (no echo), hash with
                     pbkdf2_sha256, write to ~/.chat_team/admin/users.json.
  init-certs         Generate a self-signed RSA-2048 cert+key pair to
                     ~/.chat_team/admin/{cert,key}.pem (0600). Valid 1 year.
  serve              Start the HTTPS admin web server (default if no
                     sub-command given).

The settings object is loaded the same way the main bot loads it
(``chat_team.config.load_settings``), so the admin process reads the same
config.yaml as the bot — the ``admin:`` block there is the source of truth
for host/port/tls/session-timeout etc.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import logging
import os
import sys
from pathlib import Path

from ..config import load_settings
from ..paths import resolve_home
from .auth import PBKDF2_ALGO_TAG, User, UserStore

log = logging.getLogger(__name__)


def _admin_dir(home: Path) -> Path:
    d = home / "admin"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _users_path(home: Path) -> Path:
    return _admin_dir(home) / "users.json"


def _default_cert_path(home: Path) -> Path:
    return _admin_dir(home) / "cert.pem"


def _default_key_path(home: Path) -> Path:
    return _admin_dir(home) / "key.pem"


# --------------------------------------------------------------------------
# add-user
# --------------------------------------------------------------------------

def cmd_add_user(args: argparse.Namespace) -> int:
    home = resolve_home()
    store = UserStore(_users_path(home))
    username = args.name.strip()
    if not username:
        print("username must not be empty", file=sys.stderr)
        return 2

    print(f"Creating admin user '{username}'.")
    pw1 = getpass.getpass("Password: ")
    pw2 = getpass.getpass("Confirm password: ")
    if pw1 != pw2:
        print("passwords do not match", file=sys.stderr)
        return 1
    if len(pw1) < 8:
        print("password must be at least 8 characters", file=sys.stderr)
        return 1

    salt_hex, iterations, hash_hex = UserStore.hash_password(pw1)
    user = User(
        username=username,
        algo=PBKDF2_ALGO_TAG,
        iterations=iterations,
        salt=salt_hex,
        hash=hash_hex,
    )
    store.add_or_update(user)
    print(f"OK: user '{username}' written to {store.path}")
    print(f"  current users: {', '.join(store.list_users()) or '(none)'}")
    return 0


# --------------------------------------------------------------------------
# init-certs
# --------------------------------------------------------------------------

def cmd_init_certs(args: argparse.Namespace) -> int:  # noqa: ARG001
    home = resolve_home()
    cert_path = _default_cert_path(home)
    key_path = _default_key_path(home)
    if cert_path.exists() or key_path.exists():
        if not args.force:
            print(
                f"cert/key already exist at {cert_path} / {key_path}; "
                f"use --force to overwrite.",
                file=sys.stderr,
            )
            return 1
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        print(
            "cryptography library not installed (it's already a chat_team "
            "dep — install with: pip install cryptography)",
            file=sys.stderr,
        )
        return 1

    print("Generating RSA-2048 key pair...")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "chat-team-admin"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "chat_team"),
    ])
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + _dt.timedelta(days=365))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    cert_path.write_bytes(cert_pem)
    key_path.write_bytes(key_pem)
    try:
        os.chmod(key_path, 0o600)
        os.chmod(cert_path, 0o644)
    except OSError:
        pass
    print(f"OK: self-signed cert written (1 year validity):")
    print(f"  cert: {cert_path}")
    print(f"  key:  {key_path}")
    print("  (browsers will warn about the self-signed cert; trust it or")
    print("   upgrade to a Let's Encrypt cert — see README for instructions.)")
    return 0


# --------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:  # noqa: ARG001
    from .server import serve as _serve  # lazy import — keeps startup cheap
    return _serve()


# --------------------------------------------------------------------------
# top-level
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chat-team-admin",
        description="chat_team backend admin panel (HTTPS web UI)",
    )
    sub = parser.add_subparsers(dest="command")

    p_add = sub.add_parser("add-user", help="create or update an admin login user")
    p_add.add_argument("name", help="username (any non-empty string)")
    p_add.set_defaults(func=cmd_add_user)

    p_init = sub.add_parser("init-certs", help="generate a self-signed TLS cert pair")
    p_init.add_argument("--force", action="store_true", help="overwrite if exists")
    p_init.set_defaults(func=cmd_init_certs)

    p_serve = sub.add_parser("serve", help="start the HTTPS admin web server")
    p_serve.set_defaults(func=cmd_serve)

    parser.set_defaults(func=cmd_serve)
    args = parser.parse_args(argv)
    return args.func(args) or 0


def run() -> None:
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    run()
