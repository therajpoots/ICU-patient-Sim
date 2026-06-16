"""
TLS Certificate Setup
Generates a self-signed X.509 certificate for HTTPS (TLS 1.3).
Run once before starting the server: python setup_tls.py
"""

import os
import datetime
from pathlib import Path
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend


CERT_DIR = Path("certs")
CERT_FILE = CERT_DIR / "server.crt"
KEY_FILE  = CERT_DIR / "server.key"


def generate_self_signed_cert(
    common_name: str = "localhost",
    valid_days: int = 3650,
):
    """Generate a self-signed RSA-4096 TLS certificate."""
    CERT_DIR.mkdir(exist_ok=True)

    print("🔑 Generating RSA-4096 private key...")
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
        backend=default_backend(),
    )

    print("📜 Building X.509 certificate...")
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME,             "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME,   "Research"),
        x509.NameAttribute(NameOID.LOCALITY_NAME,            "ICU Lab"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME,        "ICU Biosignal Pipeline"),
        x509.NameAttribute(NameOID.COMMON_NAME,              common_name),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=valid_days))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("127.0.0.1"),
                x509.IPAddress(__import__("ipaddress").IPv4Address("127.0.0.1")),
                x509.IPAddress(__import__("ipaddress").IPv4Address("0.0.0.0")),
            ]),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .sign(key, hashes.SHA256(), default_backend())
    )

    # Write private key
    with open(KEY_FILE, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    # Write certificate
    with open(CERT_FILE, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"✅ Certificate saved: {CERT_FILE}")
    print(f"✅ Private key saved: {KEY_FILE}")
    print(f"   Valid for: {valid_days} days ({common_name})")
    print()
    print("⚠️  This is a SELF-SIGNED certificate.")
    print("   Clients must use --insecure or trust this cert.")
    print(f"   To trust it (Windows): Import {CERT_FILE} into 'Trusted Root CAs'")


if __name__ == "__main__":
    if CERT_FILE.exists() and KEY_FILE.exists():
        print(f"✅ Certificates already exist in {CERT_DIR}/")
        print("   Delete them and re-run to regenerate.")
    else:
        generate_self_signed_cert()
