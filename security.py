"""Security utilities:
1. PII Redaction: automatic regex redaction of Indian phone numbers (+91 / 10-digit) and personal emails.
2. SSRF Guard: validates URLs to prevent fetching private IPs, localhost, metadata endpoints.
"""
import re
import ipaddress
import socket
from urllib.parse import urlparse
from typing import Tuple

# Regex for email addresses
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
)

# Regex for Indian phone numbers:
# Covers +91 xxxxx xxxxx, 91xxxxxxxxxx, 0xxxxxxxxxx, or standard 10-digit mobile starting with 6, 7, 8, 9
PHONE_PATTERNS = [
    re.compile(r'(?:\+91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}\b'),
    re.compile(r'\b0[6-9]\d{9}\b'),
    re.compile(r'\b(?:\+91[\s\-]?)?[6-9]\d{9}\b'),
]


def redact_pii(text: str) -> str:
    """Redacts Indian phone numbers and email addresses before storing or sending to LLMs."""
    if not text:
        return ""

    # Redact email addresses
    redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)

    # Redact phone numbers
    for pat in PHONE_PATTERNS:
        redacted = pat.sub("[REDACTED_PHONE]", redacted)

    return redacted


def is_ssrf_safe_url(url: str) -> Tuple[bool, str]:
    """Validates if a URL is safe to fetch via HTTP:
    - Must be http or https
    - Hostname must resolve to a public IP
    - Blocks localhost, loopbacks (127.0.0.0/8), private LANs (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
    - Blocks link-local (169.254.0.0/16, cloud metadata services)
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, f"Unsupported scheme: {parsed.scheme}. Only HTTP/HTTPS permitted."

        hostname = parsed.hostname
        if not hostname:
            return False, "Missing hostname in URL."

        # Check for typical forbidden keywords
        lower_host = hostname.lower()
        if lower_host in ("localhost", "127.0.0.1", "0.0.0.0", "metadata.google.internal"):
            return False, f"Target host {hostname} is forbidden."

        # Resolve IP addresses
        addr_info = socket.getaddrinfo(hostname, None)
        for item in addr_info:
            ip_str = item[4][0]
            ip = ipaddress.ip_address(ip_str)

            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                return False, f"Target host {hostname} resolves to non-public IP: {ip_str}"

        return True, "URL is safe"
    except Exception as e:
        return False, f"Failed URL security check: {str(e)}"
