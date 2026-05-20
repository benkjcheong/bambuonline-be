from __future__ import annotations

import ftplib
import logging
import socket
import ssl
from typing import BinaryIO

log = logging.getLogger(__name__)


class _ImplicitFTP_TLS(ftplib.FTP_TLS):
    """ftplib only supports explicit (AUTH TLS) by default. Bambu uses implicit TLS.

    Also overrides ntransfercmd to reuse the control-channel TLS session on the
    data socket — Bambu printers reject data connections without session resumption
    on Python 3.12+ / OpenSSL 1.1.1+.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sock: socket.socket | None = None

    @property
    def sock(self) -> socket.socket | None:
        return self._sock

    @sock.setter
    def sock(self, value: socket.socket | None) -> None:
        if value is not None and not isinstance(value, ssl.SSLSocket):
            value = self.context.wrap_socket(value, server_hostname=self.host)
        self._sock = value

    def ntransfercmd(self, cmd, rest=None):  # type: ignore[override]
        conn, size = ftplib.FTP.ntransfercmd(self, cmd, rest)
        if self._prot_p:  # PROT P engaged → wrap data connection
            session = getattr(self.sock, "session", None) if self.sock is not None else None
            conn = self.context.wrap_socket(
                conn,
                server_hostname=self.host,
                session=session,
            )
        try:
            conn.settimeout(self.timeout)
        except Exception:
            pass
        return conn, size

    def storbinary(self, cmd, fp, blocksize=8192, callback=None, rest=None):  # type: ignore[override]
        # Standard FTP_TLS.storbinary closes the data conn via SSL.unwrap(),
        # which sends close_notify and waits for the peer's close_notify back.
        # Bambu printers never reply -> we hang forever. Skip unwrap; just close.
        self.voidcmd("TYPE I")
        conn = self.transfercmd(cmd, rest)
        try:
            while True:
                buf = fp.read(blocksize)
                if not buf:
                    break
                conn.sendall(buf)
                if callback is not None:
                    callback(buf)
        finally:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        return self.voidresp()


class FtpsUploadError(RuntimeError):
    pass


def _build_bambu_tls_context() -> ssl.SSLContext:
    # Bambu printers use a custom TLS stack: TLS 1.2 only, self-signed,
    # weak ciphers. Python 3.12+ default contexts reject this.
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    try:
        ctx.set_ciphers("DEFAULT@SECLEVEL=0")
    except ssl.SSLError:
        ctx.set_ciphers("DEFAULT")
    return ctx


def upload_3mf(
    *,
    host: str,
    access_code: str,
    src: BinaryIO,
    remote_name: str,
    port: int = 990,
    timeout_sec: float = 60.0,
) -> None:
    context = _build_bambu_tls_context()
    ftp = _ImplicitFTP_TLS(context=context, timeout=timeout_sec)
    sent = 0

    def _on_chunk(buf: bytes) -> None:
        nonlocal sent
        sent += len(buf)
        if sent % (256 * 1024) < 64 * 1024:
            log.info("ftps upload progress sent=%d", sent)

    try:
        ftp.connect(host=host, port=port, timeout=timeout_sec)
        ftp.login(user="bblp", passwd=access_code)
        ftp.prot_p()
        ftp.set_pasv(True)
        log.info("ftps upload start name=%s", remote_name)
        ftp.storbinary(f"STOR {remote_name}", src, blocksize=64 * 1024, callback=_on_chunk)
        log.info("ftps upload done name=%s sent=%d", remote_name, sent)
    except (ftplib.all_errors, OSError, ssl.SSLError) as exc:
        log.warning("ftps upload error after sent=%d: %s", sent, exc)
        raise FtpsUploadError(f"FTPS upload failed: {exc}") from exc
    finally:
        try:
            ftp.quit()
        except Exception:
            try:
                ftp.close()
            except Exception:
                pass
