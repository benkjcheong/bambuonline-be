from __future__ import annotations

import json
import logging
import socket
import ssl
import uuid
from collections.abc import Callable
from typing import Any

import paho.mqtt.client as mqtt

log = logging.getLogger(__name__)

MessageHandler = Callable[[dict[str, Any]], None]


def _create_socket_connection(client: mqtt.Client):
    proxy = client._get_proxy()
    addr = (client._host, client._port)
    kwargs = {"timeout": client._connect_timeout}

    # Paho stores the default bind config as ("", 0). Passing that through to
    # socket.create_connection() forces a local bind that can fail on macOS
    # before the outbound connect is even attempted. Only pass source_address
    # when the caller explicitly requested one.
    if client._bind_address or client._bind_port:
        kwargs["source_address"] = (client._bind_address, client._bind_port)

    if proxy:
        return mqtt.socks.create_connection(addr, **kwargs, **proxy)
    return socket.create_connection(addr, **kwargs)


class _PatchedClient(mqtt.Client):
    def _create_socket_connection(self):
        return _create_socket_connection(self)


def build_client(
    printer_ip: str,
    printer_serial: str,
    access_code: str,
    on_payload: MessageHandler,
) -> mqtt.Client:
    client_id = f"app-{uuid.uuid4().hex[:8]}"
    client = _PatchedClient(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        protocol=mqtt.MQTTv311,
    )
    client.username_pw_set("bblp", access_code)
    client.tls_set(cert_reqs=ssl.CERT_NONE, tls_version=ssl.PROTOCOL_TLSv1_2)
    client.tls_insecure_set(True)
    client.reconnect_delay_set(min_delay=1, max_delay=60)

    report_topic = f"device/{printer_serial}/report"
    request_topic = f"device/{printer_serial}/request"

    def on_connect(c: mqtt.Client, _userdata, _flags, reason_code, _props=None):
        if reason_code == 0 or getattr(reason_code, "is_failure", True) is False:
            log.info("connected, subscribing %s", report_topic)
            c.subscribe(report_topic, qos=0)
            # Ask printer for a full status push so we can prime state quickly.
            c.publish(
                request_topic,
                json.dumps({"pushing": {"sequence_id": "0", "command": "pushall"}}),
                qos=0,
            )
        else:
            log.error("connect failed: %s", reason_code)

    def on_disconnect(_c, _userdata, _flags, reason_code, _props=None):
        log.warning("disconnected: %s (auto-reconnect)", reason_code)

    def on_connect_fail(_c, _userdata):
        log.warning("connect attempt failed; retrying")

    def on_message(_c, _userdata, msg: mqtt.MQTTMessage):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            log.warning("malformed mqtt payload: %s", e)
            return
        try:
            on_payload(payload)
        except Exception:
            log.exception("handler error")

    client.on_connect = on_connect
    client.on_connect_fail = on_connect_fail
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    log.info("connecting to %s:8883 as bblp", printer_ip)
    client.connect_async(printer_ip, 8883, keepalive=60)
    return client
