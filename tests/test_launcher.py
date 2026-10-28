"""Startup regressions: existing instance and occupied-port detection."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import launch


def test_in_use_port_is_detected_even_without_http_responses():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.listen(1)
        assert launch.occupied(port)
        assert not launch.is_grand_council(port)


def test_existing_instance_is_reused_without_launching_another(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = json.dumps({'ok': True, 'version': '0.1.0', 'max_loaded': 1, 'loaded': []}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    try:
        monkeypatch.setenv('GRAND_COUNCIL_PORT', str(port))
        with patch('launch.webbrowser.open') as browser, patch('launch.subprocess.Popen') as popen:
            assert launch.main() == 0
            browser.assert_called_once_with(f'http://127.0.0.1:{port}')
            popen.assert_not_called()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
