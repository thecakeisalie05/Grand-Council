"""Reliable local Grand Council launcher with port conflict detection."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

HOST = '127.0.0.1'
PORT_COUNT = 32


def url(port):
    return f'http://{HOST}:{port}'


def occupied(port):
    """Return True if another program is listening on this local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, port))
        except OSError:
            return True
        return False


def is_grand_council(port):
    """Recognize both the original v0.1.0 server and newer releases."""
    try:
        req = urllib.request.Request(url(port) + '/api/health', headers={'Accept': 'application/json'})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=0.6) as res:
            if res.status != 200:
                return False
            data = json.load(res)
        return bool(isinstance(data, dict) and data.get('ok') is True and
                    (data.get('application') == 'grand-council' or
                     (isinstance(data.get('version'), str) and
                      isinstance(data.get('loaded'), list) and
                      isinstance(data.get('max_loaded'), int))))
    except (OSError, ValueError, TimeoutError):
        return False


def starting_port():
    try:
        value = int(os.environ.get('GRAND_COUNCIL_PORT', '8765'))
    except ValueError:
        raise ValueError('GRAND_COUNCIL_PORT must be a number between 1024 and 65535.')
    if value < 1024 or value > 65535 - PORT_COUNT:
        raise ValueError(f'GRAND_COUNCIL_PORT must be between 1024 and {65535 - PORT_COUNT}.')
    return value


def main():
    try:
        start = starting_port()
    except ValueError as exc:
        print(f'Configuration error: {exc}', flush=True)
        return 2

    # If a council was started from an earlier window, don't start a duplicate.
    for port in range(start, start + PORT_COUNT):
        if occupied(port) and is_grand_council(port):
            print(f'Grand Council is already running at {url(port)}', flush=True)
            webbrowser.open(url(port))
            return 0

    for port in range(start, start + PORT_COUNT):
        if occupied(port):
            print(f'Port {port} is occupied by another application; trying the next port.', flush=True)
            continue

        env = os.environ.copy()
        env['GRAND_COUNCIL_PORT'] = str(port)
        print(f'Starting Grand Council at {url(port)}', flush=True)
        proc = subprocess.Popen([sys.executable, '-m', 'grand_council.server'], env=env)
        try:
            # Wait until startup completed before opening the browser.
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    print(f'Server exited (code {proc.returncode}).', flush=True)
                    # A competing instance might have claimed the port just now.
                    if is_grand_council(port):
                        webbrowser.open(url(port))
                        return 0
                    break
                if is_grand_council(port):
                    print(f'Grand Council ready: {url(port)}', flush=True)
                    webbrowser.open(url(port))
                    print('Close this window or press Ctrl+C to stop this instance.', flush=True)
                    return proc.wait()
                time.sleep(0.2)
            else:
                print('Server startup timed out.', flush=True)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        except KeyboardInterrupt:
            print('\nStopping Grand Council...', flush=True)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            return 0

    print(f'Could not start Grand Council: ports {start}-{start + PORT_COUNT - 1} are unavailable '
          'or the server failed to start. Review the errors above.', flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
