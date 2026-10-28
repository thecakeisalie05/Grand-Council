"""Native Windows Qt desktop shell. Starts the local API in-process.

No local Python installation, browser tab or user-managed pip environment is needed
in the packaged executable. Model inference runs in isolated llama-server processes.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.request


def _port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(('127.0.0.1', 0))
        return server.getsockname()[1]


def serve_in_thread():
    import uvicorn
    from grand_council.server import app as application
    port = _port()
    config = uvicorn.Config(application, host='127.0.0.1', port=port,
                            log_level='warning', access_log=False)
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, name='council-api', daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{port}'
    for _ in range(150):
        if not thread.is_alive():
            raise RuntimeError('Grand Council backend failed to start.')
        try:
            with urllib.request.urlopen(endpoint + '/api/health', timeout=.5) as response:
                if json.load(response).get('ok'):
                    return endpoint, instance, thread, application
        except (OSError, ValueError):
            time.sleep(.1)
    raise RuntimeError('Grand Council backend startup timed out.')


def main():
    if '--self-test' in sys.argv:
        from grand_council.server import app
        from grand_council.acceleration import hardware_profile
        from grand_council.server import STATIC
        assert STATIC.joinpath('index.html').is_file()
        assert app.title == 'Grand Council'
        assert isinstance(hardware_profile(), dict)
        return 0
    # Qt imports belong here: API users and headless unit tests do not need Qt.
    from PySide6.QtCore import QUrl, QStandardPaths
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

    qt = QApplication(sys.argv)
    qt.setApplicationName('Grand Council')
    try:
        endpoint, server, thread, application = serve_in_thread()
    except Exception as exc:
        QMessageBox.critical(None, 'Grand Council startup failed', str(exc))
        return 1

    class LocalPage(QWebEnginePage):
        def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
            if is_main_frame and url.host() not in ('127.0.0.1', 'localhost'):
                QDesktopServices.openUrl(url)
                return False
            return super().acceptNavigationRequest(url, navigation_type, is_main_frame)

    win = QMainWindow()
    win.setWindowTitle('Grand Council')
    win.resize(1500, 950)
    win.setMinimumSize(800, 550)
    web = QWebEngineView(win)
    page = LocalPage(web)
    web.setPage(page)
    win.setCentralWidget(web)
    def accept_download(download):
        directory = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
        if directory:
            download.setDownloadDirectory(directory)
        download.accept()

    web.page().profile().downloadRequested.connect(accept_download)
    web.load(QUrl(endpoint))

    def shutdown():
        # Close local model servers before the embedded API terminates.
        application.state.engine.unload()
        application.state.runtime_manager.shutdown()
        server.should_exit = True
        thread.join(timeout=5)

    qt.aboutToQuit.connect(shutdown)
    win.show()
    return qt.exec()


if __name__ == '__main__':
    raise SystemExit(main())
