"""viewer_ipc.py — one viewer window for all the files.

The viewer (doe_unified_selector.py) runs a tiny server on 127.0.0.1 while it is open and writes where it listens in a
lock file (port + token). Whoever wants a file shown (the launcher's Viewer button) calls `send(paths)` first: if a viewer
answers, the file goes to it as a new tab of the window that is already open; if not (none open, or a stale lock file of
a closed one), the caller starts a new viewer as before.

Only this machine can connect, and a message needs the token of the lock file. Messages carry file paths, nothing else:
the viewer only opens them. The lock file can be moved with the environment variable DOE_VIEWER_LOCK (tests).

    python viewer_ipc.py --selftest
"""
import json
import os
import queue
import secrets
import socket
import sys
import tempfile
import threading

LOCK = os.environ.get("DOE_VIEWER_LOCK") or os.path.join(tempfile.gettempdir(), "doe_unified_viewer.json")
MAX_MESSAGE = 1 << 16


def send(paths, lock: str | None = None) -> bool:
    """Ask the open viewer to show `paths` (new tabs). True if it took them; False if there is none to answer."""
    try:
        with open(lock or LOCK, encoding="utf-8") as fh:
            info = json.load(fh)
        msg = json.dumps({"token": info["token"], "paths": [os.path.abspath(p) for p in paths]}) + "\n"
        with socket.create_connection(("127.0.0.1", int(info["port"])), timeout=1.0) as s:
            s.sendall(msg.encode("utf-8"))
            s.settimeout(3.0)
            return s.recv(16).strip() == b"ok"
    except (OSError, ValueError, KeyError, TypeError):
        return False


class Server:
    """Listens for `send` messages; the viewer polls `poll()` from its own (Tk) thread: nothing touches Tk here."""

    def __init__(self, lock: str | None = None):
        self.lock = lock or LOCK
        self.token = secrets.token_hex(16)
        self.q: queue.Queue = queue.Queue()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        with open(self.lock, "w", encoding="utf-8") as fh:
            json.dump({"port": self.port, "token": self.token, "pid": os.getpid()}, fh)
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:   # closed
                return
            with conn:
                try:
                    conn.settimeout(2.0)
                    data = b""
                    while not data.endswith(b"\n") and len(data) < MAX_MESSAGE:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("token") == self.token and isinstance(msg.get("paths"), list):
                        for p in msg["paths"]:
                            self.q.put(str(p))
                        conn.sendall(b"ok\n")
                except (OSError, ValueError):
                    pass

    def poll(self) -> list:
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        """Stop listening and remove the lock file if it is still ours (a newer viewer may have taken it over)."""
        try:
            self.sock.close()
        except OSError:
            pass
        try:
            with open(self.lock, encoding="utf-8") as fh:
                mine = json.load(fh).get("token") == self.token
            if mine:   # outside the `with`: Windows cannot delete a file that is still open
                os.remove(self.lock)
        except (OSError, ValueError):
            pass


def _selftest():
    lock = os.path.join(tempfile.mkdtemp(prefix="viewer_ipc_"), "lock.json")
    assert send(["a.h5"], lock) is False                       # no viewer: the caller starts one
    srv = Server(lock)
    assert send(["a.h5", "b.h5"], lock) is True
    got = []
    for _ in range(50):                                        # the thread has queued them
        got += srv.poll()
        if len(got) == 2:
            break
        threading.Event().wait(0.02)
    assert got == [os.path.abspath("a.h5"), os.path.abspath("b.h5")], got
    with socket.create_connection(("127.0.0.1", srv.port), timeout=1) as s:   # wrong token: ignored, no "ok"
        s.sendall(b'{"token": "nope", "paths": ["evil.h5"]}\n')
        s.settimeout(2)
        assert s.recv(16) == b""
    assert srv.poll() == []
    srv2 = Server(lock)                                        # a newer viewer takes the lock over
    srv.close()                                                # the old one must not remove the new one's lock
    assert os.path.isfile(lock) and send(["c.h5"], lock) is True and srv2.poll() in ([], [os.path.abspath("c.h5")])
    srv2.close()
    assert not os.path.exists(lock) and send(["a.h5"], lock) is False
    with open(lock, "w", encoding="utf-8") as fh:              # a stale lock (viewer crashed): no answer
        json.dump({"port": 1, "token": "x"}, fh)
    assert send(["a.h5"], lock) is False
    print("viewer_ipc selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
