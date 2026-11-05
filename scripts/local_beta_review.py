"""Run beta-stability with a local browser reviewer over an exported dataset."""

import argparse
import json
import mimetypes
import os
import secrets
import sqlite3
import subprocess
import webbrowser
from contextlib import closing
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

CHILD = """
import sys
from pathlib import Path
from beta_stability.edge_weights.human_db import human_db
stop = Path(sys.argv[1])
original = human_db.__call__
def review(self, query):
    if stop.exists():
        answers, _ = original(self, [])
        return answers, not bool(answers)
    return original(self, query)
human_db.__call__ = review
human_db.poll_seconds = 0.5
from beta_stability.run import main
main(['--config', sys.argv[2]])
"""

PAGE = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")


class ReviewSession:
    """Own one algorithm process, review database and explicit image allowlist."""

    def __init__(self, dataset, output, process):
        self.dataset = Path(dataset).resolve()
        self.output = Path(output).resolve()
        self.process = process
        self.db = self.output / "reviews.sqlite"
        self.token = secrets.token_urlsafe(24)
        data = json.loads((self.dataset / "annotations.json").read_text(encoding="utf-8"))
        images = {row["uuid"]: row for row in data["images"]}
        self.images, self.observations = {}, {}
        for row in data["annotations"]:
            media = images[row["image_uuid"]]
            path = (self.dataset / media["file_name"]).resolve()
            if not path.is_relative_to(self.dataset) or not path.is_file():
                raise ValueError("Export must include source images inside the dataset directory.")
            self.images[row["uuid"]] = path
            self.observations[row["uuid"]] = {
                "uuid": row["uuid"],
                "width": media["width"],
                "height": media["height"],
                "bbox": row.get("bbox"),
            }

    def connect(self):
        """Never create an empty DB before beta-stability initializes its schema."""
        return sqlite3.connect(self.db.as_uri() + "?mode=rw", uri=True, timeout=10)

    def state(self):
        """Claim the highest-priority pair while preserving an already displayed pair."""
        code = self.process.poll()
        state = {
            "finished": code is not None,
            "exit_code": code,
            "reviewed": 0,
            "pair": None,
            "output": str(self.output),
        }
        if self.db.is_file():
            try:
                with closing(self.connect()) as db, db:
                    db.row_factory = sqlite3.Row
                    db.execute("BEGIN IMMEDIATE")
                    state["reviewed"] = db.execute(
                        "SELECT COUNT(*) FROM image_verification WHERE decision != 'none'"
                    ).fetchone()[0]
                    if code is None and not (self.output / "stop.request").exists():
                        row = db.execute(
                            "SELECT * FROM image_verification WHERE status='in_progress' AND instance_id=? LIMIT 1",
                            (self.token,),
                        ).fetchone()
                        if row is None:
                            row = db.execute(
                                "SELECT * FROM image_verification WHERE status='awaiting' "
                                "ORDER BY priority IS NULL, priority, id LIMIT 1"
                            ).fetchone()
                            if row is not None:
                                db.execute(
                                    "UPDATE image_verification SET status='in_progress',instance_id=?,started_at=? "
                                    "WHERE id=? AND status='awaiting'",
                                    (self.token, datetime.now(timezone.utc).isoformat(), row["id"]),
                                )
                        if row is not None:
                            state["pair"] = {
                                "id": row["id"],
                                "left": self.observations[row["uuid1"]],
                                "right": self.observations[row["uuid2"]],
                            }
            except sqlite3.OperationalError as exc:
                if "no such table" not in str(exc):
                    raise
        results = self.output / "results/beta_stability_annots.json"
        if code == 0 and results.is_file():
            grouped = {}
            for row in json.loads(results.read_text(encoding="utf-8"))["annotations"]:
                grouped.setdefault(str(row["cluster_id"]), []).append(self.observations[row["uuid"]])
            state["clusters"] = [
                {"id": key, "observations": value}
                for key, value in sorted(grouped.items(), key=lambda item: -len(item[1]))
            ]
        return state

    def answer(self, pair_id, decision):
        """Persist only an explicit verdict for a pair claimed by this review session."""
        if decision not in {"correct", "incorrect", "cant_tell"}:
            raise ValueError("Unknown decision.")
        with closing(self.connect()) as db, db:
            changed = db.execute(
                "UPDATE image_verification SET decision=?,status='checked',completed_at=? "
                "WHERE id=? AND status='in_progress' AND instance_id=? AND decision='none'",
                (decision, datetime.now(timezone.utc).isoformat(), pair_id, self.token),
            ).rowcount
        if changed != 1:
            raise ValueError("This pair has already been answered or is no longer active.")


def make_handler(session):
    """Serve a token-scoped local UI without exposing arbitrary filesystem paths."""
    prefix = f"/{session.token}/"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            """Keep polling requests out of the launcher's terminal."""

        def send(self, status, data, content_type="application/json"):
            """Send a response with caching disabled for mutable review state."""
            body = json.dumps(data).encode() if content_type == "application/json" else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            """Render the reviewer, return state, or serve an allowlisted source image."""
            route = urlsplit(self.path).path
            if not route.startswith(prefix):
                return self.send(404, {"error": "Unknown session."})
            route = route[len(prefix) :]
            if route == "":
                return self.send(200, PAGE.replace("__BASE__", json.dumps(prefix)).encode(), "text/html; charset=utf-8")
            if route == "pair":
                return self.send(200, session.state())
            if route.startswith("image/"):
                path = session.images.get(route[6:])
                if path is not None:
                    return self.send(
                        200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    )
            self.send(404, {"error": "Not found."})

        def do_POST(self):
            """Accept same-session decisions or request a graceful algorithm finish."""
            route = urlsplit(self.path).path
            if route not in {prefix + "answer", prefix + "stop"}:
                return self.send(404, {"error": "Unknown session."})
            if self.headers.get("Content-Type") != "application/json":
                return self.send(400, {"error": "Expected JSON."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size.")
                body = json.loads(self.rfile.read(length))
                if route.endswith("/answer"):
                    session.answer(body["id"], body["decision"])
                else:
                    (session.output / "stop.request").write_text("stop", encoding="utf-8")
                self.send(200, {"ok": True})
            except (ValueError, KeyError, TypeError, sqlite3.Error) as exc:
                self.send(409, {"error": str(exc)})

    return Handler


def main():
    """Start a fresh interactive run and keep the review server alive until Ctrl+C."""
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--beta-repo", type=Path, default=Path.home() / "projects/beta-stability")
    parser.add_argument("--beta-python", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error("Review budget must be positive.")
    dataset, repo = args.dataset.resolve(), args.beta_repo.resolve()
    python = args.beta_python or repo / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    if not python.is_file():
        parser.error("Beta-stability Python was not found; provide --beta-python.")
    output = (
        args.output
        or Path(__file__).resolve().parents[1]
        / "temp_data"
        / ("beta_review_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    ).resolve()
    if output.exists():
        parser.error("Choose a new output directory; existing runs are preserved.")
    output.mkdir(parents=True)
    config = yaml.safe_load((dataset / "config.yaml").read_text(encoding="utf-8"))
    data = config["data"]
    data.update(
        annotation_file=str(dataset / data["annotation_file"]),
        images_dir=str(dataset),
        output_path=str(output / "results"),
        ui_db_path=str(output / "reviews.sqlite"),
    )
    for spec in data["embeddings"].values():
        spec["file"] = str(dataset / spec["file"])
    verifiers = config["edge_weights"]["verifier_names"].split()
    config["edge_weights"]["verifier_names"] = " ".join(
        [name for name in verifiers if name not in {"no_human", "simulated_human", "human", "ui_human"}] + ["ui_human"]
    )
    config["stability"].update(max_human_reviews=args.budget, edges_per_review_batch=1)
    config_path = output / "config.yaml"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    session = ReviewSession(dataset, output, None)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(repo), PYTHONIOENCODING="utf-8", MPLCONFIGDIR=str(output / "matplotlib-cache"))
    url = f"http://127.0.0.1:{server.server_port}/{session.token}/"
    with (output / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(python), "-u", "-c", CHILD, str(output / "stop.request"), str(config_path)],
            cwd=output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        session.process = process
        (output / "review-url.txt").write_text(url, encoding="utf-8")
        print(f"Reviewer: {url}\nResults: {output}\nCtrl+C closes the local server.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Closing reviewer. Existing decisions remain in reviews.sqlite.")
        finally:
            (output / "stop.request").write_text("stop", encoding="utf-8")
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
            server.server_close()


if __name__ == "__main__":
    main()
