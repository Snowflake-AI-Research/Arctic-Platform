"""Sandbox lifecycle over the in-pod k3s agent-sandbox controller.

One Sandbox CR per rollout, same as prime-rl's sandbox_env runtime: the agent
runs inside the sandbox and reaches the driver only through the cni0 bridge, so
the gateway is its sole route to a model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reference_paths  # noqa: E402

KUBECTL = [str(reference_paths.k3s_dir() / "bin" / "k3s"), "kubectl"]
KUBECONFIG = str(reference_paths.k3s_dir() / "kubeconfig.yaml")
NAMESPACE = "default"

# The bridge address, not loopback: a sandbox pod has no route to the driver's
# 127.0.0.1. This is the same seam prime-rl calls the interception port.
BRIDGE_HOST = "10.42.0.1"


def _kubectl(*args: str, stdin: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*KUBECTL, *args],
        env={"KUBECONFIG": KUBECONFIG, "PATH": "/usr/bin:/bin:/usr/local/bin"},
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


SANDBOX_MANIFEST = """apiVersion: agents.x-k8s.io/v1beta1
kind: Sandbox
metadata:
  name: {name}
  namespace: {namespace}
spec:
  operatingMode: Running
  shutdownPolicy: Delete
  podTemplate:
    spec:
      containers:
        - name: agent
          image: {image}
          command: ["sleep", "infinity"]
          resources:
            # The reference ratios: a rollout is idle most of its life (waiting on the
            # model), so requests are tiny and limits are generous. That
            # overcommit is what lets one node hold ~250 concurrent sandboxes
            # instead of ~15.
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "4"
              memory: "16Gi"
"""


class Sandbox:
    def __init__(self, image: str = "python:3.11-slim", name: str | None = None) -> None:
        self.name = name or f"poc-{uuid.uuid4().hex[:10]}"
        self.image = image
        self._pod: str | None = None

    def create(self, ready_timeout_s: int = 300) -> str:
        manifest = SANDBOX_MANIFEST.format(
            name=self.name, namespace=NAMESPACE, image=self.image
        )
        done = _kubectl("apply", "-f", "-", stdin=manifest)
        if done.returncode != 0:
            raise RuntimeError(f"sandbox apply failed: {done.stderr.strip()}")

        deadline = time.time() + ready_timeout_s
        while time.time() < deadline:
            pod = self._find_pod()
            if pod:
                phase = _kubectl(
                    "get", "pod", pod, "-n", NAMESPACE, "-o", "jsonpath={.status.phase}"
                ).stdout.strip()
                if phase == "Running":
                    self._pod = pod
                    return pod
                if phase in ("Failed", "Unknown"):
                    raise RuntimeError(f"sandbox pod {pod} entered {phase}")
            time.sleep(3)
        raise TimeoutError(f"sandbox {self.name} not ready in {ready_timeout_s}s")

    def _find_pod(self) -> str | None:
        done = _kubectl("get", "pods", "-n", NAMESPACE, "-o", "json")
        if done.returncode != 0:
            return None
        try:
            items = json.loads(done.stdout).get("items", [])
        except json.JSONDecodeError:
            return None
        for item in items:
            name = item.get("metadata", {}).get("name", "")
            if name.startswith(self.name):
                return name
        return None

    def exec(self, command: str, timeout: int = 900) -> tuple[int, str]:
        if self._pod is None:
            raise RuntimeError("sandbox not created")
        done = _kubectl(
            "exec", self._pod, "-n", NAMESPACE, "--", "bash", "-lc", command, timeout=timeout
        )
        return done.returncode, (done.stdout or "") + (done.stderr or "")

    def write_file(self, path: str, content: str) -> None:
        if self._pod is None:
            raise RuntimeError("sandbox not created")
        # Pipe through stdin rather than kubectl cp: no tar binary needed in the
        # image, and it keeps content out of the process arg list.
        done = _kubectl(
            "exec",
            "-i",
            self._pod,
            "-n",
            NAMESPACE,
            "--",
            "bash",
            "-lc",
            f"mkdir -p $(dirname {path}) && cat > {path}",
            stdin=content,
        )
        if done.returncode != 0:
            raise RuntimeError(f"write {path} failed: {done.stderr.strip()}")

    def delete(self) -> None:
        _kubectl("delete", "sandbox", self.name, "-n", NAMESPACE, "--wait=false")

    def __enter__(self) -> "Sandbox":
        self.create()
        return self

    def __exit__(self, *exc) -> None:
        self.delete()
