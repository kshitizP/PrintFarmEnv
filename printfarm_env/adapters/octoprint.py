"""OctoPrint/Moonraker mock adapter — logs would-be HTTP calls to stdout."""

from __future__ import annotations

import json
from typing import Optional


class OctoPrintAdapter:
    """
    Drop-in replacement for PrintFarmEnvironment methods that hits a real
    OctoPrint instance. In mock mode (default), it only logs the HTTP calls
    it *would* make — no real network traffic.

    Per ROUND2_MANUAL.md §5:
      "Even a mocked OctoPrintAdapter that logs the would-be HTTP calls is
       enough to prove the design."

    Demo usage:
        adapter = OctoPrintAdapter(host="http://192.168.1.50", api_key="abc123")
        adapter.assign_job(printer_id=1, job_id="vase_pla_0.3mm.gcode")
        adapter.run_diagnostic(printer_id=1)
        adapter.dispatch_ticket(operator_id="op_j1", ticket_type="spool_swap",
                                printer_id=1, material="PLA")

    To point at a real OctoPrint instance, subclass and override _http().
    """

    def __init__(
        self,
        host: str = "http://localhost:5000",
        api_key: str = "",
        mock: bool = True,
    ):
        self.host    = host.rstrip("/")
        self.api_key = api_key
        self.mock    = mock
        self._call_log: list[dict] = []

    # ------------------------------------------------------------------
    #  Action → API methods  (mirrors FarmActionEnum)
    # ------------------------------------------------------------------

    def assign_job(self, printer_id: int, job_id: str) -> None:
        """ASSIGN_JOB → POST /api/files/local/{job_id} (select+print)"""
        self._http("POST", f"/api/files/local/{job_id}.gcode",
                   {"command": "select", "print": True},
                   note=f"printer={printer_id}")
        # Moonraker equivalent
        self._http("POST", "/printer/print/start", {},
                   api="moonraker", note=f"printer={printer_id}")

    def cancel_job(self, job_id: str) -> None:
        """CANCEL_JOB → POST /api/job {command: cancel}"""
        self._http("POST", "/api/job", {"command": "cancel"},
                   note=f"job={job_id}")
        self._http("POST", "/printer/print/cancel", {},
                   api="moonraker", note=f"job={job_id}")

    def pause_job(self, printer_id: int) -> None:
        """PAUSE_JOB → POST /api/job {command: pause, action: pause}"""
        self._http("POST", "/api/job",
                   {"command": "pause", "action": "pause"},
                   note=f"printer={printer_id}")
        self._http("POST", "/printer/print/pause", {},
                   api="moonraker", note=f"printer={printer_id}")

    def resume_job(self, printer_id: int, job_id: str) -> None:
        """RESUME_JOB → POST /api/job {command: pause, action: resume}"""
        self._http("POST", "/api/job",
                   {"command": "pause", "action": "resume"},
                   note=f"printer={printer_id} job={job_id}")
        self._http("POST", "/printer/print/resume", {},
                   api="moonraker", note=f"printer={printer_id}")

    def run_diagnostic(self, printer_id: int) -> None:
        """RUN_DIAGNOSTIC → GET /api/printer + GET /api/printer/tool"""
        self._http("GET",  "/api/printer", {},
                   note=f"printer={printer_id} (temperature + state)")
        self._http("GET",  "/api/printer/tool", {},
                   note=f"printer={printer_id} (hotend temp)")
        # Moonraker: poll all objects
        self._http("GET",  "/printer/objects/query",
                   {"objects": {"heater_bed": None, "extruder": None,
                                "print_stats": None, "endstops": None}},
                   api="moonraker", note=f"printer={printer_id}")

    def dispatch_ticket(
        self,
        operator_id: str,
        ticket_type: str,
        printer_id: int,
        material: Optional[str] = None,
        maintenance_type: Optional[str] = None,
    ) -> None:
        """
        DISPATCH_TICKET / REQUEST_SPOOL_SWAP / REQUEST_MAINTENANCE
        → POST to an external work-order queue (Slack / internal API).
        """
        payload: dict = {
            "operator_id":  operator_id,
            "ticket_type":  ticket_type,
            "printer_id":   printer_id,
        }
        if material:
            payload["material"] = material
        if maintenance_type:
            payload["maintenance_type"] = maintenance_type

        self._http("POST", "/external/ticket-queue", payload,
                   api="internal",
                   note="→ paged to operator device / Slack webhook")

    def override_operator(self, ticket_id: str, reason: str) -> None:
        """OVERRIDE_OPERATOR → DELETE/PATCH ticket in work-order queue"""
        self._http("PATCH", f"/external/ticket-queue/{ticket_id}",
                   {"status": "cancelled", "reason": reason},
                   api="internal")

    # ------------------------------------------------------------------
    #  Telemetry reader  (observation side)
    # ------------------------------------------------------------------

    def get_printer_state(self, printer_id: int) -> dict:
        """Polls OctoPrint for current printer state (used to build observation)."""
        self._http("GET", "/api/printer", {},
                   note=f"printer={printer_id}")
        # In mock mode, return an empty dict; a real impl would parse the response
        return {}

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _http(
        self,
        method:  str,
        path:    str,
        body:    dict,
        api:     str = "octoprint",
        note:    str = "",
    ) -> None:
        entry = {
            "api":    api,
            "method": method,
            "url":    f"{self.host}{path}",
            "body":   body,
        }
        self._call_log.append(entry)

        tag  = f"[{api.upper()}]"
        body_s = json.dumps(body) if body else ""
        note_s = f"  # {note}" if note else ""
        print(f"{tag} {method:5s} {self.host}{path}  {body_s}{note_s}", flush=True)

    def call_log(self) -> list[dict]:
        """Return all logged HTTP calls (useful for tests)."""
        return list(self._call_log)

    def clear_log(self) -> None:
        self._call_log.clear()
