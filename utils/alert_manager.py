"""
=============================================================================
AI VIGILANCE — REAL-TIME ALERT SYSTEM
File: utils/alert_manager.py

Sends a rich HTML email via SMTP whenever a registered person (or an unknown
intruder) is detected on any camera.

Email contains:
  • Detected person name  (or "Unknown Person")
  • Camera ID
  • Timestamp
  • Snapshot image embedded as Base64 (no external hosting needed)

Setup (add these to a .env file or your OS environment):
  ALERT_SMTP_HOST     = smtp.gmail.com          (or smtp.office365.com etc.)
  ALERT_SMTP_PORT     = 587                     (465 for SSL, 587 for STARTTLS)
  ALERT_SMTP_USER     = you@gmail.com
  ALERT_SMTP_PASS     = your_app_password       (Gmail: generate App Password)
  ALERT_FROM          = AI Vigilance <you@gmail.com>
  ALERT_TO            = you@gmail.com,boss@example.com   (comma-separated)
  ALERT_COOLDOWN      = 60                      (seconds between alerts for same person)
  ALERT_UNKNOWN       = true                    (send alerts for unknown persons too)
  ALERT_ENABLED       = true                    (master switch)

Gmail Quick-Start:
  1. Go to https://myaccount.google.com/apppasswords
  2. Generate an App Password for "Mail"
  3. Use that 16-char password as ALERT_SMTP_PASS
=============================================================================
"""

import os
import base64
import threading
import time
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
from typing import Optional
from pathlib import Path

# ─── Logger ───────────────────────────────────────────────────────────────────
log = logging.getLogger("AlertManager")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s"
)


# ─── HTML Email Template ───────────────────────────────────────────────────────

EMAIL_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Vigilance Alert</title>
</head>
<body style="margin:0;padding:0;background:#0a0a0f;font-family:'Segoe UI',Arial,sans-serif;">

  <!-- Outer wrapper -->
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0f;padding:30px 0;">
    <tr>
      <td align="center">

        <!-- Card -->
        <table width="600" cellpadding="0" cellspacing="0"
               style="max-width:600px;width:100%;background:#151520;border-radius:16px;
                      overflow:hidden;border:1px solid #2a2a3a;">

          <!-- Header bar -->
          <tr>
            <td style="background:#0a0a0f;padding:20px 32px;border-bottom:2px solid {accent};">
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td>
                    <span style="font-family:'Courier New',monospace;font-size:18px;
                                 font-weight:bold;color:{accent};letter-spacing:3px;">
                      AI VIGILANCE
                    </span>
                  </td>
                  <td align="right">
                    <span style="background:{alert_bg};color:{alert_color};
                                 font-size:11px;font-weight:bold;padding:4px 12px;
                                 border-radius:20px;letter-spacing:1px;">
                      {alert_label}
                    </span>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Title -->
          <tr>
            <td style="padding:28px 32px 8px;">
              <p style="margin:0;font-size:11px;letter-spacing:3px;
                        text-transform:uppercase;color:#888;">Detection Alert</p>
              <h1 style="margin:8px 0 0;font-size:26px;font-weight:700;
                          color:#ffffff;line-height:1.2;">
                {person_name}
              </h1>
              <p style="margin:6px 0 0;font-size:14px;color:{accent};">
                {alert_subtitle}
              </p>
            </td>
          </tr>

          <!-- Details table -->
          <tr>
            <td style="padding:20px 32px;">
              <table width="100%" cellpadding="0" cellspacing="0"
                     style="background:#1e1e30;border-radius:10px;overflow:hidden;">
                <tr>
                  <td style="padding:14px 20px;border-bottom:1px solid #2a2a3a;">
                    <span style="color:#666;font-size:11px;letter-spacing:2px;
                                 text-transform:uppercase;display:block;margin-bottom:3px;">
                      Camera
                    </span>
                    <span style="color:#fff;font-size:15px;font-weight:600;">
                      📹 {camera_id}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td style="padding:14px 20px;border-bottom:1px solid #2a2a3a;">
                    <span style="color:#666;font-size:11px;letter-spacing:2px;
                                 text-transform:uppercase;display:block;margin-bottom:3px;">
                      Detected At
                    </span>
                    <span style="color:#fff;font-size:15px;font-weight:600;">
                      🕒 {timestamp}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td style="padding:14px 20px;">
                    <span style="color:#666;font-size:11px;letter-spacing:2px;
                                 text-transform:uppercase;display:block;margin-bottom:3px;">
                      Confidence
                    </span>
                    <span style="color:#fff;font-size:15px;font-weight:600;">
                      {confidence_display}
                    </span>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Snapshot -->
          {snapshot_section}

          <!-- Footer note -->
          <tr>
            <td style="padding:8px 32px 28px;">
              <p style="margin:0;font-size:12px;color:#444;line-height:1.6;">
                This is an automated alert from your AI Vigilance surveillance system.<br>
                To adjust alert settings, update your <code style="color:#666;">.env</code> configuration.
              </p>
            </td>
          </tr>

          <!-- Bottom bar -->
          <tr>
            <td style="background:#0a0a0f;padding:14px 32px;border-top:1px solid #222;">
              <p style="margin:0;font-size:11px;color:#444;font-family:'Courier New',monospace;">
                AI VIGILANCE · {system_time} · Automated Security Alert
              </p>
            </td>
          </tr>

        </table>
        <!-- /Card -->

      </td>
    </tr>
  </table>

</body>
</html>
"""

SNAPSHOT_SECTION_TEMPLATE = """
<tr>
  <td style="padding:0 32px 20px;">
    <p style="margin:0 0 10px;font-size:11px;letter-spacing:2px;
              text-transform:uppercase;color:#666;">Snapshot</p>
    <img src="cid:snapshot_image"
         alt="Detection Snapshot"
         style="width:100%;max-width:536px;border-radius:10px;
                display:block;border:1px solid #2a2a3a;">
  </td>
</tr>
"""

NO_SNAPSHOT_SECTION = """
<tr>
  <td style="padding:0 32px 20px;">
    <div style="background:#1e1e30;border-radius:10px;padding:20px;text-align:center;
                border:1px dashed #333;">
      <p style="margin:0;color:#555;font-size:13px;">No snapshot available</p>
    </div>
  </td>
</tr>
"""


# ═══════════════════════════════════════════════════════════════════════════════
class AlertManager:
    """
    Thread-safe SMTP email alert dispatcher for AI Vigilance.

    Usage (in app.py or engine.py):
        from utils.alert_manager import AlertManager
        alert_manager = AlertManager()          # reads config from env

        # After a face is recognised or an unknown person is snapped:
        alert_manager.fire(
            person_name  = "John Doe",          # or "Unknown"
            camera_id    = "Front Door",
            snapshot_path= "snapshots/snap.jpg",# optional — embedded in email
            confidence   = 0.82,                # optional float 0-1
        )
    """

    def __init__(self):
        # ── SMTP config ────────────────────────────────────────────────────
        self.smtp_host   = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com")
        self.smtp_port   = int(os.getenv("ALERT_SMTP_PORT", "587"))
        self.smtp_user   = os.getenv("ALERT_SMTP_USER", "")
        self.smtp_pass   = os.getenv("ALERT_SMTP_PASS", "")
        self.from_addr   = os.getenv("ALERT_FROM", self.smtp_user)

        # ── Recipients (comma-separated in env) ───────────────────────────
        _to_raw          = os.getenv("ALERT_TO", "")
        self.to_addrs    = [a.strip() for a in _to_raw.split(",") if a.strip()]

        # ── Behaviour ─────────────────────────────────────────────────────
        self.cooldown    = int(os.getenv("ALERT_COOLDOWN", "60"))   # seconds
        self.send_unknown= os.getenv("ALERT_UNKNOWN", "true").lower() == "true"
        self.enabled     = os.getenv("ALERT_ENABLED", "true").lower() == "true"

        # ── Internal state ─────────────────────────────────────────────────
        self._cooldown_map: dict[str, float] = {}  # person_key → last_sent epoch
        self._lock = threading.Lock()
        self._send_queue: list = []
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self._validate_config()

    # ─── Public API ────────────────────────────────────────────────────────────

    def fire(
        self,
        person_name:   str,
        camera_id:     str,
        snapshot_path: Optional[str] = None,
        confidence:    float = 0.0,
    ) -> bool:
        """
        Queue an alert email. Returns True if queued, False if suppressed
        (disabled, missing config, or within cooldown window).
        """
        if not self.enabled:
            return False
        if not self.smtp_user or not self.smtp_pass:
            log.warning("Alert skipped — ALERT_SMTP_USER / ALERT_SMTP_PASS not set.")
            return False
        if not self.to_addrs:
            log.warning("Alert skipped — ALERT_TO not configured.")
            return False

        is_unknown = (person_name.lower() in ("unknown", ""))
        if is_unknown and not self.send_unknown:
            return False

        # Cooldown check — keyed by person+camera to allow same person
        # on different cameras to still alert independently.
        cooldown_key = f"{person_name}::{camera_id}"
        now = time.time()
        with self._lock:
            last = self._cooldown_map.get(cooldown_key, 0)
            if now - last < self.cooldown:
                remaining = int(self.cooldown - (now - last))
                log.debug(f"Alert suppressed (cooldown {remaining}s): {cooldown_key}")
                return False
            self._cooldown_map[cooldown_key] = now
            self._send_queue.append({
                "person_name":   person_name,
                "camera_id":     camera_id,
                "snapshot_path": snapshot_path,
                "confidence":    confidence,
                "timestamp":     datetime.now(),
            })

        log.info(f"Alert queued → {person_name} on {camera_id}")
        return True

    def test(self) -> bool:
        """Send a test email to verify SMTP configuration."""
        return self.fire(
            person_name="TEST ALERT",
            camera_id="System Check",
            snapshot_path=None,
            confidence=1.0,
        )

    # ─── Background Worker ─────────────────────────────────────────────────────

    def _worker_loop(self):
        """Drains the send queue in a dedicated thread so fire() never blocks."""
        while True:
            time.sleep(0.5)
            with self._lock:
                if not self._send_queue:
                    continue
                item = self._send_queue.pop(0)
            try:
                self._send_email(**item)
            except Exception as e:
                log.error(f"Email send failed: {e}")

    # ─── Email Builder ─────────────────────────────────────────────────────────

    def _send_email(
        self,
        person_name:   str,
        camera_id:     str,
        snapshot_path: Optional[str],
        confidence:    float,
        timestamp:     datetime,
    ):
        is_unknown = person_name.lower() in ("unknown", "")

        # ── Style variables by alert type ──────────────────────────────────
        if is_unknown:
            accent       = "#ff4d6d"
            alert_bg     = "#ffe5ea"
            alert_color  = "#c0001f"
            alert_label  = "⚠ INTRUDER ALERT"
            alert_subtitle = "An unrecognised person has been detected."
        else:
            accent       = "#00e5a0"
            alert_bg     = "#e8fff5"
            alert_color  = "#007a45"
            alert_label  = "✓ PERSON DETECTED"
            alert_subtitle = f"Recognised person detected on your system."

        # ── Confidence display ─────────────────────────────────────────────
        if confidence > 0:
            pct = int(confidence * 100)
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
            confidence_display = f"{bar} {pct}%"
        else:
            confidence_display = "N/A (unknown person)"

        # ── Timestamp ─────────────────────────────────────────────────────
        ts_str = timestamp.strftime("%A, %d %B %Y at %H:%M:%S")
        system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        # ── Snapshot handling ──────────────────────────────────────────────
        snapshot_bytes: Optional[bytes] = None
        if snapshot_path and Path(snapshot_path).is_file():
            try:
                with open(snapshot_path, "rb") as f:
                    snapshot_bytes = f.read()
            except Exception as e:
                log.warning(f"Could not read snapshot {snapshot_path}: {e}")

        snapshot_section = SNAPSHOT_SECTION_TEMPLATE if snapshot_bytes else NO_SNAPSHOT_SECTION

        # ── Build HTML body ────────────────────────────────────────────────
        html_body = EMAIL_HTML_TEMPLATE.format(
            accent            = accent,
            alert_bg          = alert_bg,
            alert_color       = alert_color,
            alert_label       = alert_label,
            alert_subtitle    = alert_subtitle,
            person_name       = person_name if not is_unknown else "⚠ Unknown Person",
            camera_id         = camera_id,
            timestamp         = ts_str,
            confidence_display= confidence_display,
            snapshot_section  = snapshot_section,
            system_time       = system_time,
        )

        # ── Plain-text fallback ────────────────────────────────────────────
        plain_body = (
            f"AI Vigilance Alert\n"
            f"{'=' * 40}\n"
            f"Person    : {person_name}\n"
            f"Camera    : {camera_id}\n"
            f"Detected  : {ts_str}\n"
            f"Confidence: {int(confidence * 100)}%\n"
            f"{'=' * 40}\n"
            f"This is an automated alert from AI Vigilance.\n"
        )

        # ── Build MIME message ─────────────────────────────────────────────
        subject = (
            f"[AI Vigilance] ⚠ Unknown Person — {camera_id}"
            if is_unknown else
            f"[AI Vigilance] {person_name} detected on {camera_id}"
        )

        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"]    = self.from_addr
        msg["To"]      = ", ".join(self.to_addrs)
        msg["X-Mailer"] = "AI-Vigilance-AlertManager/1.0"

        # Attach HTML + plain alternative
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain_body, "plain", "utf-8"))
        alt.attach(MIMEText(html_body,  "html",  "utf-8"))
        msg.attach(alt)

        # Embed snapshot image (referenced as cid:snapshot_image in HTML)
        if snapshot_bytes:
            img_mime = MIMEImage(snapshot_bytes)
            img_mime.add_header("Content-ID", "<snapshot_image>")
            img_mime.add_header(
                "Content-Disposition", "inline",
                filename=Path(snapshot_path).name
            )
            msg.attach(img_mime)

        # ── Send via SMTP ──────────────────────────────────────────────────
        try:
            if self.smtp_port == 465:
                # SSL from the start
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=15) as server:
                    server.login(self.smtp_user, self.smtp_pass)
                    server.sendmail(self.from_addr, self.to_addrs, msg.as_bytes())
            else:
                # STARTTLS (port 587 — default for Gmail, Outlook, Yahoo)
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(self.smtp_user, self.smtp_pass)
                    server.sendmail(self.from_addr, self.to_addrs, msg.as_bytes())

            log.info(f"✅ Alert email sent → {self.to_addrs} | Subject: {subject}")

        except smtplib.SMTPAuthenticationError:
            log.error(
                "SMTP authentication failed. "
                "For Gmail, use an App Password (not your account password). "
                "See: https://myaccount.google.com/apppasswords"
            )
        except smtplib.SMTPConnectError as e:
            log.error(f"Could not connect to SMTP server {self.smtp_host}:{self.smtp_port} — {e}")
        except smtplib.SMTPRecipientsRefused as e:
            log.error(f"Recipient(s) refused by SMTP server: {e}")
        except Exception as e:
            log.error(f"Unexpected email error: {e}")
            raise

    # ─── Config Validation ────────────────────────────────────────────────────

    def _validate_config(self):
        issues = []
        if not self.smtp_user:
            issues.append("ALERT_SMTP_USER is not set")
        if not self.smtp_pass:
            issues.append("ALERT_SMTP_PASS is not set")
        if not self.to_addrs:
            issues.append("ALERT_TO is not set")

        if issues:
            log.warning(
                "AlertManager configured with issues — emails will NOT send:\n  • "
                + "\n  • ".join(issues)
            )
        else:
            log.info(
                f"AlertManager ready │ SMTP: {self.smtp_host}:{self.smtp_port} │ "
                f"Recipients: {self.to_addrs} │ Cooldown: {self.cooldown}s"
            )

    # ─── Status ───────────────────────────────────────────────────────────────

    @property
    def status(self) -> dict:
        """Returns current configuration status (safe to expose via API)."""
        return {
            "enabled":       self.enabled,
            "smtp_host":     self.smtp_host,
            "smtp_port":     self.smtp_port,
            "smtp_user":     self.smtp_user,
            "to_addrs":      self.to_addrs,
            "cooldown_secs": self.cooldown,
            "send_unknown":  self.send_unknown,
            "queue_depth":   len(self._send_queue),
            "configured":    bool(self.smtp_user and self.smtp_pass and self.to_addrs),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION GUIDE — paste these snippets into app.py
# ═══════════════════════════════════════════════════════════════════════════════
#
# ── 1. Import & create a singleton (near top of app.py, after imports) ─────────
#
#   from utils.alert_manager import AlertManager
#   alert_manager = AlertManager()
#
#
# ── 2. Fire alert when a known person is recognised (in process_camera) ────────
#
#   # Inside the recognition block, after:
#   #   name, conf = recognizer.recognize(frame, [fx1, fy1, fx2, fy2])
#   if name != "Unknown" and conf > 0.35:
#       recognition_cache[tid] = (name, conf, frame_count)
#
#       # Save snapshot
#       ts  = int(time.time())
#       snap = f"snapshots/detected_{camera_id}_{tid}_{ts}.jpg"
#       cv2.imwrite(snap, frame)
#       db_manager.log_detection(None, camera_id, snap)
#
#       # Fire alert
#       alert_manager.fire(
#           person_name   = name,
#           camera_id     = camera_id,
#           snapshot_path = snap,
#           confidence    = conf,
#       )
#
#
# ── 3. Fire alert when an unknown person is snapped (already in app.py) ────────
#
#   # Replace the existing unknown-snap block with:
#   if tid not in unknown_snapped:
#       unknown_snapped.add(tid)
#       ts   = int(time.time())
#       snap = f"snapshots/unknown_{camera_id}_{tid}_{ts}.jpg"
#       cv2.imwrite(snap, frame)
#       db_manager.log_detection(None, camera_id, snap)
#
#       alert_manager.fire(
#           person_name   = "Unknown",
#           camera_id     = camera_id,
#           snapshot_path = snap,
#           confidence    = 0.0,
#       )
#
#
# ── 4. Add API endpoints to app.py ────────────────────────────────────────────
#
#   @app.get("/api/alerts/status")
#   async def alert_status():
#       return alert_manager.status
#
#   @app.post("/api/alerts/test")
#   async def alert_test():
#       queued = alert_manager.test()
#       return {"status": "queued" if queued else "failed",
#               "message": "Test email queued — check your inbox in ~10 seconds."}
#
#
# ── 5. .env file example ──────────────────────────────────────────────────────
#
#   ALERT_ENABLED=true
#   ALERT_SMTP_HOST=smtp.gmail.com
#   ALERT_SMTP_PORT=587
#   ALERT_SMTP_USER=youraddress@gmail.com
#   ALERT_SMTP_PASS=xxxx xxxx xxxx xxxx    ← Gmail App Password (16 chars)
#   ALERT_FROM=AI Vigilance <youraddress@gmail.com>
#   ALERT_TO=youraddress@gmail.com,securityteam@company.com
#   ALERT_COOLDOWN=60
#   ALERT_UNKNOWN=true
#
# ═══════════════════════════════════════════════════════════════════════════════


# ─── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to test your SMTP settings:
        python utils/alert_manager.py
    
    Set env vars first (or create a .env and load with python-dotenv).
    """
    import sys

    # Load .env if python-dotenv is installed
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Loaded .env file")
    except ImportError:
        print("  (python-dotenv not installed — reading system env vars only)")

    print("\n─── AI Vigilance Alert Manager — Self Test ───")
    manager = AlertManager()
    print(f"\nConfiguration:\n{manager.status}\n")

    if not manager.status["configured"]:
        print("❌ Cannot test — SMTP credentials not configured.")
        print("   Set ALERT_SMTP_USER, ALERT_SMTP_PASS, and ALERT_TO in your .env")
        sys.exit(1)

    # Test 1: Known person alert
    print("Sending test alert for a known person...")
    manager.fire(
        person_name   = "John Doe",
        camera_id     = "Front Door",
        snapshot_path = None,       # replace with a real .jpg path to test image embed
        confidence    = 0.87,
    )

    # Test 2: Unknown person alert (fires immediately — different cooldown key)
    print("Sending test alert for an unknown person...")
    manager.fire(
        person_name   = "Unknown",
        camera_id     = "Back Yard",
        snapshot_path = None,
        confidence    = 0.0,
    )

    # Give the worker thread time to send
    print("\nWaiting 10 seconds for emails to send...")
    time.sleep(10)
    print("Done. Check your inbox.")
