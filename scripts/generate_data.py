"""Generate the synthetic support dataset shared by every app in the suite.

Everything here is made up: a fictional laptop brand ("Aurora"), random customers,
and tickets whose SLA outcomes follow realistic patterns (P1s are tight, email is
slower than phone, busy days breach more) so the dashboard and risk model have
something real to find. Seeded, so re-running gives the same files.

    python scripts/generate_data.py
"""
from __future__ import annotations

import csv
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
rng = random.Random(42)

END = datetime(2026, 10, 20, 18, 0)   # "now" for the dataset
DAYS = 120
MODELS = {"Aurora Book 14": "consumer", "Aurora Book Pro 16": "consumer", "Aurora Edge X1": "business", "Aurora Dock U3": "accessory"}
COUNTRIES = ["MY", "SG", "PH", "TH", "ID", "VN", "AU", "IN"]
FIRST = "Aisha Ben Chen Daniel Farah Grace Hiro Ivan Joy Kumar Lina Marco Nadia Omar Priya Ravi Siti Tan Umar Wei Yusuf Zara".split()
LAST = "Abdullah Lim Santos Nguyen Tan Wong Reyes Kumar Lee Ong Rahman Garcia Smith Ito Park Chua Hassan".split()

# category -> (weight, base hours to resolve, subject templates, body templates, typical priority)
CATEGORIES = {
    "Battery & charging": (18, 30, ["Battery not charging", "Laptop stuck at 80% charge", "Battery drains fast"],
                           ["My {model} shows plugged in but not charging since yesterday. Serial {serial}.",
                            "Battery is stuck at 80% and won't go higher. I use the original charger. Serial is {serial}.",
                            "Battery health says Replace after only {months} months. Is this covered? {serial}"], "P3"),
    "Battery safety": (2, 10, ["URGENT battery swelling", "Smell of burning from laptop"],
                       ["The bottom case of my {model} is lifting and the trackpad is pushed up. Serial {serial}. Is it safe?",
                        "There is a burning smell near the charging port when I plug in. {serial}"], "P1"),
    "No power": (9, 40, ["Laptop won't turn on", "Dead after update"],
                 ["My {model} will not power on at all. Tried another charger. Serial {serial}. I have a deadline!",
                  "After the BIOS update the screen stays black and nothing happens. {serial}"], "P2"),
    "Blue screen": (10, 28, ["Blue screen after update", "BSOD keeps restarting"],
                    ["Getting INACCESSIBLE_BOOT_DEVICE after the latest update. Serial {serial}.",
                     "Laptop crashes with a blue screen several times a day since Windows update. {serial}"], "P3"),
    "Wi-Fi & network": (12, 20, ["Wi-Fi keeps dropping", "Slow wireless"],
                        ["Wi-Fi disconnects every few minutes on my {model}. Other devices are fine. {serial}",
                         "Wireless option disappeared from Windows. Serial {serial}"], "P3"),
    "Display": (8, 45, ["Screen flickering", "Dead pixels on new laptop"],
                ["Screen flickers and shows lines, even in BIOS. Serial {serial}.",
                 "Brand new {model} has 4 dark pixels in the middle. Want a replacement. {serial}"], "P3"),
    "Physical damage": (7, 70, ["Cracked screen claim", "Spilled water on laptop"],
                        ["I dropped my {model} and the screen cracked. Do I have accidental damage cover? {serial}",
                         "Spilled coffee on the keyboard last night, now some keys don't work. Serial {serial}"], "P3"),
    "Dock & display out": (6, 22, ["Dock not detecting monitors", "Third monitor not working"],
                           ["Aurora Dock U3 is not detecting my two monitors after the update. Laptop serial {serial}.",
                            "Only two of my three monitors work through the dock. {serial}"], "P3"),
    "Returns & refunds": (9, 60, ["Return request", "Refund status"],
                          ["I want to return my {model} bought {days} days ago from your store. Serial {serial}.",
                           "Returned my laptop 3 weeks ago and still no refund. This is unacceptable. {serial}"], "P4"),
    "Warranty question": (12, 12, ["Warranty check", "Extend warranty"],
                          ["Can you tell me if serial {serial} is still under warranty?",
                           "My warranty is ending soon, how can I extend it? {serial}"], "P4"),
    "Fleet outage": (2, 16, ["Multiple Edge X1 laptops failing", "Fleet BSOD after update"],
                     ["About 30 of our Aurora Edge X1 laptops are blue-screening after last night's firmware push. One serial is {serial}.",
                      "Our office has 12 devices that won't boot. Need someone today. Example serial {serial}."], "P1"),
}
FIRST_RESPONSE_H = {"P1": 1, "P2": 4, "P3": 8, "P4": 24}
RESOLUTION_H = {"P1": 24, "P2": 48, "P3": 120, "P4": 168}
PRIORITY_EFFORT = {"P1": 0.8, "P2": 0.62, "P3": 1.5, "P4": 1.8}
CHANNEL_SPEED = {"phone": 0.7, "chat": 0.85, "email": 1.25}
REGION_STAFFING = {"MY": 0.9, "SG": 0.85, "PH": 1.0, "TH": 1.1, "ID": 1.2, "VN": 1.15, "AU": 0.95, "IN": 1.1}
AGENTS = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(18)]
AGENT_SKILL = {a: rng.uniform(0.75, 1.3) for a in AGENTS}


def serial() -> str:
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    s = rng.choice(letters) + rng.choice(letters) + "".join(rng.choice(chars) for _ in range(8))
    return s if any(c.isdigit() for c in s) else s[:-1] + str(ord(s[-1]) % 10)  # always at least one digit


def make_warranty(n: int = 420) -> list[dict]:
    rows = []
    for _ in range(n):
        model = rng.choices(list(MODELS), weights=[38, 26, 30, 6])[0]
        segment = MODELS[model]
        purchased = date(2026, 10, 20) - timedelta(days=rng.randint(10, 365 * 4))
        if segment == "business":
            plan, months = rng.choices([("Premier Support", 36), ("Premier Support", 60), ("Premium Care", 36), ("Base Warranty", 12)], weights=[35, 15, 30, 20])[0]
        else:
            plan, months = rng.choices([("Base Warranty", 12), ("Premium Care", 24), ("Premium Care", 36)], weights=[55, 30, 15])[0]
        end = purchased + timedelta(days=int(months * 30.44))
        rows.append({
            "serial": serial(), "model": model, "segment": segment, "plan": plan,
            "purchase_date": purchased.isoformat(), "warranty_end": end.isoformat(),
            "adp": rng.random() < (0.45 if segment == "business" else 0.25),
            "country": rng.choice(COUNTRIES),
            "owner": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "company": rng.choice(["Kestrel Logistics", "BlueHarbor Bank", "Nimbus Media", "Orchid Clinics", "Tidewater Schools"]) if segment == "business" else None,
        })
    return rows


def make_tickets(warranty: list[dict], n: int = 1400) -> list[dict]:
    cats, weights = list(CATEGORIES), [c[0] for c in CATEGORIES.values()]
    start = END - timedelta(days=DAYS)
    daily_load: dict[date, int] = {}
    created = sorted(start + timedelta(minutes=rng.randint(0, DAYS * 24 * 60)) for _ in range(n))
    # A launch-week spike and a bad-firmware week make the trend chart interesting.
    created += [datetime(2026, 8, 10, 9) + timedelta(minutes=rng.randint(0, 5 * 24 * 60)) for _ in range(120)]
    created.sort()
    for c in created:
        daily_load[c.date()] = daily_load.get(c.date(), 0) + 1
    avg_load = sum(daily_load.values()) / len(daily_load)

    tickets = []
    for i, created_at in enumerate(created):
        cat = rng.choices(cats, weights=weights)[0]
        if datetime(2026, 8, 10) <= created_at <= datetime(2026, 8, 15) and rng.random() < 0.5:
            cat = rng.choice(["Blue screen", "No power", "Fleet outage"])
        _, base_h, subjects, bodies, prio = CATEGORIES[cat]
        w = rng.choice(warranty)
        if cat == "Fleet outage":
            w = rng.choice([x for x in warranty if x["segment"] == "business"])
        tier = "enterprise" if w["plan"] == "Premier Support" else ("premium" if w["plan"] == "Premium Care" else "standard")
        if prio == "P3" and tier == "enterprise" and cat in ("No power", "Display") or (prio == "P3" and rng.random() < 0.08):
            prio = "P2"
        channel = rng.choices(["email", "chat", "phone"], weights=[45, 35, 20])[0]
        agent = rng.choice(AGENTS)
        load = daily_load[created_at.date()] / avg_load
        speed = CHANNEL_SPEED[channel] * REGION_STAFFING[w["country"]] * AGENT_SKILL[agent] * (0.6 + 0.5 * load)
        frt_target, res_target = FIRST_RESPONSE_H[prio], RESOLUTION_H[prio] / (2 if tier == "enterprise" else 1)
        first_resp_h = max(0.05, rng.lognormvariate(0, 0.6) * frt_target * 0.55 * speed)
        resolve_h = max(first_resp_h + 0.2, rng.lognormvariate(0, 0.55) * base_h * PRIORITY_EFFORT[prio] * speed * (0.7 if tier == "enterprise" else 1))
        reopened = rng.random() < (0.06 + 0.1 * (cat in ("Blue screen", "Wi-Fi & network")))
        if reopened:
            resolve_h *= 1.6
        resolved_at = created_at + timedelta(hours=resolve_h)
        status = "resolved" if resolved_at <= END else "open"
        breached = resolve_h > res_target
        sentiment = max(-1.0, min(1.0, rng.gauss(-0.1 - 0.4 * breached - 0.3 * (prio == "P1"), 0.35)))
        csat = None
        if status == "resolved" and rng.random() < 0.55:
            csat = max(1, min(5, round(rng.gauss(4.3 - 1.6 * breached - 0.6 * reopened, 0.8))))
        tmpl_vars = {"model": w["model"], "serial": w["serial"], "months": rng.randint(4, 30), "days": rng.randint(3, 45)}
        tickets.append({
            "ticket_id": f"CS-{260000 + i}", "created_at": created_at.isoformat(timespec="minutes"),
            "channel": channel, "country": w["country"], "customer_tier": tier, "product": w["model"],
            "serial": w["serial"], "category": cat, "priority": prio, "agent": agent,
            "subject": rng.choice(subjects), "body": rng.choice(bodies).format(**tmpl_vars),
            "first_response_at": (created_at + timedelta(hours=first_resp_h)).isoformat(timespec="minutes")
                                 if created_at + timedelta(hours=first_resp_h) <= END else "",
            "resolved_at": resolved_at.isoformat(timespec="minutes") if status == "resolved" else "",
            "status": status, "sla_first_response_h": frt_target, "sla_resolution_h": res_target,
            "reopened": int(reopened), "sentiment": round(sentiment, 2), "csat": csat if csat is not None else "",
            "queue_load": round(load, 2),
        })
    return tickets


def main() -> None:
    OUT.mkdir(exist_ok=True)
    warranty = make_warranty()
    (OUT / "warranty.json").write_text(json.dumps(warranty, indent=1))
    tickets = make_tickets(warranty)
    with open(OUT / "tickets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tickets[0]))
        w.writeheader()
        w.writerows(tickets)
    print(f"{len(warranty)} warranty records, {len(tickets)} tickets "
          f"({sum(t['status'] == 'open' for t in tickets)} open)")


if __name__ == "__main__":
    main()
