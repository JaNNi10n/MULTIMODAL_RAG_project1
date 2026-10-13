"""Generate a synthetic, privacy-safe document corpus with known question/answer pairs.

Every document is fictional (made-up companies, people and numbers) and carries a
"SAMPLE - SYNTHETIC DATA" banner. Each document is one PDF page with a real text
layer, plus a degraded JPEG "scan" of the same page for the OCR / VLM path.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageFilter

FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Ananya", "Vikram", "Isha", "Arjun", "Nisha",
         "Kunal", "Priya", "Siddharth", "Tara", "Manav", "Riya", "Dev", "Sana", "Yash", "Pooja"]
LAST = ["Malhotra", "Iyer", "Chawla", "Reddy", "Bose", "Kulkarni", "Sethi", "Nair", "Gill", "Menon",
        "Kapoor", "Joshi", "Rao", "Bajwa", "Dutta"]
VENDORS = ["Bluepeak Office Supplies", "Corvid Cloud Services", "Lotus Print Works", "Saffron Logistics",
           "Quartz Lab Instruments", "Neem Tree Catering", "Monsoon Electricals", "Indigo Furniture Co",
           "Peacock Travel Desk", "Granite IT Solutions"]
ITEMS = ["A4 paper ream", "Laser toner cartridge", "Cloud VM hours", "Brochure printing", "Courier pickup",
         "Pipette set", "Lunch buffet", "LED panel", "Ergonomic chair", "Laptop repair", "Air ticket fee",
         "Network switch", "Whiteboard", "Sample storage", "Annual support plan"]
MERCHANTS = ["FreshMart Grocers", "CityFuel Station", "Zenith Pharmacy", "Railway e-Ticket", "Bookworm Store",
             "Metro Card Recharge", "Cafe Aroma", "Sunrise Gym", "Hilltop Hardware", "SkyNet Broadband"]
UTILITIES = ["Riverside Power Distribution", "Greenline Electricity Board", "Valley Energy Ltd"]
BANNER = "SAMPLE - SYNTHETIC DATA - NOT A REAL DOCUMENT"


def inr(x: float) -> str:
    return f"INR {x:,.2f}"


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def make_invoice(rng: random.Random, n: int) -> tuple[list[str], list[dict]]:
    inv = f"INV-{rng.randint(1000, 9999)}-{n:02d}"
    vendor = rng.choice(VENDORS)
    customer = _person(rng)
    lines, subtotal = [], 0.0
    for item in rng.sample(ITEMS, rng.randint(3, 5)):
        qty, price = rng.randint(1, 12), rng.randint(150, 9000) + rng.choice([0, 0.5, 0.25])
        amt = qty * price
        subtotal += amt
        lines.append(f"{item:<24}{qty:>4}   {inr(price):>16}   {inr(amt):>16}")
    gst = round(subtotal * 0.18, 2)
    total = round(subtotal + gst, 2)
    text = [f"TAX INVOICE   {inv}", f"Issued by: {vendor}", f"Bill to: {customer}",
            f"Invoice date: 2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", "",
            f"{'Item':<24}{'Qty':>4}   {'Unit price':>16}   {'Amount':>16}", *lines, "",
            f"Subtotal: {inr(subtotal)}", f"GST (18%): {inr(gst)}", f"Total amount due: {inr(total)}"]
    qa = [
        {"question": f"What is the total amount due on invoice {inv}?", "answer": f"{total:,.2f}"},
        {"question": f"Which company issued invoice {inv}?", "answer": vendor},
        {"question": f"How much GST is charged on invoice {inv}?", "answer": f"{gst:,.2f}"},
    ]
    return text, qa


def make_statement(rng: random.Random, n: int) -> tuple[list[str], list[dict]]:
    acct = f"{rng.randint(1000, 9999)}"
    holder = _person(rng)
    bal = float(rng.randint(5000, 90000))
    opening = bal
    entries, paid = [], {}  # (description, signed amount)
    for merchant in rng.sample(MERCHANTS, rng.randint(4, 6)):
        if rng.random() < 0.2:
            entries.append(("Salary credit", float(rng.randint(2000, 30000))))
        paid[merchant] = rng.randint(80, 6000) + rng.choice([0.0, 0.5])
        entries.append((merchant, -paid[merchant]))
    rows = []
    for day, (desc, amt) in zip(sorted(rng.sample(range(1, 29), len(entries))), entries, strict=False):
        bal += amt
        debit, credit = (inr(-amt), "") if amt < 0 else ("", inr(amt))
        rows.append(f"{day:02d}-Jun  {desc:<22}{debit:>14}{credit:>16}{inr(bal):>18}")
    text = ["NORTHWIND DEMO BANK - ACCOUNT STATEMENT", f"Account holder: {holder}",
            f"Account number: XXXX-XXXX-{acct}", "Statement period: 01-Jun-2025 to 30-Jun-2025", "",
            f"Opening balance: {inr(opening)}", "", f"{'Date':<7}{'Description':<22}{'Debit':>14}{'Credit':>16}"
            f"{'Balance':>18}", *rows, "", f"Closing balance: {inr(bal)}"]
    merchant = rng.choice(list(paid))
    qa = [
        {"question": f"What is the closing balance for the statement of account ending {acct}?",
         "answer": f"{bal:,.2f}"},
        {"question": f"How much did account {acct} pay to {merchant}?", "answer": f"{paid[merchant]:,.2f}"},
        {"question": f"Who is the account holder of account ending {acct}?", "answer": holder},
    ]
    return text, qa


def make_bill(rng: random.Random, n: int) -> tuple[list[str], list[dict]]:
    consumer = f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
    utility = rng.choice(UTILITIES)
    units = rng.randint(90, 900)
    rate = rng.choice([5.5, 6.25, 7.0, 7.75])
    energy = units * rate
    fixed = float(rng.choice([120, 150, 200]))
    payable = round(energy + fixed, 2)
    text = [f"{utility.upper()} - ELECTRICITY BILL", f"Consumer number: {consumer}",
            f"Consumer name: {_person(rng)}", "Billing month: August 2025", "",
            f"Previous reading: {rng.randint(10000, 40000)}", f"Units consumed: {units} kWh",
            f"Energy charge ({units} x {rate:.2f}): {inr(energy)}", f"Fixed charge: {inr(fixed)}", "",
            f"Amount payable: {inr(payable)}", f"Due date: 2025-09-{rng.randint(10, 28)}"]
    qa = [
        {"question": f"How many units were consumed for consumer number {consumer}?", "answer": str(units)},
        {"question": f"What is the amount payable for consumer number {consumer}?", "answer": f"{payable:,.2f}"},
    ]
    return text, qa


MAKERS = {"invoice": make_invoice, "statement": make_statement, "bill": make_bill}


def _write_pdf(lines: list[str], path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 in points
    page.insert_text((40, 30), BANNER, fontsize=9, fontname="helv", color=(0.7, 0, 0))
    y = 70
    for i, line in enumerate(lines):
        page.insert_text((40, y), line, fontsize=13 if i == 0 else 10, fontname="cour")
        y += 20 if i == 0 else 15
    doc.save(path)
    doc.close()


def _scan(pdf_path: Path, out: Path, rng: random.Random, dpi: int = 130) -> None:
    """Render the PDF page and degrade it like a phone/flatbed scan (tilt, blur, noise, JPEG)."""
    with fitz.open(pdf_path) as doc:
        pix = doc[0].get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    img = img.rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor=255, resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    arr = np.asarray(img, dtype=np.float32)
    arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, 9, arr.shape)
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(out, quality=80)


def generate(out_dir: Path, n_docs: int = 45, seed: int = 7) -> Path:
    """Write PDFs, scans and qa.jsonl to out_dir. Returns the path of qa.jsonl."""
    rng = random.Random(seed)
    (out_dir / "pdf").mkdir(parents=True, exist_ok=True)
    (out_dir / "scans").mkdir(parents=True, exist_ok=True)
    kinds = list(MAKERS)
    qa_path = out_dir / "qa.jsonl"
    with qa_path.open("w", encoding="utf-8") as f:
        for n in range(n_docs):
            kind = kinds[n % len(kinds)]
            doc_id = f"{kind}_{n:03d}"
            lines, qa = MAKERS[kind](rng, n)
            pdf = out_dir / "pdf" / f"{doc_id}.pdf"
            _write_pdf(lines, pdf)
            _scan(pdf, out_dir / "scans" / f"{doc_id}.jpg", rng)
            for q in qa:
                f.write(json.dumps({"doc_id": doc_id, "page": 1, "kind": kind, **q}) + "\n")
    return qa_path
