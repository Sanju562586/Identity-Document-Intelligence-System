"""
stage1_data_factory/templates.py

PIL-based renderers for three identity document types:
  1. Driving Licence (Indian style)
  2. Aadhaar-style card
  3. Bank Statement (single-page)

Each renderer returns:
  - PIL.Image (RGB)
  - dict of ground-truth field values
"""
import io
import os
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

fake = Faker("en_IN")

# ──────────────────────────────────────────────────────────────────────────────
# Font helpers
# ──────────────────────────────────────────────────────────────────────────────

_FONT_CACHE: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

def _font(path: str, size: int) -> ImageFont.ImageFont:
    key = (path, size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, size)
        except (IOError, OSError):
            # Fallback to PIL default if font file not found
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _get_fonts(cfg: Dict) -> Dict[str, str]:
    fonts = cfg.get("stage1", {}).get("fonts", {})
    return {
        "regular": fonts.get("regular", "data/raw/fonts/DejaVuSans.ttf"),
        "bold":    fonts.get("bold",    "data/raw/fonts/DejaVuSans-Bold.ttf"),
        "mono":    fonts.get("mono",    "data/raw/fonts/DejaVuSansMono.ttf"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Colour palettes per card type
# ──────────────────────────────────────────────────────────────────────────────

DL_PALETTE = {
    "bg": (240, 248, 255),          # alice blue
    "header": (0, 47, 108),         # deep blue
    "header_text": (255, 255, 255),
    "label": (80, 80, 80),
    "value": (20, 20, 20),
    "border": (0, 47, 108),
    "photo_bg": (200, 210, 220),
    "accent": (255, 165, 0),        # orange stripe
}

AADHAAR_PALETTE = {
    "bg": (255, 255, 255),
    "header": (237, 28, 36),        # Indian govt red
    "header_text": (255, 255, 255),
    "label": (100, 100, 100),
    "value": (10, 10, 10),
    "border": (237, 28, 36),
    "photo_bg": (210, 210, 210),
    "uid_text": (0, 0, 180),
}

BANK_PALETTE = {
    "bg": (252, 252, 252),
    "header": (30, 70, 150),
    "header_text": (255, 255, 255),
    "row_even": (240, 245, 255),
    "row_odd": (255, 255, 255),
    "label": (60, 60, 60),
    "value": (10, 10, 10),
    "border": (180, 190, 210),
    "debit": (180, 0, 0),
    "credit": (0, 130, 0),
}


# ──────────────────────────────────────────────────────────────────────────────
# Helper: draw rounded rectangle
# ──────────────────────────────────────────────────────────────────────────────

def _draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    radius: int,
    fill: Tuple,
    outline: Optional[Tuple] = None,
    width: int = 2,
):
    x0, y0, x1, y1 = bbox
    draw.rectangle([x0 + radius, y0, x1 - radius, y1], fill=fill)
    draw.rectangle([x0, y0 + radius, x1, y1 - radius], fill=fill)
    draw.pieslice([x0, y0, x0 + 2 * radius, y0 + 2 * radius], 180, 270, fill=fill)
    draw.pieslice([x1 - 2 * radius, y0, x1, y0 + 2 * radius], 270, 360, fill=fill)
    draw.pieslice([x0, y1 - 2 * radius, x0 + 2 * radius, y1], 90, 180, fill=fill)
    draw.pieslice([x1 - 2 * radius, y1 - 2 * radius, x1, y1], 0, 90, fill=fill)
    if outline:
        draw.rounded_rectangle(bbox, radius=radius, outline=outline, width=width)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: fake QR-code placeholder
# ──────────────────────────────────────────────────────────────────────────────

def _draw_qr_placeholder(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int]):
    """Draw a simplified QR-code-like block pattern."""
    x0, y0, x1, y1 = bbox
    draw.rectangle(bbox, fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    cell = max(2, (x1 - x0) // 12)
    for i in range(12):
        for j in range(12):
            if random.random() > 0.5:
                cx, cy = x0 + i * cell, y0 + j * cell
                draw.rectangle([cx, cy, cx + cell - 1, cy + cell - 1], fill=(0, 0, 0))
    # Corner squares
    for cx, cy in [(x0, y0), (x1 - 4 * cell, y0), (x0, y1 - 4 * cell)]:
        draw.rectangle([cx, cy, cx + 4 * cell, cy + 4 * cell], outline=(0, 0, 0), width=2)
        draw.rectangle([cx + cell, cy + cell, cx + 3 * cell, cy + 3 * cell], fill=(0, 0, 0))


# ──────────────────────────────────────────────────────────────────────────────
# 1. Driving Licence
# ──────────────────────────────────────────────────────────────────────────────

def render_driving_licence(cfg: Dict) -> Tuple[Image.Image, Dict]:
    W, H = cfg.get("stage1", {}).get("image_size", [640, 400])
    fonts = _get_fonts(cfg)
    p = DL_PALETTE

    img = Image.new("RGB", (W, H), p["bg"])
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([2, 2, W - 3, H - 3], outline=p["border"], width=3)

    # Header bar
    draw.rectangle([0, 0, W, 52], fill=p["header"])
    draw.rectangle([0, 48, W, 56], fill=p["accent"])

    # Header text
    fh_big = _font(fonts["bold"], 18)
    fh_sm  = _font(fonts["regular"], 11)
    draw.text((W // 2, 14), "GOVERNMENT OF INDIA", fill=p["header_text"],
              font=fh_big, anchor="mm")
    draw.text((W // 2, 36), "DRIVING LICENCE", fill=p["header_text"],
              font=fh_sm, anchor="mm")

    # Photo placeholder
    ph_x, ph_y, ph_w, ph_h = 16, 68, 110, 140
    draw.rectangle([ph_x, ph_y, ph_x + ph_w, ph_y + ph_h], fill=p["photo_bg"])
    draw.text((ph_x + ph_w // 2, ph_y + ph_h // 2), "PHOTO", fill=(140, 140, 140),
              font=_font(fonts["regular"], 11), anchor="mm")

    # Fields
    states = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Uttar Pradesh",
              "West Bengal", "Gujarat", "Rajasthan", "Kerala", "Punjab"]
    state = random.choice(states)
    state_codes = {"Maharashtra": "MH", "Delhi": "DL", "Karnataka": "KA",
                   "Tamil Nadu": "TN", "Uttar Pradesh": "UP", "West Bengal": "WB",
                   "Gujarat": "GJ", "Rajasthan": "RJ", "Kerala": "KL", "Punjab": "PB"}
    code = state_codes[state]

    name    = fake.name()
    dob     = fake.date_of_birth(minimum_age=18, maximum_age=65).strftime("%d/%m/%Y")
    dl_num  = f"{code}-{fake.numerify(text='##')}-{fake.numerify(text='####')}-{fake.numerify(text='#######')}"
    address = fake.address().replace("\n", ", ")[:60]
    issue   = fake.date_between(start_date="-5y", end_date="today").strftime("%d/%m/%Y")
    expiry  = fake.date_between(start_date="today", end_date="+10y").strftime("%d/%m/%Y")
    vehicle_classes = random.sample(["LMV", "MCWG", "TRANS", "HMV"], k=random.randint(1, 3))
    blood_group = random.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])

    lbl_font = _font(fonts["regular"], 10)
    val_font = _font(fonts["bold"], 11)
    sm_font  = _font(fonts["regular"], 9)

    tx = ph_x + ph_w + 14
    rows = [
        ("Licence No.:", dl_num),
        ("Name:",        name),
        ("Date of Birth:", dob),
        ("Blood Group:", blood_group),
        ("Vehicle Class:", ", ".join(vehicle_classes)),
        ("Issue Date:", issue),
        ("Valid Till:", expiry),
        ("Address:", None),
    ]
    ty = 70
    for lbl, val in rows:
        draw.text((tx, ty), lbl, fill=p["label"], font=lbl_font)
        if val:
            draw.text((tx + 100, ty), val, fill=p["value"], font=val_font)
        ty += 18

    # Wrap address
    for line in textwrap.wrap(address, width=42):
        draw.text((tx, ty), line, fill=p["value"], font=sm_font)
        ty += 13

    # State watermark (light)
    wm_font = _font(fonts["bold"], 28)
    wm_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    wm_draw = ImageDraw.Draw(wm_img)
    wm_draw.text((W // 2, H - 60), state.upper(), fill=(0, 47, 108, 30),
                 font=wm_font, anchor="mm")
    img = Image.alpha_composite(img.convert("RGBA"), wm_img).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Footer
    draw.rectangle([0, H - 26, W, H], fill=p["header"])
    draw.text((16, H - 16), f"www.parivahan.gov.in  |  State: {state}",
              fill=p["header_text"], font=sm_font, anchor="lm")

    fields = {
        "name": name,
        "dob": dob,
        "id_number": dl_num,
        "address": address,
        "issue_date": issue,
        "expiry_date": expiry,
        "blood_group": blood_group,
        "vehicle_class": ", ".join(vehicle_classes),
        "state": state,
    }
    return img, fields


# ──────────────────────────────────────────────────────────────────────────────
# 2. Aadhaar-style Card
# ──────────────────────────────────────────────────────────────────────────────

def render_aadhaar(cfg: Dict) -> Tuple[Image.Image, Dict]:
    W, H = cfg.get("stage1", {}).get("image_size", [640, 400])
    fonts = _get_fonts(cfg)
    p = AADHAAR_PALETTE

    img = Image.new("RGB", (W, H), p["bg"])
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([2, 2, W - 3, H - 3], outline=p["border"], width=3)

    # Header
    draw.rectangle([0, 0, W, 56], fill=p["header"])
    fh_big = _font(fonts["bold"], 20)
    fh_sm  = _font(fonts["regular"], 11)
    draw.text((W // 2, 18), "भारत सरकार / Government of India", fill=p["header_text"],
              font=_font(fonts["bold"], 13), anchor="mm")
    draw.text((W // 2, 40), "आधार / AADHAAR", fill=p["header_text"],
              font=fh_big, anchor="mm")

    # Photo
    ph_x, ph_y, ph_w, ph_h = 16, 70, 100, 120
    draw.rectangle([ph_x, ph_y, ph_x + ph_w, ph_y + ph_h], fill=p["photo_bg"])
    draw.text((ph_x + ph_w // 2, ph_y + ph_h // 2), "PHOTO",
              fill=(140, 140, 140), font=_font(fonts["regular"], 10), anchor="mm")

    # Fields
    name   = fake.name()
    dob    = fake.date_of_birth(minimum_age=5, maximum_age=90).strftime("%d/%m/%Y")
    gender = random.choice(["Male", "Female", "Transgender"])
    uid    = " ".join([fake.numerify(text="####") for _ in range(3)])
    address = fake.address().replace("\n", ", ")[:70]

    lbl_font = _font(fonts["regular"], 10)
    val_font = _font(fonts["bold"], 12)

    tx = ph_x + ph_w + 16
    ty = 76
    for lbl, val in [("Name:", name), ("Date of Birth:", dob), ("Gender:", gender)]:
        draw.text((tx, ty), lbl, fill=p["label"], font=lbl_font)
        draw.text((tx, ty + 14), val, fill=p["value"], font=val_font)
        ty += 42

    draw.text((tx, ty), "Address:", fill=p["label"], font=lbl_font)
    ty += 14
    for line in textwrap.wrap(address, width=38):
        draw.text((tx, ty), line, fill=p["value"], font=_font(fonts["regular"], 10))
        ty += 13

    # UID number — big and prominent
    uid_font = _font(fonts["bold"], 22)
    draw.text((W // 2, H - 55), uid, fill=p["uid_text"], font=uid_font, anchor="mm")
    draw.text((W // 2, H - 30), "UID Number", fill=p["label"],
              font=_font(fonts["regular"], 9), anchor="mm")

    # QR code
    qr_size = 90
    qr_x = W - qr_size - 12
    qr_y = H - qr_size - 12
    _draw_qr_placeholder(draw, (qr_x, qr_y, qr_x + qr_size, qr_y + qr_size))

    # Decorative saffron-white-green stripe at bottom
    draw.rectangle([0, H - 10, W // 3, H], fill=(255, 153, 51))
    draw.rectangle([W // 3, H - 10, 2 * W // 3, H], fill=(255, 255, 255))
    draw.rectangle([2 * W // 3, H - 10, W, H], fill=(19, 136, 8))

    fields = {
        "name": name,
        "dob": dob,
        "gender": gender,
        "id_number": uid.replace(" ", ""),
        "address": address,
    }
    return img, fields


# ──────────────────────────────────────────────────────────────────────────────
# 3. Bank Statement
# ──────────────────────────────────────────────────────────────────────────────

def render_bank_statement(cfg: Dict) -> Tuple[Image.Image, Dict]:
    W, H = cfg.get("stage1", {}).get("image_size", [640, 400])
    fonts = _get_fonts(cfg)
    p = BANK_PALETTE

    # Taller for the table
    H_stmt = max(H, 520)
    img = Image.new("RGB", (W, H_stmt), p["bg"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, W - 3, H_stmt - 3], outline=p["border"], width=2)

    # Header
    draw.rectangle([0, 0, W, 60], fill=p["header"])
    banks = ["State Bank of India", "HDFC Bank Ltd.", "ICICI Bank", "Axis Bank",
             "Punjab National Bank", "Canara Bank"]
    bank_name = random.choice(banks)
    draw.text((20, 18), bank_name.upper(), fill=p["header_text"],
              font=_font(fonts["bold"], 16), anchor="lm")
    draw.text((20, 40), "Account Statement", fill=(200, 220, 255),
              font=_font(fonts["regular"], 11), anchor="lm")

    # Account meta
    name      = fake.name()
    acc_num   = fake.numerify(text="############")
    ifsc      = fake.bothify(text="????0######").upper()
    branch    = fake.city()
    stmt_from = fake.date_between(start_date="-6m", end_date="-3m").strftime("%d/%m/%Y")
    stmt_to   = fake.date_between(start_date="-2m", end_date="today").strftime("%d/%m/%Y")
    open_bal  = round(random.uniform(1000, 100000), 2)

    lbl_f = _font(fonts["regular"], 10)
    val_f = _font(fonts["bold"], 10)
    ty = 70
    meta = [
        ("Account Holder:", name),
        ("Account Number:", acc_num),
        ("IFSC Code:", ifsc),
        ("Branch:", branch),
        ("Statement Period:", f"{stmt_from} to {stmt_to}"),
        ("Opening Balance:", f"INR {open_bal:,.2f}"),
    ]
    for lbl, val in meta:
        draw.text((16, ty), lbl, fill=p["label"], font=lbl_f)
        draw.text((180, ty), val, fill=p["value"], font=val_f)
        ty += 16

    # Transaction table
    ty += 8
    cols = ["Date", "Description", "Debit", "Credit", "Balance"]
    col_x = [16, 100, 360, 440, 530]
    th_font = _font(fonts["bold"], 10)
    td_font = _font(fonts["regular"], 9)

    # Table header
    draw.rectangle([12, ty, W - 12, ty + 18], fill=p["header"])
    for cx, ch in zip(col_x, cols):
        draw.text((cx, ty + 9), ch, fill=(255, 255, 255), font=th_font, anchor="lm")
    ty += 18

    # Rows
    balance = open_bal
    n_rows = random.randint(6, 10)
    transactions = []
    for i in range(n_rows):
        date = fake.date_between(start_date="-3m", end_date="today").strftime("%d/%m/%Y")
        desc = random.choice([
            "NEFT Transfer", "ATM Withdrawal", "UPI Payment", "EMI Debit",
            "Salary Credit", "Interest Credit", "Online Purchase", "Refund Credit",
        ])
        is_credit = random.random() > 0.5
        amount = round(random.uniform(100, 20000), 2)
        if is_credit:
            balance += amount
            debit, credit = "-", f"{amount:,.2f}"
            amt_col, col = credit, p["credit"]
        else:
            balance = max(0, balance - amount)
            debit, credit = f"{amount:,.2f}", "-"
            amt_col, col = debit, p["debit"]

        row_bg = p["row_even"] if i % 2 == 0 else p["row_odd"]
        draw.rectangle([12, ty, W - 12, ty + 15], fill=row_bg)
        for cx, txt in zip(col_x, [date, desc[:32], debit, credit, f"{balance:,.2f}"]):
            clr = p["debit"] if cx == col_x[2] and debit != "-" else \
                  p["credit"] if cx == col_x[3] and credit != "-" else p["value"]
            draw.text((cx, ty + 7), txt, fill=clr, font=td_font, anchor="lm")
        ty += 16
        transactions.append({"date": date, "description": desc, "debit": debit,
                              "credit": credit, "balance": f"{balance:,.2f}"})

    # Closing balance
    draw.rectangle([12, ty, W - 12, ty + 18], fill=(230, 240, 255))
    draw.text((col_x[0], ty + 9), "Closing Balance:", fill=p["value"], font=th_font, anchor="lm")
    draw.text((col_x[4], ty + 9), f"INR {balance:,.2f}", fill=p["header"], font=th_font, anchor="lm")
    ty += 24

    # Footer
    draw.text((W // 2, H_stmt - 12), "This is a computer-generated statement — No signature required.",
              fill=(140, 140, 140), font=_font(fonts["regular"], 8), anchor="mm")

    # Resize back to target H if needed
    if H_stmt != H:
        img = img.resize((W, H), Image.LANCZOS)

    fields = {
        "name": name,
        "account_number": acc_num,
        "ifsc": ifsc,
        "branch": branch,
        "statement_from": stmt_from,
        "statement_to": stmt_to,
        "opening_balance": str(open_bal),
        "closing_balance": str(round(balance, 2)),
        "bank_name": bank_name,
    }
    return img, fields


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

RENDERERS = {
    "driving_licence":  render_driving_licence,
    "aadhaar":          render_aadhaar,
    "bank_statement":   render_bank_statement,
}


def render_card(card_type: str, cfg: Dict) -> Tuple[Image.Image, Dict]:
    """Render a single identity card of the specified type."""
    if card_type not in RENDERERS:
        raise ValueError(f"Unknown card type: {card_type}. Choose from {list(RENDERERS)}")
    return RENDERERS[card_type](cfg)
