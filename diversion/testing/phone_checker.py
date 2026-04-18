# phone_checker.py
# ============================================================
# PHONE NUMBER SPAM CHECKER
#
# FLOW:
#   1. Normalize number  -> +91XXXXXXXXXX
#   2. Check local CSV   -> instant, no internet
#   3. Check AbstractAPI -> carrier, risk, line type
#   4. Return SPAM / SUSPICIOUS / SAFE / UNKNOWN
#
# HOW TO RUN:
#   python phone_checker.py 9876543210
#   python phone_checker.py +919876543210
#   python phone_checker.py 9876543210 call.wav
#
# HOW TO ADD SPAM NUMBERS MANUALLY:
#   Open spam_numbers.csv and add:
#   +91XXXXXXXXXX,SPAM,Your reason,2026-03-01
# ============================================================

import re
import csv
import json
import sys
import requests
from pathlib import Path
from datetime import datetime


# ================================================================
# SETTINGS
# ================================================================

ABSTRACT_API_KEY = "c617510f88014170b3e9f9703e618727"
ABSTRACT_API_URL = "https://phoneintelligence.abstractapi.com/v1/"

LOCAL_DB_PATH    = Path("spam_numbers.csv")
CACHE_PATH       = Path("api_cache.json")
DEFAULT_COUNTRY  = "91"   # India


# ================================================================
# STEP 1 - NORMALIZE NUMBER
# ================================================================

def normalize(raw: str) -> str:
    """
    Converts any format to E.164 (+91XXXXXXXXXX)

    Examples:
        9876543210    -> +919876543210
        09876543210   -> +919876543210
        +919876543210 -> +919876543210  (unchanged)
        +14155552671  -> +14155552671   (non-Indian, unchanged)

    ANDROID: PhoneNumberUtils.formatNumberToE164(number, "IN")
    """
    cleaned = re.sub(r"[\s\-\(\)\.]", "", raw.strip())
    if cleaned.startswith("+"):
        return cleaned
    if cleaned.startswith("91") and len(cleaned) == 12:
        return "+" + cleaned
    if cleaned.startswith("0") and len(cleaned) == 11:
        return "+" + DEFAULT_COUNTRY + cleaned[1:]
    if len(cleaned) == 10 and cleaned[0] in "6789":
        return "+" + DEFAULT_COUNTRY + cleaned
    return "+" + cleaned


# ================================================================
# STEP 2 - LOCAL CSV DATABASE
# ================================================================

def load_local_db() -> dict:
    """
    Loads spam_numbers.csv into memory as a dict.
    Runs instantly with no internet.

    CSV FORMAT:
        number,label,reason,date_added
        +919000000001,SPAM,Fake bank KYC,2026-01-01

    ANDROID: SQLite table, query WHERE number = ?
    """
    if not LOCAL_DB_PATH.exists():
        _create_sample_db()
    db = {}
    with open(LOCAL_DB_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            num = row.get("number", "").strip()
            if num:
                db[num] = {
                    "label":  row.get("label",  "SPAM"),
                    "reason": row.get("reason", "In local spam database"),
                }
    return db


def _create_sample_db():
    rows = [
        ["+919000000001", "SPAM", "Fake bank KYC scam",     "2026-01-01"],
        ["+919000000002", "SPAM", "Fake police arrest scam", "2026-01-05"],
        ["+918800000001", "SPAM", "OTP phishing",            "2026-01-10"],
        ["+911800000001", "SPAM", "Fake customer care",      "2026-01-15"],
    ]
    with open(LOCAL_DB_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "label", "reason", "date_added"])
        writer.writerows(rows)
    print(f"  Created: {LOCAL_DB_PATH}  <- add your spam numbers here")


def add_to_local_db(e164: str, label: str, reason: str):
    """
    Adds a number to local database.
    Call this after audio model detects SCAM to remember the number.
    """
    if not LOCAL_DB_PATH.exists():
        _create_sample_db()
    today = datetime.now().strftime("%Y-%m-%d")
    with open(LOCAL_DB_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([e164, label, reason, today])
    print(f"  Saved: {e164} -> {label}")


def check_local_db(e164: str, db: dict) -> dict:
    if e164 in db:
        return {
            "found":  True,
            "label":  db[e164]["label"],
            "reason": db[e164]["reason"],
            "source": "local_database",
        }
    return {"found": False, "label": "UNKNOWN",
            "reason": "Not in local database", "source": "local_database"}


# ================================================================
# STEP 3 - ABSTRACTAPI LOOKUP
# ================================================================

def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def check_api(e164: str) -> dict:
    """
    Calls AbstractAPI and maps response to SPAM/SUSPICIOUS/SAFE.

    API RESPONSE FIELDS USED:
    +--------------------------------------------------+
    | phone_validation.is_valid   False = bad number   |
    | phone_validation.is_voip    True  = likely scam  |
    | phone_validation.line_status "inactive" = dead   |
    | phone_risk.risk_level       high/medium/low       |
    | phone_risk.is_abuse_detected True = reported      |
    | phone_risk.is_disposable    True = burner number  |
    | phone_carrier.line_type     mobile/voip/landline  |
    | phone_carrier.name          Airtel/Jio/etc        |
    | phone_location.country_name India/other           |
    | phone_breaches.total_breaches data breach count  |
    +--------------------------------------------------+

    DECISION PRIORITY:
        1. is_valid = False          -> INVALID
        2. is_abuse OR risk=high     -> SPAM
        3. is_voip OR type=voip      -> SUSPICIOUS
        4. is_disposable             -> SUSPICIOUS
        5. risk = medium             -> SUSPICIOUS
        6. line_status = inactive    -> SUSPICIOUS
        7. total_breaches > 0        -> SUSPICIOUS
        8. all clear                 -> SAFE

    ANDROID: OkHttp GET request, parse with JSONObject or Gson
    """
    cache = load_cache()
    if e164 in cache:
        print(f"  (cached from {cache[e164].get('cached_on', '?')})")
        return cache[e164]

    try:
        resp = requests.get(
            ABSTRACT_API_URL,
            params={"api_key": ABSTRACT_API_KEY, "phone": e164},
            timeout=8,
        )

        if resp.status_code != 200:
            return {"found": False, "label": "UNKNOWN",
                    "reason": f"API returned HTTP {resp.status_code}",
                    "source": "abstractapi"}

        d = resp.json()

        # Extract fields with safe defaults
        is_valid       = d.get("phone_validation", {}).get("is_valid",         True)
        is_voip        = d.get("phone_validation", {}).get("is_voip",          False)
        line_status    = d.get("phone_validation", {}).get("line_status",       "unknown")
        risk_level     = d.get("phone_risk",       {}).get("risk_level",        "unknown")
        is_abuse       = d.get("phone_risk",       {}).get("is_abuse_detected", False)
        is_disposable  = d.get("phone_risk",       {}).get("is_disposable",     False)
        line_type      = d.get("phone_carrier",    {}).get("line_type",         "unknown")
        carrier_name   = d.get("phone_carrier",    {}).get("name",              "Unknown")
        country        = d.get("phone_location",   {}).get("country_name",      "Unknown")
        total_breaches = d.get("phone_breaches",   {}).get("total_breaches")    or 0

        # Decision tree
        if not is_valid:
            label  = "INVALID"
            reason = "Not a valid phone number"

        elif is_abuse or risk_level == "high":
            label  = "SPAM"
            reason = (f"Abuse reported: {is_abuse} | "
                      f"Risk: {risk_level} | Carrier: {carrier_name}")

        elif is_voip or line_type == "voip":
            label  = "SUSPICIOUS"
            reason = f"VOIP number (commonly used by scammers) | Carrier: {carrier_name}"

        elif is_disposable:
            label  = "SUSPICIOUS"
            reason = f"Disposable/burner number | Carrier: {carrier_name}"

        elif risk_level == "medium":
            label  = "SUSPICIOUS"
            reason = f"Medium risk | Carrier: {carrier_name} | Country: {country}"

        elif line_status == "inactive":
            label  = "SUSPICIOUS"
            reason = f"Line is inactive | Carrier: {carrier_name}"

        elif int(total_breaches) > 0:
            label  = "SUSPICIOUS"
            reason = f"Found in {total_breaches} data breach(es) | Carrier: {carrier_name}"

        else:
            label  = "SAFE"
            reason = (f"Valid {line_type} | Risk: {risk_level} | "
                      f"Carrier: {carrier_name} | Country: {country}")

        result = {
            "found":      True,
            "label":      label,
            "reason":     reason,
            "source":     "abstractapi",
            "carrier":    carrier_name,
            "country":    country,
            "line_type":  line_type,
            "risk_level": risk_level,
            "cached_on":  datetime.now().strftime("%Y-%m-%d"),
        }
        cache[e164] = result
        save_cache(cache)
        return result

    except requests.exceptions.Timeout:
        return {"found": False, "label": "UNKNOWN",
                "reason": "API timed out", "source": "abstractapi"}
    except Exception as e:
        return {"found": False, "label": "UNKNOWN",
                "reason": f"API error: {str(e)}", "source": "abstractapi"}


# ================================================================
# MAIN CHECKER
# ================================================================

def check_number(raw: str) -> dict:
    """
    Full pipeline: normalize -> local DB -> API -> verdict.

    Returns dict with:
        number     : +91XXXXXXXXXX
        verdict    : SPAM / SUSPICIOUS / SAFE / UNKNOWN / INVALID
        reason     : explanation
        source     : local_database / abstractapi
        skip_model : True if SPAM (no need to run audio model)

    ANDROID: Call in CallScreeningService.onScreenCall()
    """
    print(f"\n{'='*52}")
    print(f"  Checking: {raw}")
    print(f"{'='*52}")

    e164 = normalize(raw)
    print(f"  Normalized: {e164}")

    # Step 2: Local DB (instant)
    print(f"\n  [1/2] Local database...")
    db    = load_local_db()
    local = check_local_db(e164, db)

    if local["found"]:
        print(f"  Found! {local['label']}: {local['reason']}")
        result = {
            "number":     e164,
            "verdict":    local["label"],
            "reason":     local["reason"],
            "source":     "local_database",
            "skip_model": local["label"] == "SPAM",
        }
        _print_verdict(result)
        return result

    print(f"  Not found in local database")

    # Step 3: AbstractAPI
    print(f"\n  [2/2] AbstractAPI lookup...")
    api = check_api(e164)
    print(f"  Label  : {api['label']}")
    print(f"  Reason : {api['reason']}")

    result = {
        "number":     e164,
        "verdict":    api["label"],
        "reason":     api["reason"],
        "source":     api["source"],
        "carrier":    api.get("carrier",    "Unknown"),
        "country":    api.get("country",    "Unknown"),
        "line_type":  api.get("line_type",  "unknown"),
        "risk_level": api.get("risk_level", "unknown"),
        "skip_model": api["label"] == "SPAM",
    }
    _print_verdict(result)
    return result


def _print_verdict(r: dict):
    icons = {"SPAM": "SCAM", "SUSPICIOUS": "WARNING",
             "SAFE": "SAFE", "UNKNOWN": "UNKNOWN", "INVALID": "INVALID"}
    emojis = {"SPAM": "X", "SUSPICIOUS": "!", "SAFE": "OK",
              "UNKNOWN": "?", "INVALID": "X"}

    v = r["verdict"]
    print(f"\n{'='*52}")
    print(f"  RESULT  : [{emojis.get(v,'?')}] {v}")
    print(f"  Number  : {r['number']}")
    print(f"  Reason  : {r['reason']}")
    print(f"  Source  : {r['source']}")
    if r.get("carrier"): print(f"  Carrier : {r['carrier']}")
    if r.get("country"): print(f"  Country : {r['country']}")
    print()
    if r["skip_model"]:
        print(f"  >> SKIP audio model - number already flagged as SPAM")
    else:
        print(f"  >> Proceed to audio model analysis")
    print(f"{'='*52}\n")


# ================================================================
# COMBINED: PHONE CHECK + AUDIO MODEL
# ================================================================

def full_pipeline(phone: str, audio_file: str = None) -> dict:
    """
    Step 1: Check phone number
       -> SPAM?  Show warning immediately, skip audio model
       -> SAFE?  Run audio model

    Step 2: Run audio model (if file provided)
       -> Combine both results

    ANDROID FLOW:
        onIncomingCall(number) {
            result = checkNumber(number)
            if (result.skipModel) { showAlert("SCAM"); return; }
            startAudioRecording()
        }
    """
    phone_result = check_number(phone)

    if phone_result["skip_model"]:
        return {
            "final_verdict": "SCAM",
            "confidence":    1.0,
            "reason":        f"Known spam number: {phone_result['reason']}",
            "phone_check":   phone_result,
            "audio_check":   None,
        }

    if not audio_file:
        return {
            "final_verdict": phone_result["verdict"],
            "confidence":    0.5,
            "reason":        phone_result["reason"],
            "phone_check":   phone_result,
            "audio_check":   None,
        }

    print(f"  Running audio model on: {audio_file}")
    try:
        from inference_for_android import analyse
        audio_result = analyse(audio_file)

        if (phone_result["verdict"] in ["SPAM", "SUSPICIOUS"] or
                audio_result["verdict"] == "SCAM"):
            final = "SCAM"
        else:
            final = "SAFE"

        return {
            "final_verdict": final,
            "confidence":    audio_result["confidence"],
            "reason":        (f"Phone: {phone_result['verdict']} | "
                              f"Audio: {audio_result['verdict']}"),
            "phone_check":   phone_result,
            "audio_check":   audio_result,
        }

    except ImportError:
        print("  inference_for_android.py not found - skipping audio")
        return {
            "final_verdict": phone_result["verdict"],
            "confidence":    0.5,
            "reason":        phone_result["reason"],
            "phone_check":   phone_result,
            "audio_check":   None,
        }


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python phone_checker.py <number>")
        print("  python phone_checker.py <number> <audio.wav>")
        print()
        print("Examples:")
        print("  python phone_checker.py 9876543210")
        print("  python phone_checker.py +919876543210")
        print("  python phone_checker.py 9876543210 call.wav")
        sys.exit(1)

    phone = sys.argv[1]
    audio = sys.argv[2] if len(sys.argv) >= 3 else None
    result = full_pipeline(phone, audio)
    sys.exit(1 if result["final_verdict"] in ["SCAM", "SPAM"] else 0)