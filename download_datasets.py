"""Download and unpack KTH Actions and UR Fall (camera-0 RGB) into the folders ActiSense expects.

python download_datasets.py            # both datasets
python download_datasets.py --only kth
python download_datasets.py --only urfall

Result:
    videos/walking/*.avi  videos/jogging/  videos/running/  videos/boxing/  videos/handwaving/  videos/handclapping/
    urfall/falls/fall-01-cam0-rgb/*.png ...   urfall/adls/adl-01-cam0-rgb/*.png ...
Already-downloaded files are skipped, so you can re-run it after an interruption.
Licences: both datasets are for non-commercial academic use; cite them in your report.
"""
import argparse
import os
import ssl
import time
import urllib.error
import urllib.request
import zipfile


def make_ssl_context():
    """Verify HTTPS certificates with the operating system's trust store (what the browser uses).
    python.org builds of Python on Windows do not use it by default, which causes
    'CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate'. Verification stays ON."""
    try:
        import truststore                      # pip install truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), "Windows/OS certificate store"
    except ImportError:
        pass
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where()), "certifi bundle"
    except ImportError:
        return ssl.create_default_context(), "Python default"


SSL_CTX, SSL_SOURCE = make_ssl_context()

KTH_BASE = "https://www.csc.kth.se/cvap/actions/"
KTH_ACTIONS = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
UR_BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data/"

p = argparse.ArgumentParser()
p.add_argument("--only", choices=["kth", "urfall"])
p.add_argument("--downloads", default="downloads", help="where the zip files are kept")
p.add_argument("--kth-base", default=KTH_BASE)
p.add_argument("--ur-base", default=UR_BASE)
args = p.parse_args()
os.makedirs(args.downloads, exist_ok=True)


def fetch(url, dest, tries=3):
    """Download url to dest. Returns False if the file does not exist on the server."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    for attempt in range(1, tries + 1):
        try:
            print(f"  downloading {url}")
            tmp = dest + ".part"
            with urllib.request.urlopen(url, timeout=60, context=SSL_CTX) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            os.replace(tmp, dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            print(f"  HTTP {e.code}, retry {attempt}/{tries}")
        except Exception as e:  # network hiccup
            print(f"  {e}, retry {attempt}/{tries}")
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                break                      # retrying will not fix a certificate problem
        time.sleep(3 * attempt)
    raise SystemExit(
        f"Could not download {url}\n"
        f"(certificates checked with: {SSL_SOURCE})\n"
        f"Fix: run  pip install truststore  and try again, or download the file in your browser and save it as\n"
        f"  {os.path.abspath(dest)}\nthen run this script again (it will unzip it).")


def unzip(zip_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


print(f"Checking HTTPS certificates with: {SSL_SOURCE}")
if args.only in (None, "kth"):
    print("== KTH Actions (~1.1 GB) ==")
    for a in KTH_ACTIONS:
        z = os.path.join(args.downloads, f"{a}.zip")
        if not fetch(args.kth_base + f"{a}.zip", z):
            raise SystemExit(f"{a}.zip not found on the KTH server")
        out = os.path.join("videos", a)
        if not any(f.endswith(".avi") for f in os.listdir(out)) if os.path.isdir(out) else True:
            unzip(z, out)
        print(f"  videos/{a}: {len([f for f in os.listdir(out) if f.endswith('.avi')])} videos")

if args.only in (None, "urfall"):
    print("== UR Fall, camera 0 RGB only ==")
    for kind, folder, count in (("fall", "falls", 30), ("adl", "adls", 40)):
        got = 0
        for i in range(1, count + 1):
            name = f"{kind}-{i:02d}-cam0-rgb"
            z = os.path.join(args.downloads, name + ".zip")
            if not fetch(args.ur_base + name + ".zip", z):
                print(f"  {name}.zip not on server, skipped")
                continue
            out = os.path.join("urfall", folder, name)
            if not os.path.isdir(out):
                # the zip may or may not contain its own top folder; both layouts work later
                unzip(z, out)
            got += 1
        print(f"  urfall/{folder}: {got} sequences")

print("\nDone. Next: python extract_dataset.py --root videos ...  (see GUIDE.md)")