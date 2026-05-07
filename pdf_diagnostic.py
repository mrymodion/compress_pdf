"""
PDF Diagnostic v2 — Khusus untuk content-stream dominant PDF (FPDF, LaTeX, dll.)
"""
import sys, os, zlib, struct

try:
    import pikepdf
    import fitz
except ImportError as e:
    print(f"Import error: {e}"); sys.exit(1)

def diagnose(path: str):
    if not os.path.exists(path):
        print(f"File tidak ditemukan: {path}"); return

    total_bytes = os.path.getsize(path)
    print(f"\n{'='*65}")
    print(f"  PDF DIAGNOSTIC REPORT v2 (Content-Stream Analysis)")
    print(f"  File  : {os.path.basename(path)}")
    print(f"  Size  : {total_bytes/1024:.1f} KB  ({total_bytes:,} bytes)")
    print(f"{'='*65}")

    pdf = pikepdf.Pdf.open(path)

    # ── Scan SEMUA objek dengan pendekatan berbeda ───────────────────────────
    categories = {
        "content_stream": 0,   # /Type /Page content streams
        "font_stream":    0,   # FontFile, FontFile2, FontFile3
        "image_stream":   0,   # /Subtype /Image
        "xmp_metadata":   0,   # /Type /Metadata
        "icc_profile":    0,   # ICCBased color
        "form_xobject":   0,   # /Subtype /Form
        "acroform":       0,   # AcroForm fields
        "other_stream":   0,   # semua stream lainnya
        "non_stream":     0,   # objek tanpa stream (dict, array, dll.)
    }

    stream_details = []
    content_streams = []
    raw_sizes       = []
    compressed_sizes= []

    for obj in pdf.objects:
        try:
            is_stream = hasattr(obj, "read_bytes") or hasattr(obj, "read_raw_bytes")

            if not is_stream:
                # Estimasi ukuran non-stream object via string repr
                try:
                    categories["non_stream"] += len(str(obj))
                except Exception:
                    pass
                continue

            # Baca raw bytes (compressed)
            try:
                raw = obj.read_raw_bytes()
                raw_len = len(raw)
            except Exception:
                raw_len = 0

            # Baca decoded bytes (uncompressed)
            try:
                decoded = obj.read_bytes()
                dec_len = len(decoded)
            except Exception:
                decoded = b""
                dec_len = 0

            raw_sizes.append(raw_len)
            compressed_sizes.append(dec_len)

            subtype  = str(obj.get("/Subtype", ""))
            obj_type = str(obj.get("/Type",    ""))
            flt      = str(obj.get("/Filter",  ""))

            # Klasifikasi
            if subtype == "/Image":
                categories["image_stream"] += raw_len
            elif "/FontFile" in str(list(obj.keys())):
                categories["font_stream"] += raw_len
            elif obj_type == "/Metadata":
                categories["xmp_metadata"] += raw_len
            elif subtype == "/Form":
                categories["form_xobject"] += raw_len
            elif obj_type == "/Page":
                categories["content_stream"] += raw_len
            else:
                categories["other_stream"] += raw_len

            if dec_len > 2000:
                stream_details.append({
                    "raw": raw_len, "dec": dec_len,
                    "type": obj_type, "sub": subtype, "filter": flt,
                    "decoded_preview": decoded[:120] if decoded else b""
                })

        except Exception:
            continue

    # ── Analisis page content streams via fitz ───────────────────────────────
    doc = fitz.open(path)
    page_stream_total_raw  = 0
    page_stream_total_dec  = 0
    page_drawing_cmds      = 0
    page_text_cmds         = 0
    page_samples           = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Get raw page content stream
        try:
            xref = page.xref  # xref number of page
            raw_content = doc.xref_stream_raw(xref)
            dec_content = doc.xref_stream(xref)
            if raw_content:
                page_stream_total_raw += len(raw_content)
            if dec_content:
                page_stream_total_dec += len(dec_content)
                # Hitung PDF commands
                page_drawing_cmds += dec_content.count(b" re ") + dec_content.count(b" l ") + dec_content.count(b" m ")
                page_text_cmds    += dec_content.count(b"Tj") + dec_content.count(b"TJ") + dec_content.count(b"Tf")
                if page_num < 3:
                    page_samples.append((page_num+1, dec_content[:300]))
        except Exception as e:
            # Fallback: get all content streams from page
            try:
                contents = page.read_contents()
                page_stream_total_dec += len(contents)
                page_drawing_cmds += contents.count(b" re ") + contents.count(b" l ")
                page_text_cmds    += contents.count(b"Tj") + contents.count(b"TJ")
                if page_num < 3:
                    page_samples.append((page_num+1, contents[:300]))
            except Exception:
                pass

    doc.close()

    # ── Ukuran xref table ────────────────────────────────────────────────────
    try:
        with open(path, "rb") as f:
            raw_pdf = f.read()
        xref_size = raw_pdf.count(b"\n") - raw_pdf.count(b"\r\n")  # rough estimate
        startxref_pos = raw_pdf.rfind(b"startxref")
        trailer_size  = len(raw_pdf) - startxref_pos if startxref_pos > 0 else 0
    except Exception:
        trailer_size = 0

    # ── Print Results ────────────────────────────────────────────────────────
    print(f"\n  [BREAKDOWN UKURAN — CONTENT STREAM ANALYSIS]")
    print(f"  Page content streams (raw/compressed) : {page_stream_total_raw/1024:>8.1f} KB")
    print(f"  Page content streams (decoded/actual)  : {page_stream_total_dec/1024:>8.1f} KB")
    print(f"  Kompres ratio content stream           : {(1 - page_stream_total_raw/max(page_stream_total_dec,1))*100:.1f}% sudah terkompresi")
    print()

    accounted = sum(categories.values())
    for name, b in sorted(categories.items(), key=lambda x: -x[1]):
        if b < 100: continue
        pct = b / total_bytes * 100
        bar = "=" * int(pct / 2)
        print(f"  {name:<22} {b/1024:>8.1f} KB  {pct:5.1f}%  [{bar}]")

    print(f"\n  [PAGE CONTENT COMMANDS]")
    print(f"  Drawing commands (re/l/m) : {page_drawing_cmds:,}")
    print(f"  Text commands (Tj/TJ/Tf)  : {page_text_cmds:,}")
    print(f"  Avg per page              : {(page_drawing_cmds+page_text_cmds)/max(len(pdf.pages),1):.0f} commands/page")

    # ── Content stream sample ────────────────────────────────────────────────
    if page_samples:
        print(f"\n  [SAMPLE CONTENT STREAM — Halaman 1, 120 bytes pertama]")
        for pnum, sample in page_samples[:1]:
            try:
                decoded_str = sample.decode("latin-1").replace("\n", " ").replace("\r", " ")
                print(f"  [{decoded_str[:200]}]")
            except Exception:
                print(f"  [binary data: {sample[:60]}]")

    # ── AcroForm analysis ────────────────────────────────────────────────────
    print(f"\n  [ACROFORM ANALYSIS]")
    try:
        if "/AcroForm" in pdf.Root:
            acroform = pdf.Root["/AcroForm"]
            fields   = acroform.get("/Fields", [])
            print(f"  Fields count: {len(fields)}")
            print(f"  AcroForm keys: {list(acroform.keys())[:10]}")
        else:
            print("  Tidak ada AcroForm")
    except Exception as e:
        print(f"  AcroForm error: {e}")

    # ── ilovepdf comparison insight ──────────────────────────────────────────
    ILOVEPDF_SIZE = 773 * 1024  # 773 KB
    gap = total_bytes - ILOVEPDF_SIZE
    print(f"\n  [GAP ANALYSIS vs ilovepdf]")
    print(f"  File kita    : {total_bytes/1024:.1f} KB")
    print(f"  ilovepdf     : {ILOVEPDF_SIZE/1024:.1f} KB")
    print(f"  Selisih      : {gap/1024:.1f} KB  ({gap/total_bytes*100:.1f}% dari ukuran kita)")
    print()

    # Estimasi teknik yang dipakai ilovepdf berdasarkan ratio
    ratio = total_bytes / ILOVEPDF_SIZE
    print(f"  Compression ratio ilovepdf: {ratio:.2f}x dari file kita")
    print()
    print(f"  HIPOTESIS TEKNIK ilovepdf (berdasarkan profil FPDF PDF):")
    print(f"  1. Re-encode content stream Flate level 9 (kita sudah coba tapi belum efektif)")
    print(f"  2. Linearisasi + cross-reference stream (xref table sebagai stream terkompresi)")
    print(f"  3. Remove AcroForm atau flatten form fields")
    print(f"  4. Strip /StructTreeRoot dan /MarkInfo")
    print(f"  5. Object stream consolidation (semua kecil-kecil jadi 1 stream terkompresi)")

    pdf.close()

    # ── Cek xref format ──────────────────────────────────────────────────────
    print(f"\n  [PDF STRUCTURE FORMAT]")
    try:
        with open(path, "rb") as f:
            header = f.read(20)
            f.seek(-1024, 2)
            tail = f.read()
        version = header[:8].decode("latin-1")
        has_xref_stream = b"xref" not in tail and b"/XRef" in raw_pdf
        has_linearized  = b"/Linearized" in raw_pdf[:1024]
        has_objstm      = b"/ObjStm" in raw_pdf
        print(f"  PDF Version   : {version}")
        print(f"  Xref stream   : {'Ya (modern)' if has_xref_stream else 'Tidak (table lama)'}")
        print(f"  Linearized    : {'Ya' if has_linearized else 'Tidak'}")
        print(f"  Object stream : {'Ya (ObjStm)' if has_objstm else 'Tidak (belum compressed)'}")
        print(f"  => ilovepdf kemungkinan besar mengaktifkan ObjStm + xref stream")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n{'='*65}\n")

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "test_A_baseline.pdf"
    diagnose(target)
    if len(sys.argv) > 2:
        diagnose(sys.argv[2])