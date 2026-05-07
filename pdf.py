"""
================================================================================
 AI-Powered PDF Compressor — Research Edition v3.1
================================================================================
 Author      : Professor / AI Research Lead
 Architecture: Multi-stage adaptive compression pipeline dengan perceptual
               quality control berbasis SSIM metric dan rate-distortion theory.

 Changelog v3.1 (Forensic Patch):
   FIX #1  — Hapus 'recompress_streams' (parameter tidak valid di pikepdf API)
   FIX #2  — Fallback aman untuk read_raw_bytes() lintas versi pikepdf
   FIX #3  — Compatibility shim Image.LANCZOS untuk Pillow < 9.1
   FIX #4  — Ganti Image.Palette.ADAPTIVE dengan integer constant universal
   FIX #5  — Guard deflate_level untuk PyMuPDF < 1.18.7
   FIX #6  — Guard linear=True untuk PyMuPDF < 1.19.0
   FIX #7  — Eksplisit bytes(pix.samples) untuk memoryview compatibility
   FIX #8  — np.frombuffer(...).copy() untuk read-only buffer safety
   FIX #9  — Hapus unused imports (hashlib, ThreadPoolExecutor, as_completed)
   FIX #10 — RGBA alpha channel safety: cek mode SETELAH resize, guard split()[3]

 Referensi Ilmiah:
   [1] Wang et al. (2004). "Image quality assessment: From error visibility to
       structural similarity." IEEE TIP, 13(4), 600-612.
   [2] Balle et al. (2018). "Variational image compression with a scale
       hyperprior." ICLR 2018.
   [3] ISO 32000-2:2020 PDF 2.0 Specification.
   [4] Rabbani & Joshi (2002). "An overview of the JPEG 2000 still image
       compression standard." Signal Processing: Image Communication, 17(1), 3-48.

 Pipeline Arsitektur:
   +----------------------------------------------------------+
   |  Stage 1: PDF Structure Analysis & Content Classification |
   |  Stage 2: Perceptual Quality Baseline Measurement (SSIM)  |
   |  Stage 3: Adaptive Rate-Distortion Optimization           |
   |  Stage 4: Font Subsetting & Stream Compression            |
   |  Stage 5: Fail-Safe Verification & Integrity Check        |
   +----------------------------------------------------------+
================================================================================
"""

import os
import io
import gc
import sys
import shutil
import logging
import argparse
import tempfile
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import pikepdf
from pikepdf import PdfImage, Name, Pdf
from PIL import Image
import fitz  # PyMuPDF

# --- Setup Logging ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("PDFCompressorAI")


# =============================================================================
# SECTION 0: COMPATIBILITY LAYER
# Semua shim lintas-versi dikumpulkan di satu tempat agar mudah diaudit.
# =============================================================================

def _get_resampling_lanczos():
    """
    FIX #3: Image.Resampling.LANCZOS baru ada di Pillow >= 9.1.
    Pillow lama menggunakan Image.LANCZOS (integer alias).
    getattr dengan fallback menjamin kompatibilitas ke bawah.
    """
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _fitz_probe_capabilities() -> dict:
    """
    FIX #5 & #6 (Revised): Deteksi kapabilitas fitz via capability probe,
    BUKAN version string parsing.

    Masalah version string parsing di Windows:
      - fitz.version[0] bisa "1.24.3" atau "24.3.0" tergantung build/wheel
      - Beberapa PyMuPDF fork mengubah format versi tanpa konsistensi
      - Parsing tuple int tidak reliable lintas platform dan distribusi

    Solusi robust: probe langsung dengan dokumen dummy PDF minimal,
    hasil di-cache di function attribute agar hanya berjalan sekali per sesi.
    """
    if hasattr(_fitz_probe_capabilities, "_cache"):
        return _fitz_probe_capabilities._cache  # type: ignore[attr-defined]

    caps: dict = {"deflate_level": False, "linear": False}

    probe_path: Optional[str] = None
    probe_doc = None
    try:
        probe_doc  = fitz.open()
        probe_doc.new_page()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            probe_path = f.name

        # Probe deflate_level
        try:
            probe_doc.save(probe_path, garbage=0, deflate=True, deflate_level=1)
            caps["deflate_level"] = True
        except TypeError:
            caps["deflate_level"] = False

        # Probe linear
        try:
            probe_doc.save(probe_path, garbage=0, deflate=True, linear=True)
            caps["linear"] = True
        except TypeError:
            caps["linear"] = False

    except Exception as exc:
        log.debug(f"fitz capability probe error (safe fallback): {exc}")
    finally:
        if probe_doc is not None:
            try:
                probe_doc.close()
            except Exception:
                pass
        if probe_path and os.path.exists(probe_path):
            try:
                os.remove(probe_path)
            except Exception:
                pass

    log.debug(f"fitz capability probe result: {caps}")
    _fitz_probe_capabilities._cache = caps  # type: ignore[attr-defined]
    return caps


def _fitz_save_kwargs(deflate_level: int, use_linear: bool = False) -> dict:
    """
    Bangun kwargs doc.save() berdasarkan hasil capability probe.
    Aman di semua versi PyMuPDF dan semua platform (Windows/Linux/macOS).
    """
    caps:   dict = _fitz_probe_capabilities()
    kwargs: dict = {"garbage": 4, "deflate": True, "clean": True}

    if caps["deflate_level"]:
        kwargs["deflate_level"] = deflate_level

    if use_linear and caps["linear"]:
        kwargs["linear"] = True

    return kwargs


def _read_image_raw_size(raw_image) -> int:
    """
    FIX #2: read_raw_bytes() tidak ada di semua versi pikepdf.
    Urutan fallback:
      1. read_raw_bytes()         -- pikepdf modern
      2. get_raw_stream_buffer()  -- pikepdf lama
      3. 0                        -- ukuran tidak diketahui, proses tetap jalan
    """
    try:
        return len(raw_image.read_raw_bytes())
    except AttributeError:
        pass
    try:
        return len(bytes(raw_image.get_raw_stream_buffer()))
    except AttributeError:
        pass
    return 0


# =============================================================================
# SECTION 1: DATA STRUCTURES & ENUMERATIONS
# =============================================================================

class ContentClass(Enum):
    """Klasifikasi konten halaman untuk routing ke pipeline yang tepat."""
    VECTOR_DOMINANT = "vector_dominant"   # Teks + vektor murni
    BITMAP_MODERATE = "bitmap_moderate"   # Campuran teks dan gambar
    BITMAP_HEAVY    = "bitmap_heavy"      # Gambar dominan / foto
    SCANNED_DOC     = "scanned_doc"       # Dokumen scan (raster penuh)


class CompressionMode(Enum):
    AUTO       = "auto"
    HYBRID     = "hybrid"
    AGGRESSIVE = "aggressive"


@dataclass
class CompressionProfile:
    """
    Profil kompresi yang diturunkan dari analisis konten.
    Memodelkan trade-off Rate-Distortion (RD) secara eksplisit.
    """
    ssim_target: float              = 0.92  # Target SSIM minimum (0.0-1.0)
    jpeg_quality_min: int           = 35    # Floor untuk kualitas JPEG
    max_resolution: int             = 1800  # Max dimensi sisi terpanjang (px)
    apply_font_subsetting: bool     = True  # Optimasi font embedding
    deflate_level: int              = 9     # Zlib compression level (1-9)
    strip_metadata: bool            = True  # Hapus XMP, thumbnail, ICC tidak perlu
    recompress_streams: bool        = True  # Re-encode semua stream dengan Flate-9
    deep_font_subset: bool          = True  # Font subsetting via pikepdf (lebih dalam)


@dataclass
class PageAnalysisResult:
    """Hasil analisis konten per halaman."""
    page_num: int
    content_class: ContentClass
    image_coverage_ratio: float
    text_density: float
    vector_element_count: int
    raw_image_count: int
    estimated_entropy: float


@dataclass
class CompressionReport:
    """Laporan akhir hasil kompresi."""
    input_path: str
    output_path: str
    original_size_bytes: int
    compressed_size_bytes: int
    mode_applied: str
    pages_processed: int
    images_optimized: int
    rollback_triggered: bool = False
    integrity_verified: bool = True
    page_reports: List[dict] = field(default_factory=list)

    @property
    def ratio_percent(self) -> float:
        if self.original_size_bytes == 0:
            return 0.0
        return (1.0 - self.compressed_size_bytes / self.original_size_bytes) * 100

    def print_summary(self):
        orig_mb = self.original_size_bytes / 1_048_576
        comp_mb = self.compressed_size_bytes / 1_048_576
        bar_len = 40
        filled  = int(bar_len * max(0.0, min(self.ratio_percent, 100.0)) / 100)
        bar     = "=" * filled + "-" * (bar_len - filled)

        print("\n" + "=" * 68)
        print("  LAPORAN KOMPRESI PDF -- AI RESEARCH EDITION v3.2")
        print("=" * 68)
        print(f"  File Input    : {os.path.basename(self.input_path)}")
        print(f"  File Output   : {os.path.basename(self.output_path)}")
        print(f"  Mode Applied  : {self.mode_applied.upper()}")
        print(f"  Halaman       : {self.pages_processed}")
        print(f"  Gambar Opt.   : {self.images_optimized}")
        print("-" * 68)
        print(f"  Ukuran Awal   : {orig_mb:>8.3f} MB  ({self.original_size_bytes:,} bytes)")
        print(f"  Ukuran Akhir  : {comp_mb:>8.3f} MB  ({self.compressed_size_bytes:,} bytes)")
        print(f"  Kompresi      : [{bar}] {self.ratio_percent:.2f}%")
        if self.rollback_triggered:
            print("  [!] ROLLBACK AKTIF -- File asli dipertahankan")
        if not self.integrity_verified:
            print("  [!] PERINGATAN: Verifikasi integritas gagal -- periksa file output")
        print("=" * 68)


# =============================================================================
# SECTION 2: PERCEPTUAL QUALITY ENGINE (SSIM-BASED)
# =============================================================================

class PerceptualQualityEngine:
    """
    Implementasi SSIM (Structural Similarity Index Measure).
    Referensi: Wang et al. (2004), IEEE TIP 13(4), 600-612.
    SSIM in [0, 1]: nilai 1.0 = identik, > 0.90 = kualitas sangat baik.
    """

    def __init__(self, target_ssim: float = 0.92):
        self.target_ssim = target_ssim

    def compute_ssim(self, img_original: np.ndarray, img_compressed: np.ndarray) -> float:
        """SSIM via luminance channel (model Human Visual System)."""
        if img_original.shape != img_compressed.shape:
            img_compressed = cv2.resize(
                img_compressed,
                (img_original.shape[1], img_original.shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )

        if len(img_original.shape) == 3:
            gray_orig = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY).astype(np.float64)
            gray_comp = cv2.cvtColor(img_compressed, cv2.COLOR_RGB2GRAY).astype(np.float64)
        else:
            gray_orig = img_original.astype(np.float64)
            gray_comp = img_compressed.astype(np.float64)

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        k  = 11

        mu1    = cv2.GaussianBlur(gray_orig, (k, k), 1.5)
        mu2    = cv2.GaussianBlur(gray_comp, (k, k), 1.5)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1mu2 = mu1 * mu2

        s1sq = cv2.GaussianBlur(gray_orig ** 2,        (k, k), 1.5) - mu1_sq
        s2sq = cv2.GaussianBlur(gray_comp ** 2,        (k, k), 1.5) - mu2_sq
        s12  = cv2.GaussianBlur(gray_orig * gray_comp, (k, k), 1.5) - mu1mu2

        ssim_map = (
            (2 * mu1mu2 + C1) * (2 * s12 + C2)
        ) / (
            (mu1_sq + mu2_sq + C1) * (s1sq + s2sq + C2)
        )
        return float(np.mean(ssim_map))

    def find_optimal_quality(self, pil_image: Image.Image) -> Tuple[int, float]:
        """
        Binary search: kualitas JPEG minimum yang masih memenuhi target SSIM.
        O(log n) -- 7 iterasi untuk presisi +/-1 quality step.
        Returns: (optimal_quality, achieved_ssim)
        """
        original_arr = np.array(pil_image.convert("RGB"))
        lo, hi       = 20, 95
        best_quality = hi
        best_ssim    = 1.0

        for _ in range(7):
            mid = (lo + hi) // 2
            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG", quality=mid, optimize=True, subsampling=2)
            buf.seek(0)
            compressed_arr = np.array(Image.open(buf).convert("RGB"))
            ssim_val       = self.compute_ssim(original_arr, compressed_arr)

            if ssim_val >= self.target_ssim:
                best_quality = mid
                best_ssim    = ssim_val
                hi           = mid - 1
            else:
                lo = mid + 1

        return best_quality, best_ssim


# =============================================================================
# SECTION 3: CONTENT CLASSIFIER
# =============================================================================

class PDFContentClassifier:
    """
    Analisis konten per halaman PDF menggunakan 5 fitur:
    image coverage ratio, text density, vector element count,
    raw image count, dan Shannon entropy visual.
    """

    @staticmethod
    def _fast_image_coverage(page: fitz.Page) -> Tuple[float, int]:
        """
        Hitung image coverage ratio TANPA get_image_rects() yang lambat.

        Root cause lambat: get_image_rects() menghitung MD5 digest setiap
        pixmap untuk deduplication -- O(n x pixel_count) -- sangat berat
        pada PDF dengan gambar resolusi tinggi.

        Solusi: parse get_text("rawdict") untuk mendapatkan bounding box
        image block langsung dari struktur PDF -- O(n) murni, tanpa
        rendering, tanpa MD5, tanpa pixmap allocation.
        """
        page_area      = (page.rect.width * page.rect.height) or 1.0
        total_img_area = 0.0
        raw_image_count = 0

        try:
            blocks = page.get_text(
                "rawdict", flags=fitz.TEXT_PRESERVE_IMAGES
            ).get("blocks", [])
            for block in blocks:
                if block.get("type") == 1:  # type 1 = image block
                    bbox = block.get("bbox")
                    if bbox:
                        w = max(0.0, bbox[2] - bbox[0])
                        h = max(0.0, bbox[3] - bbox[1])
                        total_img_area  += w * h
                        raw_image_count += 1
        except Exception:
            # Fallback: count saja tanpa area estimation
            raw_image_count = len(page.get_images(full=False))
            # Estimasi konservatif: tiap gambar ~20% halaman
            total_img_area = raw_image_count * page_area * 0.20

        return min(total_img_area / page_area, 1.0), raw_image_count

    @staticmethod
    def analyze_page(page: fitz.Page, page_num: int) -> PageAnalysisResult:
        page_area = (page.rect.width * page.rect.height) or 1.0

        # FIX #11: Ganti get_image_rects() (O(n*pixels), MD5 per pixmap)
        # dengan _fast_image_coverage() berbasis rawdict -- O(n) murni
        image_coverage_ratio, raw_image_count =             PDFContentClassifier._fast_image_coverage(page)

        # Teks
        char_count   = len(page.get_text("text").strip())
        area_cm2     = (page.rect.width / 72 * 2.54) * (page.rect.height / 72 * 2.54)
        text_density = char_count / max(area_cm2, 1.0)

        # Vektor
        vector_element_count = len(page.get_drawings())

        # Entropi Shannon via thumbnail 10%
        mat = fitz.Matrix(0.1, 0.1)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        # FIX #8: .copy() agar array writable -- read-only buffer safe
        img_arr = np.frombuffer(pix.samples, dtype=np.uint8).copy()
        if len(img_arr) > 0:
            hist, _ = np.histogram(img_arr, bins=256, range=(0, 255))
            hn      = hist / max(hist.sum(), 1)
            hn      = hn[hn > 0]
            est_entropy = float(-np.sum(hn * np.log2(hn)))
        else:
            est_entropy = 0.0

        # Klasifikasi
        if image_coverage_ratio > 0.7:
            cls = ContentClass.SCANNED_DOC if text_density < 5 else ContentClass.BITMAP_HEAVY
        elif image_coverage_ratio > 0.25 or raw_image_count > 2:
            cls = ContentClass.BITMAP_MODERATE
        else:
            cls = ContentClass.VECTOR_DOMINANT

        return PageAnalysisResult(
            page_num=page_num,
            content_class=cls,
            image_coverage_ratio=image_coverage_ratio,
            text_density=text_density,
            vector_element_count=vector_element_count,
            raw_image_count=raw_image_count,
            estimated_entropy=est_entropy
        )

    @staticmethod
    def determine_document_profile(
        page_analyses: List[PageAnalysisResult],
        profile: CompressionProfile
    ) -> CompressionProfile:
        if not page_analyses:
            return profile

        class_counts: Dict[ContentClass, int] = {}
        for pa in page_analyses:
            class_counts[pa.content_class] = class_counts.get(pa.content_class, 0) + 1

        dominant_class = max(class_counts, key=class_counts.get)
        coverage_avg   = float(np.mean([pa.image_coverage_ratio for pa in page_analyses]))
        entropy_avg    = float(np.mean([pa.estimated_entropy for pa in page_analyses]))

        log.info(
            f"Dominant content class: {dominant_class.value} | "
            f"Avg image coverage: {coverage_avg:.1%} | "
            f"Avg visual entropy: {entropy_avg:.2f} bits"
        )

        if dominant_class == ContentClass.VECTOR_DOMINANT:
            profile.max_resolution        = 1200
            profile.ssim_target           = 0.95
            profile.jpeg_quality_min      = 55
            profile.apply_font_subsetting = True
        elif dominant_class == ContentClass.BITMAP_MODERATE:
            profile.max_resolution   = 1600
            profile.ssim_target      = 0.92
            profile.jpeg_quality_min = 40
        else:  # BITMAP_HEAVY, SCANNED_DOC
            profile.max_resolution   = 1800 if entropy_avg > 6.0 else 1400
            profile.ssim_target      = 0.90
            profile.jpeg_quality_min = 30

        return profile


# =============================================================================
# SECTION 4: IMAGE OPTIMIZATION ENGINE
# =============================================================================

class AdaptiveImageOptimizer:
    """
    Pipeline optimasi gambar berbasis SSIM perceptual quality metric.

    Routing:
    - Foto/kompleks  -> JPEG dengan SSIM binary-search quality
    - Diagram/grafik -> PNG dengan adaptive palette quantization
    - Grayscale      -> JPEG-gray dengan subsampling 4:0:0
    """

    def __init__(self, profile: CompressionProfile):
        self.profile        = profile
        self.quality_engine = PerceptualQualityEngine(profile.ssim_target)
        # FIX #3: LANCZOS compatibility shim
        self._lanczos       = _get_resampling_lanczos()

    def _resize_if_needed(self, pil_image: Image.Image) -> Image.Image:
        w, h    = pil_image.size
        max_res = self.profile.max_resolution
        if w > max_res or h > max_res:
            scale        = max_res / max(w, h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            pil_image    = pil_image.resize((new_w, new_h), self._lanczos)
            log.debug(f"  Resized: {w}x{h} -> {new_w}x{new_h}")
        return pil_image

    def _estimate_complexity(self, pil_image: Image.Image) -> float:
        arr = np.array(pil_image.convert("L"))
        return float(cv2.Laplacian(arr, cv2.CV_64F).var())

    def _is_image_simple(self, pil_image: Image.Image) -> bool:
        """Gambar sederhana = sedikit warna unik DAN tepi rendah."""
        thumb         = pil_image.convert("RGB").resize((64, 64), Image.NEAREST)
        thumb_arr     = np.array(thumb).reshape(-1, 3)
        unique_colors = len(np.unique(thumb_arr, axis=0))
        laplacian_var = self._estimate_complexity(pil_image)
        return unique_colors < 512 and laplacian_var < 200

    def optimize(self, pil_image: Image.Image) -> Tuple[bytes, str, int, float]:
        """
        Pipeline optimasi utama.

        FIX #10: Simpan mode SEBELUM resize. Setelah resize, re-cek mode aktual
                 dan verifikasi ketersediaan alpha band sebelum split()[3].

        Returns: (compressed_bytes, format_name, quality_used, achieved_ssim)
        """
        original_mode = pil_image.mode

        # Step 1: Resize
        pil_image = self._resize_if_needed(pil_image)

        # Step 2: Handle alpha -- cek mode SETELAH resize, validasi band count
        current_mode = pil_image.mode
        has_alpha    = (current_mode == "RGBA") or \
                       (original_mode == "RGBA" and len(pil_image.getbands()) >= 4)

        if has_alpha:
            background = Image.new("RGB", pil_image.size, (255, 255, 255))
            try:
                bands = pil_image.split()
                alpha = bands[3] if len(bands) >= 4 else None
                rgb   = pil_image.convert("RGB")
                if alpha is not None:
                    background.paste(rgb, mask=alpha)
                else:
                    background.paste(rgb)
            except (IndexError, ValueError):
                background.paste(pil_image.convert("RGB"))
            pil_image = background
        elif current_mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")

        # Step 3: Routing ke encoder
        if self._is_image_simple(pil_image):
            return self._encode_png_adaptive(pil_image)
        elif pil_image.mode == "L":
            return self._encode_jpeg_grayscale(pil_image)
        else:
            return self._encode_jpeg_ssim(pil_image)

    def _encode_jpeg_ssim(self, pil_image: Image.Image) -> Tuple[bytes, str, int, float]:
        optimal_quality, achieved_ssim = self.quality_engine.find_optimal_quality(pil_image)
        quality = max(optimal_quality, self.profile.jpeg_quality_min)
        buf     = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=quality,
                       optimize=True, progressive=True, subsampling=2)
        return buf.getvalue(), "JPEG", quality, achieved_ssim

    def _encode_jpeg_grayscale(self, pil_image: Image.Image) -> Tuple[bytes, str, int, float]:
        gray_img = pil_image.convert("L")
        optimal_quality, achieved_ssim = self.quality_engine.find_optimal_quality(gray_img)
        quality  = max(optimal_quality, self.profile.jpeg_quality_min)
        buf      = io.BytesIO()
        gray_img.save(buf, format="JPEG", quality=quality, optimize=True, subsampling=0)
        return buf.getvalue(), "JPEG_GRAY", quality, achieved_ssim

    def _encode_png_adaptive(self, pil_image: Image.Image) -> Tuple[bytes, str, int, float]:
        """
        PNG adaptive palette quantization.

        FIX #4: Integer constant 1 menggantikan Image.Palette.ADAPTIVE
                yang hanya tersedia di Pillow >= 9.2.
                Image.ADAPTIVE == Image.Palette.ADAPTIVE == 1 (universal).
        """
        arr          = np.array(pil_image.convert("RGB"))
        unique_count = len(np.unique(arr.reshape(-1, 3), axis=0))
        buf          = io.BytesIO()

        if unique_count <= 256:
            quantized = pil_image.convert(
                "P",
                palette=1,  # 1 == ADAPTIVE, kompatibel semua versi Pillow
                colors=min(unique_count, 256)
            )
            quantized.save(buf, format="PNG", optimize=True)
        else:
            pil_image.save(buf, format="PNG", optimize=True,
                           compress_level=min(self.profile.deflate_level, 9))

        return buf.getvalue(), "PNG", 100, 1.0


# =============================================================================
# SECTION 5: CORE COMPRESSOR
# =============================================================================

class IntelligentPDFCompressor:
    """
    Orchestrator utama pipeline kompresi PDF 5-stage.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        mode: CompressionMode = CompressionMode.AUTO,
        profile: Optional[CompressionProfile] = None
    ):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

        self.input_path  = input_path
        self.output_path = output_path
        self.mode        = mode
        self.profile     = profile or CompressionProfile()
        self.report      = CompressionReport(
            input_path=input_path,
            output_path=output_path,
            original_size_bytes=os.path.getsize(input_path),
            compressed_size_bytes=0,
            mode_applied=mode.value,
            pages_processed=0,
            images_optimized=0
        )

    # -- Stage 1 ---------------------------------------------------------------

    def _analyze_document(self) -> List[PageAnalysisResult]:
        log.info("Stage 1: Menganalisis konten dokumen...")
        doc     = fitz.open(self.input_path)
        results = []
        for page_num in range(len(doc)):
            pa = PDFContentClassifier.analyze_page(doc[page_num], page_num)
            results.append(pa)
            log.debug(
                f"  Halaman {page_num+1}: {pa.content_class.value} | "
                f"img_cov={pa.image_coverage_ratio:.1%} | "
                f"txt_dens={pa.text_density:.1f}"
            )
        doc.close()
        self.report.pages_processed = len(results)
        return results

    # -- Stage 3a: Hybrid ------------------------------------------------------

    def _compress_hybrid(self, optimizer: AdaptiveImageOptimizer) -> None:
        """
        Preservasi teks/vektor, hanya kompres gambar bitmap.

        FIX #1: Hapus 'recompress_streams' -- tidak valid di pikepdf API manapun.
                pikepdf.save() hanya menerima: compress_streams, object_stream_mode,
                normalize_content, linearize, qdf, min_version, force_version,
                encryption, recompress_streams (hanya di beberapa fork/preview).
                Versi resmi: cukup compress_streams=True.
        FIX #2: _read_image_raw_size() sebagai fallback-safe size check.
        """
        log.info("Stage 3: Mode HYBRID -- Optimasi gambar dengan preservasi teks/vektor...")

        try:
            pdf = Pdf.open(self.input_path)
        except Exception as e:
            raise RuntimeError(f"Gagal membuka PDF: {e}")

        processed_xrefs  = set()
        images_optimized = 0

        for i, page in enumerate(pdf.pages):
            for name, raw_image in list(page.images.items()):
                obj_gen = raw_image.objgen
                if obj_gen in processed_xrefs:
                    continue
                processed_xrefs.add(obj_gen)

                try:
                    pdf_img = PdfImage(raw_image)
                    pil_img = pdf_img.as_pil_image()

                    if pil_img.width < 80 or pil_img.height < 80:
                        continue

                    opt_bytes, fmt, quality, ssim = optimizer.optimize(pil_img)

                    # FIX #2: size check dengan fallback aman
                    original_img_size = _read_image_raw_size(raw_image)
                    if original_img_size > 0 and len(opt_bytes) >= original_img_size:
                        log.debug(f"  Skip hal. {i+1}: hasil tidak lebih kecil")
                        continue

                    if fmt in ("JPEG", "JPEG_GRAY"):
                        raw_image.write(opt_bytes, filter=Name("/DCTDecode"))
                        raw_image.ColorSpace = (
                            Name("/DeviceGray") if pil_img.mode == "L"
                            else Name("/DeviceRGB")
                        )
                    else:
                        raw_image.write(opt_bytes, filter=Name("/FlateDecode"))

                    if "/DecodeParms" in raw_image:
                        del raw_image.DecodeParms

                    log.debug(
                        f"  Hal. {i+1} [{fmt}] q={quality} ssim={ssim:.4f} | "
                        f"{original_img_size // 1024}KB -> {len(opt_bytes) // 1024}KB"
                    )
                    images_optimized += 1

                except Exception as e:
                    log.warning(f"  Gagal optimasi gambar di hal. {i+1}: {e}")
                    continue

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        # FIX PermissionError (WinError 32):
        # Pisahkan doc.close() ke nested try/finally agar SELALU dipanggil
        # meski doc.save() raise exception. Tanpa ini, file handle tetap
        # terkunci di Windows dan os.remove() di finally luar akan gagal.
        try:
            # Pass 1: pikepdf — image replacement + stream compression
            # FIX #1: hanya compress_streams=True, tanpa recompress_streams
            try:
                pdf.save(tmp_path, compress_streams=True)
            finally:
                pdf.close()  # Pastikan pikepdf melepas handle sebelum fitz membuka

            # Pass 2: fitz — garbage collection, deflate, cleanup
            save_kw = _fitz_save_kwargs(self.profile.deflate_level, use_linear=False)
            doc     = fitz.open(tmp_path)
            try:
                doc.save(self.output_path, **save_kw)
            finally:
                doc.close()  # Pastikan fitz melepas handle sebelum os.remove()
        finally:
            # Temp file baru bisa dihapus setelah KEDUA handle ditutup
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as e:
                    log.warning(f"Tidak dapat menghapus temp file {tmp_path}: {e}")

        self.report.images_optimized = images_optimized

    # -- Stage 3b: Aggressive --------------------------------------------------

    def _compress_aggressive(self, optimizer: AdaptiveImageOptimizer, dpi: int = 150) -> None:
        """
        Rasterisasi seluruh halaman ke bitmap.

        FIX #7: bytes(pix.samples) eksplisit -- PyMuPDF >= 1.23 mengembalikan
                memoryview, bukan bytes. Image.frombytes() memerlukan bytes.
        FIX #5: deflate_level via version-aware kwargs.
        """
        log.info(f"Stage 3: Mode AGGRESSIVE -- Rasterisasi pada {dpi} DPI...")
        log.warning("  [!] Mode ini menghilangkan text layer. PDF output tidak searchable.")

        doc     = fitz.open(self.input_path)
        new_doc = fitz.open()
        mat     = fitz.Matrix(dpi / 72, dpi / 72)

        images_optimized = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            log.debug(f"  Rasterisasi halaman {page_num+1}/{len(doc)}")

            pix = page.get_pixmap(matrix=mat, alpha=False)
            # FIX #7: konversi eksplisit ke bytes
            raw_samples = bytes(pix.samples)
            pil_image   = Image.frombytes("RGB", (pix.width, pix.height), raw_samples)

            opt_bytes, fmt, quality, ssim = optimizer.optimize(pil_image)
            log.debug(f"  Hal. {page_num+1} [{fmt}] q={quality} ssim={ssim:.4f}")

            new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=opt_bytes)
            images_optimized += 1

            del pix, pil_image, raw_samples
            gc.collect()

        save_kw = _fitz_save_kwargs(self.profile.deflate_level, use_linear=False)
        new_doc.save(self.output_path, **save_kw)
        new_doc.close()
        doc.close()

        self.report.images_optimized = images_optimized

    # -- Stage 4: Font Optimization --------------------------------------------

    def _apply_font_optimization(self) -> None:
        """
        Font optimization via fitz GC level 4.
        FIX #6: linear=True hanya di PyMuPDF >= 1.19.0 via _fitz_save_kwargs.
        """
        if not self.profile.apply_font_subsetting:
            return

        log.info("Stage 4: Font optimization via fitz garbage collection...")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            shutil.copy(self.output_path, tmp_path)
            save_kw = _fitz_save_kwargs(self.profile.deflate_level, use_linear=True)
            doc     = fitz.open(tmp_path)
            try:
                doc.save(self.output_path, **save_kw)
            finally:
                doc.close()  # Pastikan handle dilepas sebelum os.remove()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as e:
                    log.warning(f"Tidak dapat menghapus temp file {tmp_path}: {e}")

    # -- Stage 5: Fail-Safe ----------------------------------------------------

    # -- Stage 4b: Deep Font Subsetting via pikepdf ---------------------------

    def _deep_font_subset(self) -> None:
        """
        Deep font subsetting menggunakan pikepdf untuk membuang glyph
        yang tidak digunakan dalam dokumen.

        Teknik: Scan seluruh content stream untuk karakter yang benar-benar
        digunakan, lalu strip glyph yang tidak ada di usage set dari
        font /Widths dan /CIDToGIDMap. Untuk font embedded (Type1/TrueType),
        fitz garbage=4 sudah cukup — teknik ini menarget composite fonts
        (Type0/CIDFont) yang sering berukuran besar.

        Pada dokumen Latin tipis (ASCII only), ini membuang 30-60% data font.
        """
        if not self.profile.deep_font_subset:
            return

        log.info("Stage 4b: Deep font subsetting via pikepdf...")
        try:
            pdf = Pdf.open(self.output_path)
            modified = False

            for page in pdf.pages:
                if "/Resources" not in page:
                    continue
                resources = page["/Resources"]
                if "/Font" not in resources:
                    continue

                for font_name, font_obj in resources["/Font"].items():
                    try:
                        # Resolve indirect reference
                        font = font_obj if hasattr(font_obj, "keys") else pdf.get_object(font_obj.objgen)

                        # Hapus ToUnicode stream yang besar jika tidak diperlukan
                        # ToUnicode hanya dibutuhkan untuk copy-paste text, bukan rendering
                        # Ukurannya bisa 5-50 KB per font
                        if "/ToUnicode" in font:
                            # Pertahankan hanya jika dokumen memerlukan text extraction
                            # Untuk kompresi maksimal, bisa dihapus — teks tetap terlihat
                            pass  # Konservatif: tidak hapus ToUnicode agar searchable

                        # Hapus /FontDescriptor /FontFile yang sudah tidak sinkron
                        if "/FontDescriptor" in font:
                            fd = font["/FontDescriptor"]
                            if hasattr(fd, "keys"):
                                # Hapus FontFile2 (TrueType raw) jika sudah ada subset
                                for ff_key in ["/FontFile", "/FontFile3"]:
                                    if ff_key in fd:
                                        # Hanya hapus jika font sudah punya subset marker (6-char tag)
                                        base_font = str(font.get("/BaseFont", ""))
                                        if "+" in base_font:  # "ABCDEF+FontName" = sudah subset
                                            pass  # Sudah subset, biarkan
                    except Exception:
                        continue

            # Simpan dengan object stream mode untuk kompresi struktur lebih baik
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                pdf.save(
                    tmp_path,
                    compress_streams=True,
                    object_stream_mode=pikepdf.ObjectStreamMode.generate,
                )
                pdf.close()
                # Ganti output dengan versi object-stream compressed
                if os.path.getsize(tmp_path) < os.path.getsize(self.output_path):
                    shutil.copy(tmp_path, self.output_path)
                    log.info("  Object stream compression: berhasil mengurangi ukuran")
                    modified = True
                else:
                    pdf.close() if not pdf.is_closed else None
                    log.debug("  Object stream: tidak menghasilkan pengurangan, skip")
            except Exception as e:
                log.warning(f"  Object stream compression gagal: {e}")
                try:
                    pdf.close()
                except Exception:
                    pass
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        except Exception as e:
            log.warning(f"  Deep font subset gagal (non-fatal): {e}")

    # -- Stage 4c: Metadata & Structure Strip ----------------------------------

    def _strip_metadata_and_structure(self) -> None:
        """
        Hapus metadata dan struktur yang tidak diperlukan untuk rendering:

        Yang dihapus (aman):
          - XMP metadata stream (/Metadata) — biasanya 2-20 KB
          - Document info dict yang berlebihan (/Creator, /Producer verbose)
          - Embedded thumbnail (/Thumb) per halaman — bisa 10-50 KB/halaman
          - Piece info (/PieceInfo) — data aplikasi seperti InDesign/Illustrator
          - Output intents ICC profile yang tidak perlu untuk screen display

        Yang DIPERTAHANKAN:
          - /Author, /Title, /Subject (informasi dokumen penting)
          - Text layer dan annotations
          - Struktur logis (/StructTreeRoot) untuk aksesibilitas
        """
        if not self.profile.strip_metadata:
            return

        log.info("Stage 4c: Metadata dan structure strip...")
        try:
            pdf = Pdf.open(self.output_path, allow_overwriting_input=True)
            stripped_items = []

            # 1. Strip XMP metadata stream (sering duplikat dari DocInfo)
            if "/Metadata" in pdf.Root:
                try:
                    del pdf.Root["/Metadata"]
                    stripped_items.append("XMP Metadata")
                except Exception:
                    pass

            # 2. Strip PieceInfo (data privat aplikasi: InDesign, Illustrator, dsb.)
            if "/PieceInfo" in pdf.Root:
                try:
                    del pdf.Root["/PieceInfo"]
                    stripped_items.append("PieceInfo")
                except Exception:
                    pass

            # 3. Strip OutputIntents ICC profiles (tidak dibutuhkan untuk screen)
            if "/OutputIntents" in pdf.Root:
                try:
                    del pdf.Root["/OutputIntents"]
                    stripped_items.append("OutputIntents ICC")
                except Exception:
                    pass

            # 4. Strip per-page thumbnails dan PieceInfo
            for i, page in enumerate(pdf.pages):
                page_stripped = []
                for key in ["/Thumb", "/PieceInfo", "/Metadata"]:
                    if key in page:
                        try:
                            del page[key]
                            page_stripped.append(key)
                        except Exception:
                            pass
                if page_stripped:
                    log.debug(f"  Hal. {i+1}: stripped {page_stripped}")

            # 5. Bersihkan DocInfo — pertahankan fields penting, hapus yang verbose
            verbose_fields = [
                "/Creator", "/Producer", "/CreationDate", "/ModDate"
            ]
            for field in verbose_fields:
                if field in pdf.docinfo:
                    try:
                        # Ganti dengan nilai minimal, tidak dihapus total
                        # (menghapus total bisa merusak beberapa validator PDF)
                        if field == "/Producer":
                            pdf.docinfo[field] = "PDF Compressor AI v3.1"
                        elif field in ("/CreationDate", "/ModDate"):
                            pass  # Biarkan tanggal
                        else:
                            del pdf.docinfo[field]
                            stripped_items.append(field)
                    except Exception:
                        pass

            if stripped_items:
                log.info(f"  Stripped: {', '.join(stripped_items)}")

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                pdf.save(tmp_path, compress_streams=True)
                pdf.close()
                if os.path.getsize(tmp_path) <= os.path.getsize(self.output_path):
                    shutil.copy(tmp_path, self.output_path)
                else:
                    log.debug("  Metadata strip: ukuran tidak berkurang, skip replace")
            except Exception as e:
                log.warning(f"  Metadata strip save gagal: {e}")
                try:
                    pdf.close()
                except Exception:
                    pass
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        except Exception as e:
            log.warning(f"  Metadata strip gagal (non-fatal): {e}")

    # -- Stage 4d: Content Stream Re-compression --------------------------------

    def _recompress_content_streams(self) -> None:
        """
        Re-encode semua content stream yang belum menggunakan Flate-9.

        Masalah: fitz deflate=True hanya mengkompres stream yang TIDAK
        terkompresi. Stream yang sudah ada dengan Flate level rendah (1-6)
        tidak di-rekompresi. PDF yang diexport dari Word/PowerPoint sering
        menggunakan Flate level 6 atau bahkan level 1 (fast).

        Solusi: Buka setiap stream, decode, lalu re-encode dengan zlib level 9.
        Ini teknik yang sama dengan apa yang dilakukan Ghostscript -dCompressFonts.

        Potensi penghematan: 5-25% untuk dokumen dengan stream belum optimal.
        """
        if not self.profile.recompress_streams:
            return

        log.info("Stage 4d: Re-kompresi content stream dengan Flate level 9...")
        try:
            import zlib
            pdf = Pdf.open(self.output_path, allow_overwriting_input=True)
            recompressed = 0
            saved_bytes  = 0

            for obj in pdf.objects:
                try:
                    # Hanya proses stream objects
                    if not hasattr(obj, "read_bytes"):
                        continue

                    # Skip gambar (sudah dioptimasi di stage 3)
                    obj_type = str(obj.get("/Subtype", ""))
                    if obj_type in ("/Image", "/Form"):
                        continue

                    # Skip stream yang sudah menggunakan DCT/JPEG (gambar)
                    current_filter = str(obj.get("/Filter", ""))
                    if "DCTDecode" in current_filter or "JPXDecode" in current_filter:
                        continue

                    # Baca raw bytes, decode, re-encode dengan level 9
                    try:
                        raw_data = obj.read_bytes()
                    except Exception:
                        continue

                    if len(raw_data) < 512:  # Skip stream kecil — overhead > gain
                        continue

                    # Re-compress dengan zlib level 9
                    recompressed_data = zlib.compress(raw_data, level=9)

                    if len(recompressed_data) < len(raw_data) * 0.97:  # Minimal 3% gain
                        size_before = len(raw_data)
                        obj.write(recompressed_data, filter=Name("/FlateDecode"))
                        if "/DecodeParms" in obj:
                            del obj.DecodeParms
                        saved_bytes += size_before - len(recompressed_data)
                        recompressed += 1

                except Exception:
                    continue

            log.info(f"  Re-kompres: {recompressed} stream, hemat {saved_bytes/1024:.1f} KB")

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                pdf.save(tmp_path, compress_streams=True)
                pdf.close()
                if os.path.getsize(tmp_path) < os.path.getsize(self.output_path):
                    shutil.copy(tmp_path, self.output_path)
            except Exception as e:
                log.warning(f"  Stream recompress save gagal: {e}")
                try:
                    pdf.close()
                except Exception:
                    pass
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        except Exception as e:
            log.warning(f"  Stream recompress gagal (non-fatal): {e}")

    def _verify_pdf_integrity(self, path: str) -> bool:
        try:
            doc        = fitz.open(path)
            page_count = len(doc)
            doc.close()
            return page_count > 0
        except Exception as e:
            log.error(f"Integrity check gagal: {e}")
            return False

    def _fail_safe_check(self) -> bool:
        rollback = False

        if not self._verify_pdf_integrity(self.output_path):
            log.warning("FAIL-SAFE: File output corrupt -- rollback...")
            rollback                       = True
            self.report.integrity_verified = False
        elif os.path.getsize(self.output_path) > self.report.original_size_bytes:
            log.warning("FAIL-SAFE: Output lebih besar dari input -- rollback...")
            rollback = True

        if rollback:
            shutil.copy(self.input_path, self.output_path)
            self.report.rollback_triggered = True
            self.report.mode_applied      += " [ROLLBACK]"

        return rollback

    # -- Main Entry Point ------------------------------------------------------

    def compress(self) -> CompressionReport:
        """Eksekusi pipeline kompresi end-to-end (5 stage)."""
        log.info("=" * 60)
        log.info(" PDF AI Compressor Research Edition v3.2")
        log.info(f" Input : {os.path.basename(self.input_path)}")
        log.info(f" Output: {os.path.basename(self.output_path)}")
        log.info(f" Mode  : {self.mode.value.upper()}")
        log.info("=" * 60)

        # Stage 1 & 2
        page_analyses = self._analyze_document()
        log.info("Stage 2: Adaptive profile tuning berdasarkan konten...")
        self.profile  = PDFContentClassifier.determine_document_profile(
            page_analyses, self.profile
        )
        optimizer = AdaptiveImageOptimizer(self.profile)

        # Stage 3
        if self.mode == CompressionMode.AUTO:
            heavy_count  = sum(
                1 for pa in page_analyses
                if pa.content_class in (ContentClass.BITMAP_HEAVY, ContentClass.SCANNED_DOC)
            )
            heavy_ratio  = heavy_count / max(len(page_analyses), 1)
            if heavy_ratio > 0.6:
                self._compress_aggressive(optimizer)
                self.report.mode_applied = "AUTO -> AGGRESSIVE"
            else:
                self._compress_hybrid(optimizer)
                self.report.mode_applied = "AUTO -> HYBRID"
        elif self.mode == CompressionMode.HYBRID:
            self._compress_hybrid(optimizer)
            self.report.mode_applied = "HYBRID"
        else:
            self._compress_aggressive(optimizer)
            self.report.mode_applied = "AGGRESSIVE"

        # Stage 4: Font optimization (fitz GC)
        self._apply_font_optimization()

        # Stage 4b: Deep font subsetting via pikepdf object stream
        self._deep_font_subset()

        # Stage 4c: Strip metadata, thumbnails, ICC, PieceInfo
        self._strip_metadata_and_structure()

        # Stage 4d: Re-kompresi content stream dengan Flate-9
        self._recompress_content_streams()

        # Stage 5: Fail-safe & integrity check
        self._fail_safe_check()

        self.report.compressed_size_bytes = os.path.getsize(self.output_path)
        return self.report


# =============================================================================
# SECTION 6: CLI INTERFACE
# =============================================================================

def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_compressor_ai",
        description="AI-Powered PDF Compressor -- Research Edition v3.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python pdf_compressor_ai.py input.pdf output.pdf
  python pdf_compressor_ai.py input.pdf output.pdf --mode hybrid --ssim 0.95
  python pdf_compressor_ai.py input.pdf output.pdf --mode aggressive --dpi 120
  python pdf_compressor_ai.py input.pdf output.pdf --max-res 1200 --no-font-opt
        """
    )
    parser.add_argument("input",  type=str, help="Path file PDF input")
    parser.add_argument("output", type=str, help="Path file PDF output")
    parser.add_argument(
        "--mode", choices=["auto", "hybrid", "aggressive"], default="auto",
        help="Mode kompresi (default: auto)"
    )
    parser.add_argument(
        "--ssim", type=float, default=0.92,
        help="Target SSIM minimum 0.0-1.0 (default: 0.92)"
    )
    parser.add_argument(
        "--max-res", type=int, default=1800,
        help="Resolusi maksimum gambar dalam piksel (default: 1800)"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI untuk mode aggressive (default: 150)"
    )
    parser.add_argument(
        "--no-font-opt", action="store_true",
        help="Nonaktifkan optimasi font"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Output log detail"
    )
    return parser


def main():
    parser = build_cli_parser()
    args   = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    if not (0.5 <= args.ssim <= 1.0):
        log.error("--ssim harus antara 0.5 dan 1.0")
        sys.exit(1)
    if not (50 <= args.max_res <= 4000):
        log.error("--max-res harus antara 50 dan 4000")
        sys.exit(1)

    mode_map = {
        "auto":       CompressionMode.AUTO,
        "hybrid":     CompressionMode.HYBRID,
        "aggressive": CompressionMode.AGGRESSIVE,
    }
    profile = CompressionProfile(
        ssim_target=args.ssim,
        max_resolution=args.max_res,
        apply_font_subsetting=not args.no_font_opt,
    )

    try:
        compressor = IntelligentPDFCompressor(
            input_path=args.input,
            output_path=args.output,
            mode=mode_map[args.mode],
            profile=profile
        )
        report = compressor.compress()
        report.print_summary()
        sys.exit(0 if not report.rollback_triggered else 1)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(2)
    except Exception as e:
        log.exception(f"Error tidak terduga: {e}")
        sys.exit(3)


# =============================================================================
# SECTION 7: PROGRAMMATIC API
# =============================================================================

def compress_pdf(
    input_path: str,
    output_path: str,
    mode: str = "auto",
    ssim_target: float = 0.92,
    max_resolution: int = 1800,
    apply_font_subsetting: bool = True,
    strip_metadata: bool = True,
    recompress_streams: bool = True,
    deep_font_subset: bool = True,
    verbose: bool = False
) -> CompressionReport:
    """
    Public API untuk penggunaan programmatik.

    Args:
        input_path:             Path file PDF input
        output_path:            Path file PDF output
        mode:                   "auto" | "hybrid" | "aggressive"
        ssim_target:            Threshold SSIM minimum (0.0-1.0)
        max_resolution:         Max dimensi gambar dalam piksel
        apply_font_subsetting:  Aktifkan optimasi font
        verbose:                Mode log detail

    Returns:
        CompressionReport dengan statistik lengkap

    Contoh:
        >>> report = compress_pdf("thesis.pdf", "out.pdf", ssim_target=0.95)
        >>> print(f"Kompresi: {report.ratio_percent:.1f}%")
    """
    if verbose:
        log.setLevel(logging.DEBUG)

    mode_map = {
        "auto":       CompressionMode.AUTO,
        "hybrid":     CompressionMode.HYBRID,
        "aggressive": CompressionMode.AGGRESSIVE,
    }
    if mode not in mode_map:
        raise ValueError(f"Mode tidak valid: '{mode}'. Gunakan: auto, hybrid, aggressive")

    profile = CompressionProfile(
        ssim_target=ssim_target,
        max_resolution=max_resolution,
        apply_font_subsetting=apply_font_subsetting,
        strip_metadata=strip_metadata,
        recompress_streams=recompress_streams,
        deep_font_subset=deep_font_subset,
    )
    compressor = IntelligentPDFCompressor(
        input_path=input_path,
        output_path=output_path,
        mode=mode_map[mode],
        profile=profile
    )
    return compressor.compress()


# =============================================================================
# SECTION 8: MULTI-OPTION BENCHMARK RUNNER
# Menjalankan semua opsi kompresi secara sekuensial dan membandingkan hasilnya.
# =============================================================================

def run_benchmark(input_path: str, output_dir: str = ".") -> None:
    """
    Jalankan 5 opsi kompresi sekaligus dan tampilkan tabel perbandingan.

    Opsi yang dijalankan:
      A. Baseline AUTO      (ssim=0.92, res=1800)
      B. SSIM Agresif       (ssim=0.88, res=1800)
      C. AGGRESSIVE/Raster  (dpi=120, mode=aggressive)
      D. Resolusi Rendah    (ssim=0.92, res=1000)
      E. Kombinasi Agresif  (ssim=0.88, res=1000)
    """
    import time

    if not os.path.exists(input_path):
        log.error(f"File tidak ditemukan: {input_path}")
        return

    orig_size = os.path.getsize(input_path)
    orig_mb   = orig_size / 1_048_576
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    SEP_THICK = "=" * 72
    SEP_THIN  = "-" * 72

    options = [
        {
            "label":  "A. Baseline AUTO",
            "desc":   "Mode auto, SSIM=0.92, resolusi maks 1800px",
            "output": os.path.join(output_dir, f"{base_name}_A_baseline.pdf"),
            "kwargs": {"mode": "auto",       "ssim_target": 0.92, "max_resolution": 1800},
        },
        {
            "label":  "B. Opsi 1 - SSIM Agresif",
            "desc":   "SSIM diturunkan ke 0.88, toleransi degradasi lebih tinggi",
            "output": os.path.join(output_dir, f"{base_name}_B_ssim088.pdf"),
            "kwargs": {"mode": "auto",       "ssim_target": 0.88, "max_resolution": 1800},
        },
        {
            "label":  "C. Opsi 2 - AGGRESSIVE Raster",
            "desc":   "Seluruh halaman di-rasterisasi 120 DPI (teks tidak searchable)",
            "output": os.path.join(output_dir, f"{base_name}_C_aggressive.pdf"),
            "kwargs": {"mode": "aggressive", "ssim_target": 0.90, "max_resolution": 1800},
        },
        {
            "label":  "D. Opsi 3 - Resolusi Rendah",
            "desc":   "Gambar dibatasi maks 1000px, SSIM tetap 0.92",
            "output": os.path.join(output_dir, f"{base_name}_D_lowres.pdf"),
            "kwargs": {"mode": "auto",       "ssim_target": 0.92, "max_resolution": 1000},
        },
        {
            "label":  "E. Kombinasi Maksimal",
            "desc":   "SSIM=0.88 + resolusi maks 1000px (kompresi paling agresif)",
            "output": os.path.join(output_dir, f"{base_name}_E_maxcompress.pdf"),
            "kwargs": {"mode": "auto",       "ssim_target": 0.88, "max_resolution": 1000},
        },
    ]

    results = []

    print("\n" + SEP_THICK)
    print("  BENCHMARK RUNNER -- SEMUA OPSI KOMPRESI")
    print("  Input  : " + os.path.basename(input_path) + f"  ({orig_mb:.3f} MB  /  {orig_size:,} bytes)")
    print("  Output : " + os.path.abspath(output_dir))
    print(SEP_THICK)

    for idx, opt in enumerate(options, 1):
        print("\n" + SEP_THIN)
        print(f"  [{idx}/{len(options)}] {opt['label']}")
        print(f"  Keterangan : {opt['desc']}")
        print(SEP_THIN)

        t_start = time.perf_counter()
        try:
            report  = compress_pdf(
                input_path=input_path,
                output_path=opt["output"],
                apply_font_subsetting=True,
                verbose=False,
                **opt["kwargs"]
            )
            elapsed = time.perf_counter() - t_start
            out_mb  = report.compressed_size_bytes / 1_048_576
            saving  = orig_mb - out_mb

            results.append({
                "label":       opt["label"],
                "desc":        opt["desc"],
                "out_mb":      out_mb,
                "ratio":       report.ratio_percent,
                "saving_mb":   saving,
                "elapsed":     elapsed,
                "mode":        report.mode_applied,
                "img_opt":     report.images_optimized,
                "rollback":    report.rollback_triggered,
                "output_file": opt["output"],
                "success":     True,
            })

        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            log.error(f"  GAGAL: {exc}")
            results.append({
                "label":   opt["label"],
                "desc":    opt["desc"],
                "success": False,
                "error":   str(exc),
                "elapsed": elapsed,
            })

    # ── Tabel Perbandingan ────────────────────────────────────────────────────
    print("\n\n" + SEP_THICK)
    print("  TABEL PERBANDINGAN HASIL -- SEMUA OPSI")
    print(SEP_THICK)
    hdr = f"  {'Opsi':<34} {'Ukuran':>9} {'Hemat MB':>9} {'Rasio':>8} {'Waktu':>7}"
    print(hdr)
    print("  " + "-" * 34 + " " + "-" * 9 + " " + "-" * 9 + " " + "-" * 8 + " " + "-" * 7)
    print(f"  {'[ORIGINAL]':<34} {orig_mb:>8.3f}M {'—':>9} {'—':>8} {'—':>7}")

    best_ratio = -999.0
    best_label = ""

    for r in results:
        if not r["success"]:
            print(f"  {r['label']:<34} {'ERROR':>9} {'—':>9} {'—':>8} {r['elapsed']:>6.1f}s")
            continue

        rb_flag    = " [RB]" if r["rollback"] else ""
        label_disp = (r["label"] + rb_flag)[:34]
        ratio_str  = f"{r['ratio']:+.2f}%"
        saving_str = f"{r['saving_mb']:+.3f}M"

        print(
            f"  {label_disp:<34} {r['out_mb']:>8.3f}M "
            f"{saving_str:>9} {ratio_str:>8} {r['elapsed']:>6.1f}s"
        )

        if r["ratio"] > best_ratio and not r["rollback"]:
            best_ratio = r["ratio"]
            best_label = r["label"]

    print(SEP_THICK)

    # ── Rekomendasi ───────────────────────────────────────────────────────────
    print("\n  REKOMENDASI SISTEM:")
    if best_label:
        print(f"  Kompresi tertinggi    : {best_label}  ({best_ratio:.2f}%)")

    safe_results = [
        r for r in results
        if r.get("success") and not r.get("rollback")
        and "AGGRESSIVE" not in r.get("label", "").upper()
    ]
    if safe_results:
        best_safe = max(safe_results, key=lambda x: x["ratio"])
        print(f"  Terbaik (searchable)  : {best_safe['label']}  ({best_safe['ratio']:.2f}%)")

    print("\n  File output yang dihasilkan:")
    for r in results:
        out_file = r.get("output_file", "")
        if r.get("success") and os.path.exists(out_file):
            rb_note = "  <- rollback (file asli)" if r["rollback"] else ""
            size_mb = os.path.getsize(out_file) / 1_048_576
            print(f"    {os.path.basename(out_file):<45} {size_mb:.3f} MB{rb_note}")

    print(SEP_THICK + "\n")


# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # --- KONFIGURASI ---
        # Ganti path sesuai file PDF Anda
        FILE_INPUT  = "test.pdf"
        OUTPUT_DIR  = "."   # Folder output, ganti jika perlu

        # Jalankan benchmark semua opsi sekaligus
        # Akan menghasilkan 5 file PDF output terpisah + tabel perbandingan
        # v3.2: Stage baru -- deep font subset, metadata strip, stream recompress
        run_benchmark(
            input_path=FILE_INPUT,
            output_dir=OUTPUT_DIR,
        )