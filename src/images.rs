//! Image optimization: convert raster manifest images (JPEG/PNG) to WebP and
//! update every reference to them across the package.

use crate::opf::{ImageChange, ManifestItem};
use crate::util::{basename, quote, unquote, with_ext};
use anyhow::Result;
use image::{ColorType, DynamicImage};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use walkdir::WalkDir;
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

/// Output image format the optimizer re-encodes raster images to. WebP is the
/// EPUB 3.3 target; AVIF and JPEG XL become core media types in EPUB 3.4 and are
/// emitted under `--target 3.4` (experimental, behind the `epub34` feature).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFormat {
    #[default]
    WebP,
    Avif,
    Jxl,
}

impl ImageFormat {
    /// File extension (no dot) for this format's output files.
    pub fn extension(self) -> &'static str {
        match self {
            ImageFormat::WebP => "webp",
            ImageFormat::Avif => "avif",
            ImageFormat::Jxl => "jxl",
        }
    }

    /// OPF manifest media type for this format.
    pub fn media_type(self) -> &'static str {
        match self {
            ImageFormat::WebP => "image/webp",
            ImageFormat::Avif => "image/avif",
            ImageFormat::Jxl => "image/jxl",
        }
    }

    /// Short label for progress/log lines.
    pub fn label(self) -> &'static str {
        match self {
            ImageFormat::WebP => "WebP",
            ImageFormat::Avif => "AVIF",
            ImageFormat::Jxl => "JPEG XL",
        }
    }
}

/// How the output format is chosen for each image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatPolicy {
    /// The same format for every image (an explicit `--image-format`, or the
    /// EPUB 3.3 default of WebP).
    Fixed(ImageFormat),
    /// Content-adaptive (the EPUB 3.4 default): pick per image from its *source*
    /// format, which is a free content-type signal — publishers use JPEG for
    /// photographs and PNG for line-art/screenshots. Measured at equal perceptual
    /// quality: AVIF wins on photos, WebP wins on line-art (see
    /// docs/design/epub-3.4-image-codec-choice.md). So **JPEG → AVIF, PNG → WebP**.
    Auto,
    /// Thorough: encode every candidate format per image and keep the smallest
    /// (at matched quality, since `--quality` is calibrated). Best size, but
    /// pays the cost of multiple encodes per image (notably the slow AVIF one).
    Best,
}

impl FormatPolicy {
    /// The output format for a source of the given media type. Only meaningful
    /// for `Fixed`/`Auto`; `Best` selects per image by encoding all candidates,
    /// so it never calls this (the `WebP` arm is an unreachable safe default).
    pub fn format_for(self, media_type: &str) -> ImageFormat {
        match self {
            FormatPolicy::Fixed(f) => f,
            FormatPolicy::Auto => match media_type {
                "image/jpeg" | "image/jpg" => ImageFormat::Avif,
                _ => ImageFormat::WebP,
            },
            FormatPolicy::Best => ImageFormat::WebP,
        }
    }

    /// Short label for the conversion progress header.
    pub fn label(self) -> &'static str {
        match self {
            FormatPolicy::Fixed(f) => f.label(),
            FormatPolicy::Auto => "AVIF for photos, WebP for line-art",
            FormatPolicy::Best => "smallest per image",
        }
    }
}

/// Per-image size statistics, used by the report.
#[derive(Debug, Clone)]
pub struct ImageMetric {
    pub name: String,
    pub original_size: u64,
    pub new_size: u64,
    pub percentage: f64,
    /// True when the original was kept because the re-encode came out no smaller.
    pub kept: bool,
    /// The output format the image was re-encoded to (`None` when kept as-is).
    pub format: Option<ImageFormat>,
}

/// Result of the optimization pass.
pub struct OptimizeResult {
    pub metrics: Vec<ImageMetric>,
    /// Keyed by the raw manifest `href` -> how the manifest item is rewritten.
    pub manifest_changes: HashMap<String, ImageChange>,
    /// (old_reference, new_reference) pairs for updating document/style files.
    pub ref_pairs: Vec<(String, String)>,
}

/// Raster media types we convert, all mapped to `image/webp`.
fn is_target_media_type(mt: &str) -> bool {
    matches!(mt, "image/jpeg" | "image/jpg" | "image/png")
}

/// Convert all eligible manifest images to WebP at the given quality, deleting
/// originals and recording the manifest/reference changes.
pub fn optimize_images(
    package_dir: &Path,
    items: &[ManifestItem],
    cover_id: Option<&str>,
    quality: u8,
    policy: FormatPolicy,
    avif_speed: u8,
    progress: &dyn Fn(&str),
) -> Result<OptimizeResult> {
    let mut result = OptimizeResult {
        metrics: Vec::new(),
        manifest_changes: HashMap::new(),
        ref_pairs: Vec::new(),
    };

    // At most one manifest item may carry `properties="cover-image"` —
    // epubcheck rejects duplicates (e.g. `cover.jpg` + `cover_thumb.jpg` both
    // matching a name heuristic). Prefer the OPF-declared cover id; fall back
    // to the first name-based match only when no declared id matches any item.
    let chosen_cover_id: Option<&str> = cover_id
        .filter(|c| items.iter().any(|i| i.id == *c))
        .or_else(|| {
            items
                .iter()
                .filter(|i| is_target_media_type(&i.media_type))
                .find(|i| {
                    i.id.to_lowercase().contains("cover") || i.href.to_lowercase().contains("cover")
                })
                .map(|i| i.id.as_str())
        });

    // Output hrefs written so far in this run, to disambiguate collisions
    // (`foo.jpg` and `foo.png` both map to `foo.webp`).
    let mut produced_hrefs: HashSet<String> = HashSet::new();

    for item in items {
        if !is_target_media_type(&item.media_type) {
            continue;
        }

        // Hrefs are URL-encoded; decode to obtain the real on-disk path.
        let decoded_href = unquote(&item.href);
        // The href is attacker-controlled: refuse anything that would resolve
        // outside the package dir (`..`, absolute paths) — the write/delete
        // below must never touch files outside the extraction sandbox.
        if !crate::util::is_contained_path(&decoded_href) {
            progress(&format!(
                "  [!] Warning: skipping image with unsafe path: {}",
                item.href
            ));
            continue;
        }
        let img_path = package_dir.join(&decoded_href);

        if !img_path.exists() {
            progress(&format!(
                "  [!] Warning: Image file not found: {}",
                basename(&decoded_href)
            ));
            continue;
        }

        let old_name = basename(&decoded_href).to_string();

        // Read the original bytes once: we need its size, its raw bytes (to
        // estimate JPEG quality), and its decoded pixels.
        let orig_bytes = match fs::read(&img_path) {
            Ok(b) => b,
            Err(e) => {
                progress(&format!("  [!] Failed to read image {old_name}: {e}"));
                continue;
            }
        };
        let orig_size = orig_bytes.len() as u64;

        let dynimg = match decode_image(&orig_bytes) {
            Ok(d) => d,
            Err(e) => {
                progress(&format!("  [!] Failed to decode image {old_name}: {e}"));
                continue;
            }
        };

        // (2) Never re-encode *above* the source quality. A chart saved as a
        // q43 JPEG gains nothing from a q80 WebP — it only inflates. For JPEG
        // sources we cap the WebP quality at the source's estimated quality.
        let is_jpeg = matches!(item.media_type.as_str(), "image/jpeg" | "image/jpg");
        let eff_quality = if is_jpeg {
            estimate_jpeg_quality(&orig_bytes)
                .map(|q| quality.min(q))
                .unwrap_or(quality)
        } else {
            quality
        };

        // (3) Encode. Fixed/Auto is a single format; `Best` tries every candidate
        // and keeps the smallest — valid because `--quality` is calibrated to equal
        // perceptual quality across codecs, so smallest bytes = smallest at matched
        // quality. (Grayscale stays grayscale inside each encoder.) Encode to memory
        // first so we can compare sizes against the original.
        let candidates: Vec<ImageFormat> = match policy {
            FormatPolicy::Fixed(f) => vec![f],
            FormatPolicy::Auto => vec![policy.format_for(&item.media_type)],
            FormatPolicy::Best => vec![ImageFormat::WebP, ImageFormat::Avif, ImageFormat::Jxl],
        };
        let mut best: Option<(ImageFormat, Vec<u8>)> = None;
        for cand in candidates {
            if let Ok(bytes) = encode_image(&dynimg, eff_quality, cand, avif_speed)
                && best.as_ref().is_none_or(|(_, b)| bytes.len() < b.len())
            {
                best = Some((cand, bytes));
            }
        }
        let (format, encoded) = match best {
            Some(x) => x,
            None => {
                progress(&format!("  [!] Failed to convert image {old_name}"));
                continue;
            }
        };
        let new_size = encoded.len() as u64;

        // (1) Keep whichever is smaller. If the re-encode isn't actually smaller,
        // leave the original untouched — no rewrite, no manifest/reference change —
        // so the output can never grow.
        if new_size >= orig_size {
            progress(&format!(
                "  [=] Kept original ({} not smaller): {} ({:.1}KB <= {:.1}KB)",
                format.label(),
                old_name,
                orig_size as f64 / 1024.0,
                new_size as f64 / 1024.0
            ));
            result.metrics.push(ImageMetric {
                name: old_name,
                original_size: orig_size,
                new_size: orig_size,
                percentage: 0.0,
                kept: true,
                format: None,
            });
            continue;
        }

        // The output extension/path follow the chosen format (known only now for
        // the `Best` policy, which selects per image). Never clobber a file that
        // already exists in the package or an output produced earlier in this
        // run (`foo.jpg` + `foo.png` -> `foo.webp` + `foo_2.webp`).
        let new_href = disambiguate_href(
            package_dir,
            &with_ext(&decoded_href, format.extension()),
            &produced_hrefs,
        );
        let new_img_path = package_dir.join(&new_href);
        if let Err(e) = fs::write(&new_img_path, &encoded) {
            progress(&format!(
                "  [!] Failed to write {}: {e}",
                basename(&new_href)
            ));
            continue;
        }
        produced_hrefs.insert(new_href.clone());
        let _ = fs::remove_file(&img_path);

        let pct = (orig_size as i64 - new_size as i64) as f64 / orig_size as f64 * 100.0;
        let new_name = basename(&new_href).to_string();

        result.metrics.push(ImageMetric {
            name: old_name.clone(),
            original_size: orig_size,
            new_size,
            percentage: pct,
            kept: false,
            format: Some(format),
        });

        progress(&format!(
            "  [+] Converted: {} -> {} ({:.1}KB -> {:.1}KB, {:.1}% saved)",
            old_name,
            new_name,
            orig_size as f64 / 1024.0,
            new_size as f64 / 1024.0,
            pct
        ));

        // Only the single item chosen above may be flagged as the cover image.
        let is_cover = chosen_cover_id == Some(item.id.as_str());

        result.manifest_changes.insert(
            item.href.clone(),
            ImageChange {
                new_href_encoded: quote(&new_href),
                media_type: format.media_type().to_string(),
                is_cover,
            },
        );

        // Both the raw and decoded hrefs should be remapped to the new href.
        push_unique(&mut result.ref_pairs, item.href.clone(), new_href.clone());
        push_unique(
            &mut result.ref_pairs,
            decoded_href.clone(),
            new_href.clone(),
        );
    }

    Ok(result)
}

/// Re-encode a standalone raster image (e.g. an import cover) to WebP at
/// `quality`, returning the WebP bytes **only if they're smaller** than the
/// original (the same size-safe rule the EPUB optimizer uses). `None` means
/// "keep the original" — decode failed, or the re-encode wasn't smaller.
///
/// Unlike [`optimize_images`], this works on in-memory bytes with no manifest,
/// so the Markdown importer can shrink a cover before embedding it.
#[cfg(feature = "markdown")]
pub fn optimize_to_webp(original: &[u8], quality: u8) -> Option<Vec<u8>> {
    let dynimg = decode_image(original).ok()?;
    // Don't re-encode a JPEG above its source quality (it would only inflate).
    let is_jpeg = matches!(image::guess_format(original), Ok(image::ImageFormat::Jpeg));
    let q = if is_jpeg {
        estimate_jpeg_quality(original)
            .map(|src| quality.min(src))
            .unwrap_or(quality)
    } else {
        quality
    };
    let webp = encode_webp(&dynimg, q).ok()?;
    (webp.len() < original.len()).then_some(webp)
}

fn push_unique(pairs: &mut Vec<(String, String)>, old: String, new: String) {
    if !pairs.iter().any(|(o, _)| *o == old) {
        pairs.push((old, new));
    }
}

/// Insert `_{n}` before the extension of an href's basename
/// (`img/foo.webp` -> `img/foo_2.webp`).
fn numbered_href(href: &str, n: usize) -> String {
    let slash = href.rfind('/').map(|i| i as isize).unwrap_or(-1);
    match href.rfind('.') {
        Some(dot) if (dot as isize) > slash => format!("{}_{n}{}", &href[..dot], &href[dot..]),
        _ => format!("{href}_{n}"),
    }
}

/// Pick an output href that collides neither with a file already on disk under
/// `package_dir` (e.g. a pre-existing `foo.webp` shipped in the source EPUB)
/// nor with an href produced earlier in this run, appending `_2`, `_3`, … to
/// the stem until it is free.
fn disambiguate_href(package_dir: &Path, href: &str, produced: &HashSet<String>) -> String {
    if !produced.contains(href) && !package_dir.join(href).exists() {
        return href.to_string();
    }
    let mut n = 2usize;
    loop {
        let candidate = numbered_href(href, n);
        if !produced.contains(&candidate) && !package_dir.join(&candidate).exists() {
            return candidate;
        }
        n += 1;
    }
}

/// Bounds on a single image decode, to defend against decode-bombs (e.g. a tiny
/// PNG declaring 100k×100k). Comfortably above any real e-book image.
const MAX_IMAGE_DIM: u32 = 16_384;
const MAX_IMAGE_ALLOC: u64 = 512 * 1024 * 1024; // 512 MiB

/// Decode an in-memory image, enforcing the decode-bomb limits.
fn decode_image(bytes: &[u8]) -> Result<DynamicImage> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIM);
    limits.max_image_height = Some(MAX_IMAGE_DIM);
    limits.max_alloc = Some(MAX_IMAGE_ALLOC);
    reader.limits(limits);
    Ok(reader.decode()?)
}

/// Encode a decoded image to the requested output `format` at `quality` (1-100).
/// AVIF / JPEG XL require the experimental `epub34` build feature.
fn encode_image(
    dynimg: &DynamicImage,
    quality: u8,
    format: ImageFormat,
    avif_speed: u8,
) -> Result<Vec<u8>> {
    #[cfg(not(feature = "epub34"))]
    let _ = avif_speed;
    match format {
        ImageFormat::WebP => encode_webp(dynimg, quality),
        #[cfg(feature = "epub34")]
        ImageFormat::Avif => encode_avif(dynimg, quality, avif_speed),
        #[cfg(feature = "epub34")]
        ImageFormat::Jxl => encode_jxl(dynimg, quality),
        #[cfg(not(feature = "epub34"))]
        ImageFormat::Avif | ImageFormat::Jxl => anyhow::bail!(
            "{} output requires building epublift with the `epub34` feature",
            format.label()
        ),
    }
}

/// Encode a decoded image to WebP bytes at `quality` (1-100), choosing the
/// pixel layout that matches the image: grayscale images stay grayscale (L8 /
/// La8) so we never spend bits on empty colour channels.
fn encode_webp(dynimg: &DynamicImage, quality: u8) -> Result<Vec<u8>> {
    let config = LossyConfig::new().with_quality(quality as f32);
    let gray = matches!(
        dynimg.color(),
        ColorType::L8 | ColorType::La8 | ColorType::L16 | ColorType::La16
    );
    let alpha = dynimg.color().has_alpha();

    let memory = match (gray, alpha) {
        (true, false) => {
            let g = dynimg.to_luma8();
            let (w, h) = (g.width(), g.height());
            EncodeRequest::lossy(&config, &g, PixelLayout::L8, w, h).encode()?
        }
        (true, true) => {
            let g = dynimg.to_luma_alpha8();
            let (w, h) = (g.width(), g.height());
            EncodeRequest::lossy(&config, &g, PixelLayout::La8, w, h).encode()?
        }
        (false, true) => {
            let rgba = dynimg.to_rgba8();
            let (w, h) = (rgba.width(), rgba.height());
            EncodeRequest::lossy(&config, &rgba, PixelLayout::Rgba8, w, h).encode()?
        }
        (false, false) => {
            let rgb = dynimg.to_rgb8();
            let (w, h) = (rgb.width(), rgb.height());
            EncodeRequest::lossy(&config, &rgb, PixelLayout::Rgb8, w, h).encode()?
        }
    };
    Ok(memory)
}

/// Map a WebP-scale `--quality` (1-100, the reference) to the AVIF quality knob
/// that yields the *same* perceptual quality (butteraugli) — so `--quality N`
/// means the same thing across codecs. **Quadratic** fit over 40 JPEG-source
/// images from 11 books (the relationship curves up steeply at high quality;
/// a line under-shoots there). AVIF is only used for photographic (JPEG) content,
/// where this is calibrated. The input is clamped to the fitted range so the
/// parabola never turns back up on out-of-range inputs.
/// See docs/design/epub-3.4-image-codec-choice.md.
#[cfg(feature = "epub34")]
fn calibrated_avif_quality(webp_quality: u8) -> f32 {
    let q = (webp_quality as f32).clamp(45.0, 95.0);
    (0.017529 * q * q - 1.70862 * q + 95.783).clamp(1.0, 100.0)
}

/// Map a WebP-scale `--quality` (1-100) to the JPEG XL butteraugli *distance*
/// that matches WebP's perceptual quality (lower distance = higher quality).
/// Same quadratic 40-image / 11-book calibration as above.
#[cfg(feature = "epub34")]
fn calibrated_jxl_distance(webp_quality: u8) -> f32 {
    let q = (webp_quality as f32).clamp(45.0, 95.0);
    (-0.000494 * q * q + 0.01808 * q + 3.639).clamp(0.4, 15.0)
}

/// Encode to AVIF via the pure-Rust imazen `zenavif` (rav1e) encoder, `quality`
/// 1-100 (calibrated to the WebP scale). Grayscale is expanded to RGB (zenavif
/// encodes RGB/RGBA only). Behind the experimental `epub34` feature.
#[cfg(feature = "epub34")]
fn encode_avif(dynimg: &DynamicImage, quality: u8, speed: u8) -> Result<Vec<u8>> {
    use rgb::FromSlice;
    let cfg = zenavif::EncoderConfig::new()
        .quality(calibrated_avif_quality(quality))
        .speed(speed.clamp(1, 10));
    let stop = almost_enough::StopToken::new(zenavif::Unstoppable);
    let encoded = if dynimg.color().has_alpha() {
        let rgba = dynimg.to_rgba8();
        let (w, h) = (rgba.width() as usize, rgba.height() as usize);
        let img = imgref::Img::new(rgba.as_raw().as_rgba(), w, h);
        zenavif::encode_rgba8(img, &cfg, stop)?
    } else {
        let rgb = dynimg.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let img = imgref::Img::new(rgb.as_raw().as_rgb(), w, h);
        zenavif::encode_rgb8(img, &cfg, stop)?
    };
    Ok(encoded.avif_file)
}

/// Encode to JPEG XL via the pure-Rust imazen `zenjxl` encoder, lossy at the
/// distance mapped from `quality` 1-100. Grayscale is expanded to RGB. Behind
/// the experimental `epub34` feature. See docs/epub-3.4.md.
#[cfg(feature = "epub34")]
fn encode_jxl(dynimg: &DynamicImage, quality: u8) -> Result<Vec<u8>> {
    use rgb::FromSlice;
    let cfg = zenjxl::LossyConfig::new(calibrated_jxl_distance(quality));
    let encoded = if dynimg.color().has_alpha() {
        let rgba = dynimg.to_rgba8();
        let (w, h) = (rgba.width() as usize, rgba.height() as usize);
        let img = imgref::Img::new(rgba.as_raw().as_rgba(), w, h);
        zenjxl::encode_rgba8(img, &cfg)?
    } else {
        let rgb = dynimg.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let img = imgref::Img::new(rgb.as_raw().as_rgb(), w, h);
        zenjxl::encode_rgb8(img, &cfg)?
    };
    Ok(encoded)
}

/// The standard JPEG Annex-K luminance quantization table (quality 50 baseline).
#[rustfmt::skip]
const STD_LUMA_QT: [u16; 64] = [
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99,
];

/// Estimate the quality factor (1-100) a JPEG was saved at, by comparing its
/// luminance quantization table against the standard one. Returns `None` if the
/// JPEG can't be parsed. This is a heuristic — the IJG/libjpeg quality scale —
/// good enough to avoid re-encoding a low-quality source at a higher quality.
fn estimate_jpeg_quality(data: &[u8]) -> Option<u8> {
    let mut i = 2; // skip the SOI marker
    while i + 4 <= data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = data[i + 1];
        // Standalone markers (SOI/EOI/RSTn) carry no length payload.
        if marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
            i += 2;
            continue;
        }
        let seg_len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
        let end = i + 2 + seg_len;
        if seg_len < 2 || end > data.len() {
            break;
        }
        if marker == 0xDB {
            // DQT: one or more tables, each prefixed by a precision/id byte.
            let mut p = i + 4;
            while p < end {
                let precision = data[p] >> 4; // 0 = 8-bit, 1 = 16-bit
                let table_id = data[p] & 0x0F;
                p += 1;
                let n = if precision == 1 { 128 } else { 64 };
                if p + n > end {
                    break;
                }
                if precision == 0 && table_id == 0 {
                    let table = &data[p..p + 64];
                    let mut sum = 0.0f64;
                    let mut count = 0;
                    for (k, &q) in table.iter().enumerate() {
                        let q = q as u16;
                        if q == 0 || q >= 255 {
                            continue; // saturated entries carry no quality signal
                        }
                        sum += q as f64 * 100.0 / STD_LUMA_QT[k] as f64;
                        count += 1;
                    }
                    if count == 0 {
                        return None;
                    }
                    let s = sum / count as f64;
                    let quality = if s < 100.0 {
                        (200.0 - s) / 2.0
                    } else {
                        5000.0 / s
                    };
                    return Some(quality.round().clamp(1.0, 100.0) as u8);
                }
                p += n;
            }
        }
        if marker == 0xDA {
            break; // start of scan — no more headers
        }
        i = end;
    }
    None
}

/// Scan XHTML/HTML/CSS/SVG/NCX files and update references to converted images.
pub fn update_document_references(
    root: &Path,
    ref_pairs: &[(String, String)],
    progress: &dyn Fn(&str),
) {
    if ref_pairs.is_empty() {
        return;
    }

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let is_doc = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| {
                matches!(
                    e.to_ascii_lowercase().as_str(),
                    "xhtml" | "html" | "htm" | "css" | "svg" | "ncx"
                )
            })
            .unwrap_or(false);
        if !is_doc {
            continue;
        }

        // Work on raw bytes: decoding via `from_utf8_lossy` and writing the
        // result back would silently corrupt non-UTF-8 (e.g. Latin-1) files.
        let mut content = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                progress(&format!(
                    "  [!] Warning: Failed to update references in {}: {}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    e
                ));
                continue;
            }
        };
        let mut updated = false;

        for (old_href, new_href) in ref_pairs {
            let old_name = basename(old_href);
            let new_name = basename(new_href);
            let old_encoded = quote(old_href);
            let new_encoded = quote(new_href);

            // Multi-segment hrefs (and their percent-encoded forms) are
            // replaced exactly. Anything without a directory component is a
            // bare filename and is only replaced at token boundaries, so an
            // `a.png` rename never rewrites the tail of `data.png` or mangles
            // prose/ids/CSS that merely contain the filename as a substring.
            for (old, new) in [
                (old_href.as_str(), new_href.as_str()),
                (old_encoded.as_str(), new_encoded.as_str()),
                (old_name, new_name),
            ] {
                if old.contains('/') {
                    updated |= replace_bytes(&mut content, old.as_bytes(), new.as_bytes());
                } else {
                    updated |= replace_bytes_bounded(&mut content, old.as_bytes(), new.as_bytes());
                }
            }
        }

        if updated && let Err(e) = fs::write(path, &content) {
            progress(&format!(
                "  [!] Warning: Failed to update references in {}: {}",
                path.file_name().unwrap_or_default().to_string_lossy(),
                e
            ));
        }
    }
}

/// Bytes that can appear inside a filename token. A bare-basename match is
/// only replaced when the bytes on both sides are *not* token bytes.
fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'%' | b'-')
}

/// Replace every occurrence of `old` with `new` in `content`, returning true
/// if anything changed.
fn replace_bytes(content: &mut Vec<u8>, old: &[u8], new: &[u8]) -> bool {
    replace_bytes_impl(content, old, new, false)
}

/// Like [`replace_bytes`], but only replaces occurrences delimited by
/// non-token bytes (or the start/end of the content) on both sides — see
/// [`is_token_byte`].
fn replace_bytes_bounded(content: &mut Vec<u8>, old: &[u8], new: &[u8]) -> bool {
    replace_bytes_impl(content, old, new, true)
}

fn replace_bytes_impl(content: &mut Vec<u8>, old: &[u8], new: &[u8], bounded: bool) -> bool {
    if old.is_empty() || old == new || content.len() < old.len() {
        return false;
    }
    let mut out: Vec<u8> = Vec::with_capacity(content.len());
    let mut i = 0;
    let mut changed = false;
    while i < content.len() {
        if i + old.len() <= content.len() && &content[i..i + old.len()] == old {
            let before_ok = !bounded || i == 0 || !is_token_byte(content[i - 1]);
            let after_ok = !bounded
                || i + old.len() == content.len()
                || !is_token_byte(content[i + old.len()]);
            if before_ok && after_ok {
                out.extend_from_slice(new);
                i += old.len();
                changed = true;
                continue;
            }
        }
        out.push(content[i]);
        i += 1;
    }
    if changed {
        *content = out;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    /// A detailed (gradient) JPEG encoded at a known IJG quality factor.
    fn jpeg_at(quality: u8) -> Vec<u8> {
        let img = RgbImage::from_fn(64, 64, |x, y| Rgb([(x * 4) as u8, (y * 4) as u8, 128]));
        let mut buf = Cursor::new(Vec::new());
        let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
        DynamicImage::ImageRgb8(img)
            .write_with_encoder(enc)
            .unwrap();
        buf.into_inner()
    }

    #[test]
    fn estimates_jpeg_quality_within_tolerance() {
        // The estimate is a heuristic (encoders differ slightly from the IJG
        // scale), so allow some slack. What matters for capping is that it's in
        // the right neighbourhood and ordered by quality.
        let mut prev = 0u8;
        for q in [40u8, 60, 75, 90] {
            let est = estimate_jpeg_quality(&jpeg_at(q)).expect("should parse a real JPEG");
            assert!(
                (est as i32 - q as i32).abs() <= 10,
                "quality {q} estimated as {est}"
            );
            assert!(
                est >= prev,
                "estimate should rise with quality (q{q} -> {est})"
            );
            prev = est;
        }
    }

    #[test]
    fn estimate_returns_none_for_non_jpeg() {
        assert!(estimate_jpeg_quality(b"this is not a jpeg at all").is_none());
    }

    #[test]
    fn disambiguates_colliding_output_hrefs() {
        let dir = tempfile::tempdir().unwrap();
        let mut produced = HashSet::new();

        // First writer of img/foo.webp keeps the plain name.
        let first = disambiguate_href(dir.path(), "img/foo.webp", &produced);
        assert_eq!(first, "img/foo.webp");
        produced.insert(first);

        // A second image with the same stem (foo.png after foo.jpg) must not
        // clobber it, and the numbered name goes before the extension.
        let second = disambiguate_href(dir.path(), "img/foo.webp", &produced);
        assert_eq!(second, "img/foo_2.webp");
        produced.insert(second);
        let third = disambiguate_href(dir.path(), "img/foo.webp", &produced);
        assert_eq!(third, "img/foo_3.webp");

        // A pre-existing file on disk (shipped in the source EPUB) also forces
        // a new name, even when nothing was produced this run.
        fs::create_dir_all(dir.path().join("img")).unwrap();
        fs::write(dir.path().join("img/bar.webp"), b"old").unwrap();
        let bumped = disambiguate_href(dir.path(), "img/bar.webp", &HashSet::new());
        assert_eq!(bumped, "img/bar_2.webp");
    }

    #[test]
    fn bounded_replace_leaves_longer_filenames_alone() {
        let dir = tempfile::tempdir().unwrap();
        let doc = dir.path().join("ch1.xhtml");
        fs::write(
            &doc,
            "<img src=\"a.png\"/><img src=\"data.png\"/><p>see a.png and data.png here</p>\
             <p id=\"nota.pngid\">a.png.bak stays</p>",
        )
        .unwrap();

        let pairs = vec![("a.png".to_string(), "a.webp".to_string())];
        update_document_references(dir.path(), &pairs, &|_| {});

        let out = fs::read_to_string(&doc).unwrap();
        assert_eq!(
            out,
            "<img src=\"a.webp\"/><img src=\"data.png\"/><p>see a.webp and data.png here</p>\
             <p id=\"nota.pngid\">a.png.bak stays</p>"
        );
    }

    #[test]
    fn non_utf8_document_passes_through_byte_identical() {
        let dir = tempfile::tempdir().unwrap();
        let doc = dir.path().join("ch1.xhtml");
        // 0xE9 is Latin-1 'é' — invalid UTF-8, must survive untouched.
        let mut raw = b"<p>caf".to_vec();
        raw.push(0xE9);
        raw.extend_from_slice(b" <img src=\"pic.png\"/></p>");
        fs::write(&doc, &raw).unwrap();

        let pairs = vec![("pic.png".to_string(), "pic.webp".to_string())];
        update_document_references(dir.path(), &pairs, &|_| {});

        let mut expected = b"<p>caf".to_vec();
        expected.push(0xE9);
        expected.extend_from_slice(b" <img src=\"pic.webp\"/></p>");
        assert_eq!(fs::read(&doc).unwrap(), expected);

        // A file with no matching reference is not rewritten at all.
        let untouched = dir.path().join("other.xhtml");
        let mut latin1 = b"<p>r".to_vec();
        latin1.push(0xE9);
        latin1.extend_from_slice(b"sum</p>");
        fs::write(&untouched, &latin1).unwrap();
        update_document_references(dir.path(), &pairs, &|_| {});
        assert_eq!(fs::read(&untouched).unwrap(), latin1);
    }
}
