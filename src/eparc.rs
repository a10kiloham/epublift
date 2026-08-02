// epublift - Archival mode: the `.eparc` ePub Archive format.
//
// See docs/design/eparc-format.md. Compiled under the `archival` feature.
//
// An `.eparc` shrinks a personal EPUB library and gives the book back on demand.
// It is a **stored ZIP** holding three things:
//
//   * manifest.json — the entry inventory, provenance (whole-file SHA-256) and
//     per-entry CRC32 integrity;
//   * data.zst      — every COMPRESSIBLE entry (XHTML/CSS/OPF/… and fonts like
//     OTF/TTF) concatenated in entry order and compressed as ONE solid pure-Rust
//     zstd frame (level 19);
//   * media/<path>  — every ALREADY-compressed entry (JPEG/PNG/WebP/AVIF, WOFF,
//     audio/video) stored VERBATIM (re-compressing them is wasted CPU, and the
//     archive master must never be lossier than the original).
//
// "Verbatim" is reserved for content that is already compressed; fonts and any
// unknown extension go into the solid stream, so the archive never grows a book
// (an early bug stored OTF/TTF fonts verbatim and *inflated* small books).
//
// Phase 1 restore is **content-exact**: every inner file is byte-identical and a
// valid EPUB is re-emitted (the container is re-zipped, so the whole-file bytes
// may differ from the original — that is bit-exact, a later preservation tier).
// The codec choice (solid zstd L19, media verbatim) is justified in
// docs/design/eparc-codec-choice.md.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::MAX_TOTAL_UNCOMPRESSED;

use structured_zstd::decoding::StreamingDecoder;
use structured_zstd::encoding::{CompressionLevel, compress_to_vec};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

/// Bumped only on a backwards-incompatible layout change. A restore refuses an
/// archive whose `format_version` it does not understand.
const FORMAT_VERSION: u32 = 1;
/// Solid-stream zstd level (pure-Rust practical max; see eparc-codec-choice.md).
const ZSTD_LEVEL: i32 = 19;

const MANIFEST_NAME: &str = "manifest.json";
const STREAM_BLOB_NAME: &str = "data.zst";
const MEDIA_PREFIX: &str = "media/";

/// Cap on any `with_capacity` pre-reservation derived from an untrusted declared
/// size (ZIP headers and the manifest are attacker-controlled; a huge claim must
/// not turn into a huge allocation — the `Vec` grows on its own if the entry
/// really is bigger).
const MAX_PREALLOC: usize = 8 * 1024 * 1024; // 8 MiB

/// Cap on `manifest.json` itself (untrusted; a real manifest is a few KiB — a
/// multi-GiB claim is an attack, not a book).
const MAX_MANIFEST_SIZE: u64 = 64 * 1024 * 1024; // 64 MiB

/// A safe `with_capacity` hint from an untrusted declared size.
fn prealloc_hint(declared: u64) -> usize {
    declared.min(MAX_PREALLOC as u64) as usize
}

/// The `.eparc` manifest (serialized as human-readable `manifest.json`).
#[derive(Serialize, Deserialize)]
struct Manifest {
    format: String,
    format_version: u32,
    tool: String,
    created: String,
    source: Source,
    #[serde(skip_serializing_if = "Option::is_none")]
    epub_version: Option<String>,
    codec: Codec,
    entries: Vec<Entry>,
}

#[derive(Serialize, Deserialize)]
struct Source {
    filename: String,
    size: u64,
    /// Whole original `.epub` SHA-256 (provenance / fixity).
    sha256: String,
}

#[derive(Serialize, Deserialize)]
struct Codec {
    name: String,
    #[serde(rename = "impl")]
    implementation: String,
    level: i32,
    mode: String,
}

/// Where an entry's bytes live in the archive.
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Store {
    /// Part of the solid `data.zst` blob (sliced back out by `size`, in order).
    Stream,
    /// A verbatim file under `media/` (already-compressed content).
    Verbatim,
}

#[derive(Serialize, Deserialize)]
struct Entry {
    /// Original ZIP path inside the EPUB (also the restore order).
    path: String,
    store: Store,
    /// Uncompressed byte length (splits the stream blob; sanity-checks media).
    size: u64,
    /// CRC32 of the uncompressed bytes, hex (cheap corruption check).
    crc32: String,
    /// Reserved for Phase 2 (lossless re-pack / transcode). `"original"` for
    /// verbatim media in Phase 1; absent for streamed entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    media_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_format: Option<String>,
}

/// Outcome of [`archive_epub`].
pub struct ArchiveStats {
    pub original_size: u64,
    pub archive_size: u64,
    /// Entries folded into the compressed solid stream (text, fonts, …).
    pub compressed_entries: usize,
    /// Entries stored verbatim (already-compressed media).
    pub stored_entries: usize,
}

impl ArchiveStats {
    /// Percentage of the original `.epub` size that was saved (negative if the
    /// archive is larger — which should not happen for a real book).
    pub fn percent_saved(&self) -> f64 {
        if self.original_size > 0 {
            (1.0 - self.archive_size as f64 / self.original_size as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Outcome of [`restore_eparc`].
pub struct RestoreStats {
    pub output_size: u64,
    pub entries: usize,
}

/// Archive a single `.epub` into a `.eparc` at `output`.
///
/// The original file is never modified. Compressible entries (text + fonts) are
/// solid-zstd'd together; already-compressed media is stored verbatim; a manifest
/// records order, sizes, per-entry CRC32 and the whole-file SHA-256.
pub fn archive_epub(input: &Path, output: &Path) -> Result<ArchiveStats> {
    let original = std::fs::read(input)
        .with_context(|| format!("Failed to read EPUB: {}", input.display()))?;
    let source = Source {
        filename: file_name(input),
        size: original.len() as u64,
        sha256: sha256_hex(&original),
    };

    let entries = read_epub_entries(&original)
        .with_context(|| format!("Not a readable EPUB (zip): {}", input.display()))?;
    let epub_version = detect_epub_version(&entries);

    // Partition into the solid stream blob + verbatim media, building the
    // manifest in original entry order.
    let mut stream_blob: Vec<u8> = Vec::new();
    let mut media: Vec<(String, Vec<u8>)> = Vec::new();
    let mut manifest_entries: Vec<Entry> = Vec::with_capacity(entries.len());

    for (name, data) in &entries {
        let crc32 = crc32_hex(data);
        if store_verbatim(name) {
            media.push((format!("{MEDIA_PREFIX}{name}"), data.clone()));
            manifest_entries.push(Entry {
                path: name.clone(),
                store: Store::Verbatim,
                size: data.len() as u64,
                crc32,
                media_format: Some("original".to_string()),
                source_format: Some(extension_of(name)),
            });
        } else {
            stream_blob.extend_from_slice(data);
            manifest_entries.push(Entry {
                path: name.clone(),
                store: Store::Stream,
                size: data.len() as u64,
                crc32,
                media_format: None,
                source_format: None,
            });
        }
    }
    let compressed_entries = manifest_entries
        .iter()
        .filter(|e| e.store == Store::Stream)
        .count();

    let data_zst = compress_to_vec(stream_blob.as_slice(), CompressionLevel::Level(ZSTD_LEVEL));

    let manifest = Manifest {
        format: "eparc".to_string(),
        format_version: FORMAT_VERSION,
        tool: format!("epublift {}", env!("CARGO_PKG_VERSION")),
        created: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        source,
        epub_version,
        codec: Codec {
            name: "zstd".to_string(),
            implementation: "structured-zstd".to_string(),
            level: ZSTD_LEVEL,
            mode: "solid".to_string(),
        },
        entries: manifest_entries,
    };
    let manifest_json = serde_json::to_vec_pretty(&manifest)?;

    // Write the stored-ZIP container: manifest.json, data.zst, then media/*.
    let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    let file = File::create(output)
        .with_context(|| format!("Failed to create archive: {}", output.display()))?;
    let mut zip = ZipWriter::new(file);
    zip.start_file(MANIFEST_NAME, stored)?;
    zip.write_all(&manifest_json)?;
    zip.start_file(STREAM_BLOB_NAME, stored)?;
    zip.write_all(&data_zst)?;
    for (media_path, data) in &media {
        zip.start_file(media_path, stored)?;
        zip.write_all(data)?;
    }
    zip.finish()?;

    let archive_size = std::fs::metadata(output)?.len();
    Ok(ArchiveStats {
        original_size: original.len() as u64,
        archive_size,
        compressed_entries,
        stored_entries: media.len(),
    })
}

/// Restore a `.eparc` back to a content-exact `.epub` at `output`.
///
/// Decompresses the solid stream, splits it by the manifest sizes, pulls media
/// verbatim, verifies every entry's CRC32, and re-emits a valid EPUB (mimetype
/// stored first, the rest deflated) in the original entry order.
pub fn restore_eparc(input: &Path, output: &Path) -> Result<RestoreStats> {
    let file = File::open(input)
        .with_context(|| format!("Failed to open archive: {}", input.display()))?;
    let mut zip = ZipArchive::new(file)
        .with_context(|| format!("Not a readable .eparc (zip): {}", input.display()))?;

    let manifest: Manifest = {
        // The manifest's declared size is untrusted: read at most the cap + 1
        // bytes and bail cleanly instead of buffering a multi-GiB entry.
        let mut s = String::new();
        zip.by_name(MANIFEST_NAME)
            .context("archive is missing manifest.json")?
            .take(MAX_MANIFEST_SIZE + 1)
            .read_to_string(&mut s)?;
        if s.len() as u64 > MAX_MANIFEST_SIZE {
            bail!("corrupt archive: manifest.json exceeds {MAX_MANIFEST_SIZE} bytes");
        }
        serde_json::from_str(&s).context("manifest.json is not valid")?
    };
    if manifest.format != "eparc" {
        bail!("not an .eparc archive (format = {:?})", manifest.format);
    }
    if manifest.format_version > FORMAT_VERSION {
        bail!(
            "archive is format v{}, but this build understands only up to v{} — please update epublift",
            manifest.format_version,
            FORMAT_VERSION
        );
    }

    // Decompress the solid stream blob once. The manifest sizes are untrusted,
    // so sum them with checked arithmetic, bound the total, and never let a
    // declared size drive the allocation.
    let mut stream_total: u64 = 0;
    for e in manifest.entries.iter().filter(|e| e.store == Store::Stream) {
        stream_total = stream_total
            .checked_add(e.size)
            .ok_or_else(|| anyhow::anyhow!("corrupt archive: manifest entry sizes overflow"))?;
    }
    if stream_total > MAX_TOTAL_UNCOMPRESSED {
        bail!(
            "corrupt archive: manifest declares {stream_total} stream bytes (limit {MAX_TOTAL_UNCOMPRESSED})"
        );
    }
    // Total-uncompressed budget shared across every entry read, as in
    // `extract_epub`: enforced against bytes actually read, since the header
    // sizes are attacker-controlled.
    let mut budget: u64 = MAX_TOTAL_UNCOMPRESSED;
    let data_zst = read_zip_entry(&mut zip, STREAM_BLOB_NAME, &mut budget)?;
    // Decode incrementally with a hard output cap: the compressed frame is
    // attacker-controlled, so it must never expand past the validated total
    // (a zstd bomb would otherwise balloon in memory BEFORE the length check).
    let limit = stream_total.min(MAX_TOTAL_UNCOMPRESSED);
    let decoder = StreamingDecoder::new(data_zst.as_slice())
        .map_err(|e| anyhow::anyhow!("failed to decompress {STREAM_BLOB_NAME}: {e:?}"))?;
    let mut stream_buf = Vec::with_capacity(prealloc_hint(stream_total));
    // Read at most `limit + 1` bytes; overshooting means a bomb.
    let read = decoder
        .take(limit + 1)
        .read_to_end(&mut stream_buf)
        .map_err(|e| anyhow::anyhow!("failed to decompress {STREAM_BLOB_NAME}: {e:?}"))?
        as u64;
    if read > limit {
        bail!(
            "corrupt archive: {STREAM_BLOB_NAME} expands past the decompression limit (possible zstd bomb)"
        );
    }
    if stream_buf.len() as u64 != stream_total {
        bail!(
            "stream size mismatch (expected {stream_total}, got {}) — archive corrupt",
            stream_buf.len()
        );
    }

    // Re-emit the EPUB in manifest order.
    let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    let deflated = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let out = File::create(output)
        .with_context(|| format!("Failed to create output EPUB: {}", output.display()))?;
    let mut writer = ZipWriter::new(out);

    let mut stream_off = 0usize;
    for entry in &manifest.entries {
        let data: Vec<u8> = match entry.store {
            Store::Stream => {
                // `size` is untrusted; check it against what actually remains in
                // the decoded stream instead of slicing blind (which would panic).
                let remaining = stream_buf.len() - stream_off;
                if entry.size > remaining as u64 {
                    bail!(
                        "corrupt archive: entry '{}' claims {} stream bytes but only {remaining} remain",
                        entry.path,
                        entry.size
                    );
                }
                let end = stream_off + entry.size as usize;
                let chunk = stream_buf[stream_off..end].to_vec();
                stream_off = end;
                chunk
            }
            Store::Verbatim => read_zip_entry(
                &mut zip,
                &format!("{MEDIA_PREFIX}{}", entry.path),
                &mut budget,
            )?,
        };
        let got = crc32_hex(&data);
        if got != entry.crc32 {
            bail!(
                "integrity check failed for '{}' (crc32 {} != {})",
                entry.path,
                got,
                entry.crc32
            );
        }
        let opts = if entry.path == "mimetype" {
            stored
        } else {
            deflated
        };
        writer.start_file(&entry.path, opts)?;
        writer.write_all(&data)?;
    }
    writer.finish()?;

    let output_size = std::fs::metadata(output)?.len();
    Ok(RestoreStats {
        output_size,
        entries: manifest.entries.len(),
    })
}

// ---- helpers --------------------------------------------------------------

/// True for entries stored **verbatim**: the `mimetype`, plus content that is
/// already compressed (images, web fonts, audio, video, archives). Everything
/// else — text, OTF/TTF fonts, unknown extensions — goes into the solid stream,
/// which never meaningfully grows even incompressible input. Classification is by
/// extension so archive and restore agree without a side channel.
fn store_verbatim(name: &str) -> bool {
    if name == "mimetype" {
        return true;
    }
    let ext = extension_of(name);
    matches!(
        ext.as_str(),
        // raster images
        "jpg" | "jpeg" | "png" | "gif" | "webp" | "avif" | "jxl" | "jp2" | "j2k"
        | "jpf" | "jpx" | "heic" | "heif" | "tif" | "tiff"
        // already-compressed web fonts (WOFF = zlib, WOFF2 = brotli)
        | "woff" | "woff2"
        // audio / video
        | "mp3" | "m4a" | "aac" | "ogg" | "oga" | "opus" | "flac"
        | "mp4" | "m4v" | "webm" | "mov" | "mkv" | "avi"
        // archives / already-compressed blobs
        | "zip" | "gz" | "zst" | "br" | "xz" | "7z" | "rar"
    )
}

/// Read every file entry of an in-memory EPUB zip, in archive order, as
/// `(name, uncompressed-bytes)`.
fn read_epub_entries(epub: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    // ZIPs may legally repeat a name, but this format cannot represent that:
    // verbatim entries become colliding `media/{name}` paths and restore
    // resolves by name, so one duplicate's bytes would be silently lost and
    // the archived book could never restore content-exact. Refuse up front
    // rather than archive unrecoverably. `ZipArchive` itself collapses
    // duplicates (last one wins), so scan the raw central directory too.
    if let Some(name) = duplicate_entry_name(epub) {
        bail!("this EPUB has duplicate entry names ('{name}') and cannot be archived safely");
    }
    let mut zip = ZipArchive::new(std::io::Cursor::new(epub))?;
    let mut out = Vec::with_capacity(zip.len());
    // Total-uncompressed budget, matching `extract_epub` in lib.rs: enforced
    // against bytes actually read, since header sizes are attacker-controlled.
    let mut budget: u64 = MAX_TOTAL_UNCOMPRESSED;
    // Belt and braces: also refuse duplicates as read, in case a future zip
    // crate starts preserving them through `by_index`.
    let mut seen: HashSet<String> = HashSet::with_capacity(zip.len());
    for i in 0..zip.len() {
        let mut f = zip.by_index(i)?;
        if !f.is_file() {
            continue;
        }
        let name = f.name().to_string();
        if !seen.insert(name.clone()) {
            bail!("this EPUB has duplicate entry names ('{name}') and cannot be archived safely");
        }
        let mut data = Vec::with_capacity(prealloc_hint(f.size()));
        // Read at most `budget + 1` bytes; overshooting means a bomb.
        let read = f.by_ref().take(budget + 1).read_to_end(&mut data)? as u64;
        if read > budget {
            bail!("EPUB is too large when decompressed (possible zip bomb).");
        }
        budget -= read;
        out.push((name, data));
    }
    Ok(out)
}

/// Best-effort scan of a ZIP's central directory for a repeated entry name.
///
/// Needed because `ZipArchive` silently collapses duplicate names (the last
/// one wins), so the indexed read alone can never see them. Walks the plain
/// (non-zip64) end-of-central-directory record; that covers every EPUB this
/// tool accepts, since `MAX_TOTAL_UNCOMPRESSED` rejects zip64-sized books
/// long before this matters. Returns `None` when the directory can't be
/// walked — the untrusted bytes then still face the zip crate's own parser.
fn duplicate_entry_name(epub: &[u8]) -> Option<String> {
    const EOCD_SIG: &[u8; 4] = b"PK\x05\x06"; // end of central directory
    const CDH_SIG: &[u8; 4] = b"PK\x01\x02"; // central directory header
    let eocd = epub.windows(4).rposition(|w| w == EOCD_SIG)?;
    let rec = epub.get(eocd..eocd + 22)?;
    let count = u16::from_le_bytes([rec[10], rec[11]]) as usize;
    let cd_size = u32::from_le_bytes([rec[12], rec[13], rec[14], rec[15]]) as usize;
    // Derive the directory start from the EOCD position rather than trusting
    // the stored offset (this also tolerates data prepended to the ZIP).
    let mut pos = eocd.checked_sub(cd_size)?;
    let mut seen: HashSet<&[u8]> = HashSet::with_capacity(count);
    for _ in 0..count {
        let hdr = epub.get(pos..pos + 46)?;
        if &hdr[..4] != CDH_SIG {
            return None;
        }
        let name_len = u16::from_le_bytes([hdr[28], hdr[29]]) as usize;
        let extra_len = u16::from_le_bytes([hdr[30], hdr[31]]) as usize;
        let comment_len = u16::from_le_bytes([hdr[32], hdr[33]]) as usize;
        let name = epub.get(pos + 46..pos + 46 + name_len)?;
        if !seen.insert(name) {
            return Some(String::from_utf8_lossy(name).into_owned());
        }
        pos = pos + 46 + name_len + extra_len + comment_len;
    }
    None
}

/// Read one named entry of an open archive fully into memory, drawing from a
/// caller-held total-uncompressed `budget` (shared across calls so a crafted
/// archive cannot expand past the cap; the header's declared size is untrusted).
fn read_zip_entry<R: Read + std::io::Seek>(
    zip: &mut ZipArchive<R>,
    name: &str,
    budget: &mut u64,
) -> Result<Vec<u8>> {
    let mut e = zip
        .by_name(name)
        .with_context(|| format!("archive is missing '{name}'"))?;
    let mut data = Vec::with_capacity(prealloc_hint(e.size()));
    // Read at most `budget + 1` bytes; overshooting means a bomb.
    let read = e.by_ref().take(*budget + 1).read_to_end(&mut data)? as u64;
    if read > *budget {
        bail!("corrupt archive: entries expand past the decompression limit (possible zip bomb)");
    }
    *budget -= read;
    Ok(data)
}

/// Best-effort EPUB version: parse `<package version="…">` from the first `.opf`.
/// Purely informational in the manifest; `None` if it can't be determined.
fn detect_epub_version(entries: &[(String, Vec<u8>)]) -> Option<String> {
    let (_, opf) = entries.iter().find(|(n, _)| n.ends_with(".opf"))?;
    let xml = std::str::from_utf8(opf).ok()?;
    // Bound nesting before the recursive parse (stack-overflow guard).
    if !crate::util::xml_nesting_ok(xml, 256) {
        return None;
    }
    let doc = roxmltree::Document::parse(xml).ok()?;
    doc.descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "package")
        .and_then(|n| n.attribute("version"))
        .map(|v| v.to_string())
}

fn crc32_hex(data: &[u8]) -> String {
    format!("{:08x}", crc32fast::hash(data))
}

fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

fn extension_of(name: &str) -> String {
    name.rsplit('.').next().unwrap_or("").to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zip::write::SimpleFileOptions;

    /// Build a minimal but realistic EPUB (mimetype + container + OPF + a chapter
    /// + a verbatim binary "image") and return its bytes.
    fn sample_epub() -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut zip = ZipWriter::new(std::io::Cursor::new(&mut buf));
            let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
            let deflated =
                SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);

            zip.start_file("mimetype", stored).unwrap();
            zip.write_all(b"application/epub+zip").unwrap();

            zip.start_file("META-INF/container.xml", deflated).unwrap();
            zip.write_all(
                br#"<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"#,
            )
            .unwrap();

            zip.start_file("OEBPS/content.opf", deflated).unwrap();
            zip.write_all(
                br#"<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id"><metadata/><manifest><item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/><item id="img" href="img/cover.png" media-type="image/png"/></manifest><spine><itemref idref="c1"/></spine></package>"#,
            )
            .unwrap();

            // Repeated text so the solid stream actually has redundancy to find.
            let chapter = format!(
                "<?xml version=\"1.0\"?><!DOCTYPE html><html xmlns=\"http://www.w3.org/1999/xhtml\"><head><title>Ch 1</title></head><body>{}</body></html>",
                "<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>".repeat(50)
            );
            zip.start_file("OEBPS/ch1.xhtml", deflated).unwrap();
            zip.write_all(chapter.as_bytes()).unwrap();

            // A "media" blob — arbitrary incompressible-ish bytes, stored verbatim.
            zip.start_file("OEBPS/img/cover.png", stored).unwrap();
            let img: Vec<u8> = (0u32..4096)
                .map(|i| (i.wrapping_mul(2654435761)) as u8)
                .collect();
            zip.write_all(&img).unwrap();

            // A font (OTF) — NOT already compressed, so it must go into the
            // stream (storing it verbatim would inflate the archive).
            zip.start_file("OEBPS/fonts/body.otf", deflated).unwrap();
            zip.write_all(&b"OTTO".repeat(2000)).unwrap();

            zip.finish().unwrap();
        }
        buf
    }

    #[test]
    fn archive_then_restore_is_content_exact() {
        let dir = tempfile::tempdir().unwrap();
        let epub_path = dir.path().join("book.epub");
        std::fs::write(&epub_path, sample_epub()).unwrap();
        let eparc_path = dir.path().join("book.eparc");
        let restored_path = dir.path().join("restored.epub");

        let stats = archive_epub(&epub_path, &eparc_path).unwrap();
        // text (opf/container/chapter) + the OTF font are compressed; mimetype and
        // the png are stored verbatim.
        assert!(stats.compressed_entries >= 4, "text + font are compressed");
        assert_eq!(stats.stored_entries, 2, "mimetype + cover.png are verbatim");

        restore_eparc(&eparc_path, &restored_path).unwrap();

        // Content-exact: every original entry's bytes are reproduced exactly.
        let original = read_epub_entries(&std::fs::read(&epub_path).unwrap()).unwrap();
        let restored = read_epub_entries(&std::fs::read(&restored_path).unwrap()).unwrap();
        assert_eq!(
            original, restored,
            "restored EPUB must be byte-identical to the original, entry for entry"
        );
    }

    #[test]
    fn restore_rejects_absurd_manifest_sizes() {
        // A manifest declaring a 1 TiB streamed entry must fail cleanly — no
        // giant allocation, no panic, just a "corrupt archive" error.
        let dir = tempfile::tempdir().unwrap();
        let epub_path = dir.path().join("book.epub");
        std::fs::write(&epub_path, sample_epub()).unwrap();
        let eparc_path = dir.path().join("book.eparc");
        archive_epub(&epub_path, &eparc_path).unwrap();

        // Re-write the archive with the poisoned size in its manifest.
        let bytes = std::fs::read(&eparc_path).unwrap();
        let mut zip = ZipArchive::new(std::io::Cursor::new(bytes.as_slice())).unwrap();
        let mut manifest: Manifest = {
            let mut s = String::new();
            zip.by_name(MANIFEST_NAME)
                .unwrap()
                .read_to_string(&mut s)
                .unwrap();
            serde_json::from_str(&s).unwrap()
        };
        let streamed = manifest
            .entries
            .iter_mut()
            .find(|e| e.store == Store::Stream)
            .unwrap();
        streamed.size = 1 << 40; // 1 TiB
        let evil_path = dir.path().join("evil.eparc");
        {
            let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
            let mut w = ZipWriter::new(File::create(&evil_path).unwrap());
            for i in 0..zip.len() {
                let mut f = zip.by_index(i).unwrap();
                let name = f.name().to_string();
                w.start_file(&name, stored).unwrap();
                if name == MANIFEST_NAME {
                    w.write_all(&serde_json::to_vec_pretty(&manifest).unwrap())
                        .unwrap();
                } else {
                    let mut data = Vec::new();
                    f.read_to_end(&mut data).unwrap();
                    w.write_all(&data).unwrap();
                }
            }
            w.finish().unwrap();
        }

        let err = match restore_eparc(&evil_path, &dir.path().join("out.epub")) {
            Ok(_) => panic!("restore must reject the poisoned manifest"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("corrupt archive"),
            "expected a clean corrupt-archive error, got: {err}"
        );
    }

    #[test]
    fn archive_rejects_duplicate_entry_names() {
        // ZipWriter refuses duplicate names, so write two same-length names and
        // byte-patch the second into the first (names live uncompressed in the
        // local + central headers; stored contents avoid the needle). Real
        // tools do emit such EPUBs, and archiving one would silently lose the
        // shadowed entry's bytes — it must be refused.
        let mut buf = Vec::new();
        {
            let mut zip = ZipWriter::new(std::io::Cursor::new(&mut buf));
            let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
            zip.start_file("mimetype", stored).unwrap();
            zip.write_all(b"application/epub+zip").unwrap();
            zip.start_file("OEBPS/ch1.xhtml", stored).unwrap();
            zip.write_all(b"<html>first</html>").unwrap();
            zip.start_file("OEBPS/ch9.xhtml", stored).unwrap();
            zip.write_all(b"<html>second, different bytes</html>")
                .unwrap();
            zip.finish().unwrap();
        }
        let needle = b"ch9.xhtml";
        for i in 0..buf.len() - needle.len() {
            if &buf[i..i + needle.len()] == needle {
                buf[i..i + needle.len()].copy_from_slice(b"ch1.xhtml");
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let epub_path = dir.path().join("dupes.epub");
        std::fs::write(&epub_path, buf).unwrap();

        let err = match archive_epub(&epub_path, &dir.path().join("dupes.eparc")) {
            Ok(_) => panic!("archiving an EPUB with duplicate entry names must fail"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("duplicate entry names") && msg.contains("OEBPS/ch1.xhtml"),
            "expected a clear duplicate-name error, got: {msg}"
        );
    }

    /// Manual round-trip on a REAL EPUB (nested dirs, real entry names, fonts,
    /// images). Ignored by default; run with a real file:
    ///   EPARC_TEST_EPUB="/path/book.epub" cargo test --features archival \
    ///     --lib eparc -- --ignored --nocapture
    #[test]
    #[ignore = "needs a real EPUB via EPARC_TEST_EPUB"]
    fn roundtrip_real_epub_from_env() {
        let Ok(src) = std::env::var("EPARC_TEST_EPUB") else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let eparc = dir.path().join("book.eparc");
        let restored = dir.path().join("restored.epub");

        let stats = archive_epub(Path::new(&src), &eparc).unwrap();
        println!(
            "archived {} compressed + {} stored: {} -> {} bytes ({:.1}% saved)",
            stats.compressed_entries,
            stats.stored_entries,
            stats.original_size,
            stats.archive_size,
            stats.percent_saved(),
        );
        restore_eparc(&eparc, &restored).unwrap();

        let original = read_epub_entries(&std::fs::read(&src).unwrap()).unwrap();
        let back = read_epub_entries(&std::fs::read(&restored).unwrap()).unwrap();
        assert_eq!(original, back, "real EPUB must round-trip content-exact");
        println!("OK: {} entries round-tripped content-exact", back.len());
    }

    #[test]
    fn classification_keeps_compressible_in_the_stream() {
        // Compressed → verbatim.
        assert!(store_verbatim("mimetype"));
        assert!(store_verbatim("OEBPS/img/cover.png"));
        assert!(store_verbatim("OEBPS/fonts/body.woff2"));
        // Compressible → stream.
        assert!(!store_verbatim("OEBPS/ch1.xhtml"));
        assert!(!store_verbatim("OEBPS/content.opf"));
        assert!(!store_verbatim("OEBPS/fonts/body.otf"));
        assert!(!store_verbatim("OEBPS/fonts/body.ttf"));
    }
}
