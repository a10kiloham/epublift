//! Small shared helpers: URL (percent) encoding compatible with Python's
//! `urllib.parse.quote`/`unquote`, href manipulation, and XHTML DOCTYPE
//! standardization.

use any_ascii::any_ascii;
use anyhow::Result;
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, percent_decode_str, utf8_percent_encode};
use regex::Regex;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Characters that Python's `urllib.parse.quote` leaves untouched by default:
/// unreserved characters (`A-Z a-z 0-9 _ . - ~`) plus the `/` path separator
/// (because `quote` uses `safe='/'` by default).
const QUOTE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'_')
    .remove(b'.')
    .remove(b'-')
    .remove(b'~')
    .remove(b'/');

/// Percent-encode a string the way Python's `urllib.parse.quote` does.
pub fn quote(s: &str) -> String {
    utf8_percent_encode(s, QUOTE_SET).to_string()
}

/// Percent-decode a string the way Python's `urllib.parse.unquote` does.
pub fn unquote(s: &str) -> String {
    percent_decode_str(s).decode_utf8_lossy().into_owned()
}

/// True when a container-internal path (an OPF manifest href, the
/// container.xml `full-path`, …) stays inside the EPUB: non-empty, relative,
/// and made of plain components only — no `..`, no root, no drive prefix.
/// Joining an unchecked value onto the extraction dir would let a crafted
/// book read, overwrite, or delete files outside it.
pub fn is_contained_path(p: &str) -> bool {
    use std::path::Component;
    !p.is_empty()
        && Path::new(p)
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
}

/// True when an archive entry's contents are **text/markup** (XHTML/CSS/OPF/…)
/// rather than already-compressed media (images/fonts) or the `mimetype`.
///
/// Classification is purely by extension so independent code paths (the archival
/// packer and the experimental Zstd-OCF packer) agree without a side manifest.
/// `mimetype` is excluded so it is always stored verbatim/first.
#[cfg(feature = "zstd-experimental")]
pub fn is_text_entry(name: &str) -> bool {
    if name == "mimetype" {
        return false;
    }
    let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "xhtml" | "html" | "htm" | "css" | "opf" | "ncx" | "xml" | "svg" | "txt" | "json"
    )
}

/// Return the final path component (everything after the last `/`).
pub fn basename(href: &str) -> &str {
    match href.rfind('/') {
        Some(i) => &href[i + 1..],
        None => href,
    }
}

/// Replace the file extension of an href with `.{ext}`, mirroring
/// `pathlib.Path(href).with_suffix('.{ext}')`. `ext` is given without the dot.
pub fn with_ext(href: &str, ext: &str) -> String {
    let slash = href.rfind('/').map(|i| i as isize).unwrap_or(-1);
    match href.rfind('.') {
        Some(dot) if (dot as isize) > slash => format!("{}.{ext}", &href[..dot]),
        _ => format!("{href}.{ext}"),
    }
}

/// Transliterate a filename stem to an ASCII-safe slug for the `--ascii` option.
/// Unicode letters are romanized (e.g. Turkish `Işık Doğudan` → `Isik_Dogudan`),
/// whitespace becomes underscores, and any character outside `[A-Za-z0-9._-]` is
/// dropped. Runs of underscores are collapsed and leading/trailing ones trimmed.
/// Falls back to `output` if nothing printable remains.
pub fn slugify_ascii(stem: &str) -> String {
    let romanized = any_ascii(stem);
    let mut out = String::with_capacity(romanized.len());
    let mut prev_underscore = false;
    for ch in romanized.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '-' {
            out.push(ch);
            prev_underscore = false;
        } else if (ch.is_whitespace() || ch == '_') && !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
        // anything else (punctuation, symbols) is dropped
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "output".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Standardize HTML/XHTML files to EPUB 3 best practices:
/// - Replace legacy DOCTYPE declarations with the HTML5 `<!DOCTYPE html>`.
/// - Ensure the XHTML namespace is declared on the `<html>` element.
pub fn standardize_xhtml_files(root: &Path, progress: &dyn Fn(&str)) -> Result<()> {
    // Consume an optional bracketed internal subset (`<!DOCTYPE html [ ... ]>`)
    // so entity declarations are removed wholesale, not truncated at the first `>`.
    let doctype_re = Regex::new(r"(?is)<!DOCTYPE\s+html(?:[^>\[]|\[[^\]]*\])*>").unwrap();

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let is_html = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| matches!(e.to_ascii_lowercase().as_str(), "html" | "xhtml" | "htm"))
            .unwrap_or(false);
        if !is_html {
            continue;
        }

        let raw = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                progress(&format!(
                    "  [!] Warning: Could not modernize HTML tag in {}: {}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    e
                ));
                continue;
            }
        };
        // Never run the text transforms over non-UTF-8 content: a lossy
        // conversion would replace every non-ASCII byte with U+FFFD and the
        // corrupted result would be written back. Skip such files untouched.
        let mut content = match std::str::from_utf8(&raw) {
            Ok(s) => s.to_owned(),
            Err(_) => {
                progress(&format!(
                    "  [!] Warning: {} is not valid UTF-8; leaving it untouched",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                ));
                continue;
            }
        };

        content = doctype_re
            .replace_all(&content, "<!DOCTYPE html>")
            .into_owned();

        // Recognize both quote forms of an existing declaration, and patch only
        // the first `<html` (a blind replace would also rewrite occurrences in
        // comments or text).
        if !content.contains("xmlns=\"http://www.w3.org/1999/xhtml\"")
            && !content.contains("xmlns='http://www.w3.org/1999/xhtml'")
        {
            content = content.replacen("<html", "<html xmlns=\"http://www.w3.org/1999/xhtml\"", 1);
        }

        if let Err(e) = fs::write(path, content) {
            progress(&format!(
                "  [!] Warning: Could not modernize HTML tag in {}: {}",
                path.file_name().unwrap_or_default().to_string_lossy(),
                e
            ));
        }
    }
    Ok(())
}

/// Detect content documents that use **HTML syntax** rather than XHTML, which
/// EPUB 3.4 removed. The reliable signal is an HTML *void* element that isn't
/// self-closed (valid XHTML always self-closes them: `<br/>`, `<img .../>`).
/// Returns one human-readable warning per offending file. A heuristic for the
/// lint — we report it; converting HTML→XHTML is out of scope.
pub fn detect_html_syntax(root: &Path) -> Vec<String> {
    let void_re = Regex::new(
        r"(?is)<(?:br|hr|wbr|img|input|link|meta|area|base|col|embed|source|track|param)\b[^>]*>",
    )
    .unwrap();
    let mut warnings = Vec::new();

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let is_html = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| matches!(e.to_ascii_lowercase().as_str(), "html" | "xhtml" | "htm"))
            .unwrap_or(false);
        if !is_html {
            continue;
        }
        let Ok(content) = fs::read_to_string(path) else {
            continue;
        };
        // HTML syntax = a void element tag that doesn't self-close (`...>` not `.../>`).
        let html_syntax = void_re
            .find_iter(&content)
            .any(|m| !m.as_str().trim_end().ends_with("/>"));
        if html_syntax {
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            warnings.push(format!(
                "content document `{name}` uses HTML syntax (EPUB 3.4 requires XHTML)"
            ));
        }
    }
    warnings
}

/// Cheap pre-parse guard for untrusted XML: returns `false` when element
/// nesting ever exceeds `max_depth`. Recursive parsers (e.g. roxmltree)
/// recurse once per nesting level, so a crafted document with a few hundred
/// thousand nested elements can blow the stack; run this first and refuse
/// such input. This is a heuristic byte scan, deliberately permissive for
/// odd content (comments, CDATA, `>` inside attribute values) — it only has
/// to bound genuine deep nesting, not parse exactly.
pub(crate) fn xml_nesting_ok(xml: &str, max_depth: usize) -> bool {
    let bytes = xml.as_bytes();
    let mut depth: usize = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        match bytes.get(i + 1) {
            // Closing tag.
            Some(b'/') => {
                depth = depth.saturating_sub(1);
                i += 2;
            }
            // Declarations/comments/doctype (`<!`) and processing
            // instructions (`<?`): skip to the next `>` and ignore.
            Some(b'!') | Some(b'?') => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'>' {
                    i += 1;
                }
            }
            // Opening tag: name start (letter, `_`, `:`, or non-ASCII).
            Some(&c) if c.is_ascii_alphabetic() || c == b'_' || c == b':' || !c.is_ascii() => {
                let mut j = i + 1;
                while j < bytes.len() && bytes[j] != b'>' {
                    j += 1;
                }
                // `/>` before the tag's `>` means self-closing: net zero.
                let self_closing = j < bytes.len() && bytes[j - 1] == b'/';
                if !self_closing {
                    depth += 1;
                    if depth > max_depth {
                        return false;
                    }
                }
                i = j + 1;
            }
            // Stray `<` (or end of input): not a tag.
            _ => i += 1,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{
        detect_html_syntax, is_contained_path, slugify_ascii, standardize_xhtml_files,
        xml_nesting_ok,
    };
    use std::fs;

    #[test]
    fn containment_rejects_escaping_paths() {
        assert!(is_contained_path("OEBPS/content.opf"));
        assert!(is_contained_path("./images/pic.png"));
        assert!(!is_contained_path(""));
        assert!(!is_contained_path("../outside.opf"));
        assert!(!is_contained_path("images/../../outside.png"));
        assert!(!is_contained_path("/etc/passwd"));
    }

    #[test]
    fn flags_html_syntax_but_not_xhtml() {
        let dir = tempfile::tempdir().unwrap();
        // HTML syntax: bare void elements.
        fs::write(
            dir.path().join("html.xhtml"),
            "<html><body><p>Hi<br>there<img src=\"a.jpg\"></p></body></html>",
        )
        .unwrap();
        // Valid XHTML: self-closed voids.
        fs::write(
            dir.path().join("xhtml.xhtml"),
            "<html><body><p>Hi<br/>there<img src=\"a.jpg\"/></p></body></html>",
        )
        .unwrap();
        // Non-content file: ignored.
        fs::write(dir.path().join("styles.css"), "br { color: red; }").unwrap();

        let w = detect_html_syntax(dir.path());
        assert_eq!(w.len(), 1, "exactly one offending file: {w:?}");
        assert!(w[0].contains("html.xhtml"));
    }

    #[test]
    fn slugifies_turkish_title() {
        assert_eq!(
            slugify_ascii("Işık Doğudan Yükselir"),
            "Isik_Dogudan_Yukselir"
        );
    }

    #[test]
    fn collapses_and_trims_separators() {
        assert_eq!(slugify_ascii("  A  --  B!! "), "A_--_B");
        assert_eq!(slugify_ascii("Çöl: Bir_Öykü"), "Col_Bir_Oyku");
    }

    #[test]
    fn keeps_dots_and_dashes() {
        assert_eq!(slugify_ascii("Vol.2 - Part 3"), "Vol.2_-_Part_3");
    }

    #[test]
    fn empty_after_strip_falls_back() {
        assert_eq!(slugify_ascii("！？"), "output");
    }

    #[test]
    fn single_quoted_xmlns_is_not_duplicated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ch1.xhtml");
        fs::write(
            &path,
            "<!DOCTYPE html><html xmlns='http://www.w3.org/1999/xhtml'>\
             <head></head><body><p>hi</p></body></html>",
        )
        .unwrap();
        standardize_xhtml_files(dir.path(), &|_| ()).unwrap();
        let out = fs::read_to_string(&path).unwrap();
        assert_eq!(
            out.matches("xmlns=").count(),
            1,
            "existing single-quoted xmlns must be recognized, not duplicated: {out}"
        );
    }

    #[test]
    fn patches_only_first_html_occurrence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ch1.xhtml");
        fs::write(
            &path,
            "<html><head></head><body><p>see &lt;html&gt; and <!-- <html --> more</p>\
             </body></html>",
        )
        .unwrap();
        standardize_xhtml_files(dir.path(), &|_| ()).unwrap();
        let out = fs::read_to_string(&path).unwrap();
        assert_eq!(
            out.matches("xmlns=\"http://www.w3.org/1999/xhtml\"")
                .count(),
            1,
            "only the first <html should be patched: {out}"
        );
        assert!(out.contains("<!-- <html -->"), "comment left alone: {out}");
    }

    #[test]
    fn doctype_internal_subset_fully_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ch1.xhtml");
        fs::write(
            &path,
            "<!DOCTYPE html [ <!ENTITY nbsp \"&#160;\"> ]>\n\
             <html xmlns=\"http://www.w3.org/1999/xhtml\"><body><p>a&nbsp;b</p></body></html>",
        )
        .unwrap();
        standardize_xhtml_files(dir.path(), &|_| ()).unwrap();
        let out = fs::read_to_string(&path).unwrap();
        assert!(out.starts_with("<!DOCTYPE html>"), "got: {out}");
        assert!(!out.contains("]>"), "stray internal-subset tail: {out}");
        assert!(!out.contains("<!ENTITY"), "entity decl left behind: {out}");
    }

    #[test]
    fn non_utf8_file_is_left_byte_identical() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("latin1.xhtml");
        // Latin-1 "café" (0xE9) inside an otherwise ASCII document: not UTF-8.
        let original: Vec<u8> =
            b"<!DOCTYPE html PUBLIC \"x\"><html><body><p>caf\xE9</p></body></html>".to_vec();
        fs::write(&path, &original).unwrap();
        standardize_xhtml_files(dir.path(), &|_| ()).unwrap();
        assert_eq!(
            fs::read(&path).unwrap(),
            original,
            "non-UTF-8 file must be skipped untouched"
        );
    }

    #[test]
    fn nesting_guard_bounds_deep_documents() {
        // Normal document: comments, PI, doctype, self-closing tags all fine.
        let normal = "<?xml version=\"1.0\"?><!DOCTYPE html>\
             <html><head><meta charset=\"utf-8\"/><!-- a <b> comment --></head>\
             <body><div><p>Hi<br/>there</p></div></body></html>";
        assert!(xml_nesting_ok(normal, 64));

        // 1000 nested elements must be rejected at a 256-level bound.
        let mut deep = String::new();
        for _ in 0..1000 {
            deep.push_str("<a>");
        }
        for _ in 0..1000 {
            deep.push_str("</a>");
        }
        assert!(!xml_nesting_ok(&deep, 256));
        assert!(xml_nesting_ok(&deep, 1000));
    }
}
