// epublift - Optimize EPUB files: convert images to WebP and upgrade to EPUB 3.3.
// Copyright (C) 2024  Baris Kayadelen
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! Mistral OCR provider for Smart Import.
//!
//! Two-stage pipeline:
//!
//! 1. **OCR** — `POST https://api.mistral.ai/v1/ocr` with the PDF inlined as a
//!    base64 data URL and `include_image_base64: true`. The response is a list
//!    of pages, each with `markdown` and `images` (base64). The OCR endpoint is
//!    a pure recognition model: it accepts **no free-text instructions** (its
//!    only extra knobs are JSON-schema annotation formats), so editorial
//!    cleanup cannot happen here.
//! 2. **Cleanup** — each chunk of pages is passed through
//!    `POST https://api.mistral.ai/v1/chat/completions` with instructions to
//!    strip OCR gibberish and hard-coded page numbers and to turn footnotes
//!    into linked Markdown footnotes. A chunk whose cleanup fails (or looks
//!    truncated) falls back to its raw OCR text, so cleanup can only improve
//!    the result, never lose it.
//!
//! The live HTTP calls can't be exercised without an API key, so the response
//! reconstruction ([`parse_ocr_response`], [`parse_chat_response`]) and the
//! chunking are covered by fixture tests.

use anyhow::{Context, Result, bail};
use base64::Engine;
use serde_json::Value;

/// OCR'd document: per-page Markdown plus its decoded images (`filename`,
/// bytes) where the filenames match the rewritten `![…](filename)` references.
pub(crate) struct OcrResult {
    pub pages: Vec<String>,
    pub images: Vec<(String, Vec<u8>)>,
}

impl OcrResult {
    /// The whole document, pages joined by blank lines.
    pub fn stitched(&self) -> String {
        self.pages.join("\n\n")
    }
}

const OCR_ENDPOINT: &str = "https://api.mistral.ai/v1/ocr";
const CHAT_ENDPOINT: &str = "https://api.mistral.ai/v1/chat/completions";
/// OCR responses carry base64 page images, so allow a generous response size.
const MAX_OCR_RESPONSE: usize = 128 * 1024 * 1024;
/// Chat responses are text only.
const MAX_CHAT_RESPONSE: usize = 32 * 1024 * 1024;

/// Model for the cleanup pass. Overridable via `MISTRAL_CLEANUP_MODEL`
/// (set it to `off` / `none` to skip cleanup entirely).
const DEFAULT_CLEANUP_MODEL: &str = "mistral-small-latest";

/// Cap on the Markdown characters sent to one cleanup call. Pages are grouped
/// into chunks up to this size so footnote markers stay in the same request as
/// their page-bottom texts, and each request stays well inside the model's
/// context window.
const MAX_CHUNK_CHARS: usize = 16_000;

/// Concurrent cleanup requests. Modest, to stay friendly to API rate limits
/// while keeping wall-clock time reasonable for long books.
const CLEANUP_CONCURRENCY: usize = 4;

/// The editorial instructions for the cleanup pass.
const CLEANUP_SYSTEM_PROMPT: &str = "\
You clean raw OCR output so it can be converted into an EPUB. You receive a \
fragment of a book as Markdown and must return that same fragment, cleaned, as \
Markdown only — no commentary, no explanations, and no surrounding code fence.\n\
Rules:\n\
1. Remove OCR noise: gibberish sequences, garbled or errant characters, and \
hyphenation artifacts (rejoin words split across line breaks).\n\
2. Remove hard-coded page numbers and repeated running headers/footers left \
over from the printed page layout.\n\
3. Preserve footnotes. Convert footnote markers into Markdown footnote \
references such as [^12], and their texts into matching definitions ([^12]: …) \
placed at the end of the section in which they are referenced. Keep the \
original numbering when it exists.\n\
4. Keep image references (lines like ![…](…)) exactly as they are — never \
alter, move, or delete them.\n\
5. Preserve all real content: wording, headings, emphasis, lists, and tables. \
Do not summarise, translate, reorder, or invent anything.";

/// Run Mistral OCR on `pdf`, returning the per-page Markdown + images.
pub(crate) fn ocr(api_key: &str, pdf: &[u8]) -> Result<OcrResult> {
    let http = crate::http::RustlsHttp::new()?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(pdf);
    // Note: the OCR endpoint takes no prompt. `document_annotation_format`
    // exists but requires a JSON schema and returns a *separate* structured
    // annotation (and only for small documents) — it does not modify the page
    // Markdown, so text cleanup happens in [`cleanup`] instead.
    let request = serde_json::json!({
        "model": "mistral-ocr-latest",
        "document": {
            "type": "document_url",
            "document_url": format!("data:application/pdf;base64,{b64}"),
        },
        "include_image_base64": true,
    });
    let body = serde_json::to_vec(&request).context("failed to build OCR request")?;

    let (status, resp) = http.post_json(OCR_ENDPOINT, api_key, &body, MAX_OCR_RESPONSE)?;
    if status != 200 {
        bail!(
            "Mistral OCR API returned HTTP {status}: {}",
            error_message(&resp)
        );
    }

    let value: Value =
        serde_json::from_slice(&resp).context("Mistral OCR API returned invalid JSON")?;
    parse_ocr_response(&value)
}

/// The cleanup model from the environment, or `None` when cleanup is disabled.
fn cleanup_model() -> Option<String> {
    let m = std::env::var("MISTRAL_CLEANUP_MODEL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_CLEANUP_MODEL.to_string());
    match m.to_ascii_lowercase().as_str() {
        "off" | "none" | "disabled" => None,
        _ => Some(m),
    }
}

/// Clean the OCR'd pages with the chat model (gibberish, page numbers,
/// footnotes — see [`CLEANUP_SYSTEM_PROMPT`]), returning the stitched result.
///
/// Chunks are processed with a small worker pool; any chunk whose cleanup call
/// fails or looks truncated keeps its raw OCR text. Returns `Ok(None)` when
/// cleanup is disabled via `MISTRAL_CLEANUP_MODEL=off`.
pub(crate) fn cleanup(api_key: &str, pages: &[String]) -> Result<Option<String>> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let Some(model) = cleanup_model() else {
        return Ok(None);
    };
    let chunks = chunk_pages(pages, MAX_CHUNK_CHARS);
    if chunks.is_empty() {
        return Ok(Some(String::new()));
    }

    let http = crate::http::RustlsHttp::new()?;
    let results: Mutex<Vec<Option<String>>> = Mutex::new(vec![None; chunks.len()]);
    let next = AtomicUsize::new(0);
    let fallbacks = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for _ in 0..CLEANUP_CONCURRENCY.min(chunks.len()) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = chunks.get(i) else { break };
                    let cleaned = match clean_chunk(&http, api_key, &model, chunk) {
                        Ok(text) => text,
                        Err(e) => {
                            eprintln!(
                                "[EXPERIMENTAL] warning: cleanup of chunk {}/{} failed ({e:#}); \
                                 keeping its raw OCR text",
                                i + 1,
                                chunks.len()
                            );
                            fallbacks.fetch_add(1, Ordering::Relaxed);
                            chunk.clone()
                        }
                    };
                    results.lock().unwrap()[i] = Some(cleaned);
                }
            });
        }
    });

    let fallbacks = fallbacks.into_inner();
    if fallbacks > 0 {
        eprintln!(
            "[EXPERIMENTAL] warning: {fallbacks} of {} cleanup chunk(s) kept raw OCR text",
            chunks.len()
        );
    }
    let out: Vec<String> = results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(Option::unwrap_or_default)
        .collect();
    Ok(Some(out.join("\n\n")))
}

/// Group whole pages into chunks of at most `max_chars` (a single oversized
/// page becomes its own chunk). Blank pages are dropped.
fn chunk_pages(pages: &[String], max_chars: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for page in pages {
        if page.trim().is_empty() {
            continue;
        }
        if !cur.is_empty() && cur.len() + 2 + page.len() > max_chars {
            chunks.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push_str("\n\n");
        }
        cur.push_str(page);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Clean one chunk via the chat API. Errors (and truncated/suspiciously short
/// answers) are surfaced so the caller can fall back to the raw text.
fn clean_chunk(
    http: &crate::http::RustlsHttp,
    api_key: &str,
    model: &str,
    chunk: &str,
) -> Result<String> {
    let request = serde_json::json!({
        "model": model,
        "temperature": 0.0,
        "messages": [
            { "role": "system", "content": CLEANUP_SYSTEM_PROMPT },
            { "role": "user", "content": chunk },
        ],
    });
    let body = serde_json::to_vec(&request).context("failed to build cleanup request")?;
    let (status, resp) = http.post_json(CHAT_ENDPOINT, api_key, &body, MAX_CHAT_RESPONSE)?;
    if status != 200 {
        bail!(
            "Mistral chat API returned HTTP {status}: {}",
            error_message(&resp)
        );
    }
    let value: Value =
        serde_json::from_slice(&resp).context("Mistral chat API returned invalid JSON")?;
    let (finish_reason, content) = parse_chat_response(&value)?;
    if finish_reason != "stop" {
        bail!("cleanup response was cut off (finish_reason: {finish_reason})");
    }
    let cleaned = strip_code_fence(content.trim());
    // A cleanup pass only *removes* noise; an answer far shorter than the input
    // means the model dropped content (or answered with commentary).
    if cleaned.len() < chunk.len() / 4 {
        bail!(
            "cleanup response suspiciously short ({} of {} chars)",
            cleaned.len(),
            chunk.len()
        );
    }
    Ok(cleaned.to_string())
}

/// Extract `(finish_reason, message content)` from a chat completion response.
fn parse_chat_response(value: &Value) -> Result<(String, String)> {
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .context("chat response had no choices")?;
    let finish = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let content = choice
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .context("chat response had no message content")?
        .to_string();
    Ok((finish, content))
}

/// Remove a surrounding ``` fence if the model wrapped its answer in one.
fn strip_code_fence(s: &str) -> &str {
    let t = s.trim();
    if let Some(rest) = t.strip_prefix("```") {
        // Skip the info string ("markdown", "md", …) on the opening fence line.
        let body = rest.split_once('\n').map(|(_, b)| b).unwrap_or("");
        if let Some(inner) = body.strip_suffix("```") {
            return inner.trim_matches('\n');
        }
    }
    t
}

/// Pull a human-readable error out of an error response body (best effort).
fn error_message(resp: &[u8]) -> String {
    let text = String::from_utf8_lossy(resp);
    if let Ok(v) = serde_json::from_str::<Value>(&text) {
        // Mistral errors look like {"message": "..."} or {"detail": "..."}.
        for key in ["message", "detail", "error"] {
            if let Some(s) = v.get(key).and_then(|x| x.as_str()) {
                return truncate(s, 300);
            }
        }
    }
    truncate(text.trim(), 300)
}

/// Reconstruct per-page Markdown + decoded images from an OCR response.
///
/// Image ids are page-scoped (`img-0.jpeg` can repeat per page), so we namespace
/// each as `p{page}_{id}` and rewrite that page's links to match.
fn parse_ocr_response(value: &Value) -> Result<OcrResult> {
    let pages = value
        .get("pages")
        .and_then(|p| p.as_array())
        .context("OCR response had no 'pages' array")?;

    let mut page_md_list: Vec<String> = Vec::new();
    let mut images: Vec<(String, Vec<u8>)> = Vec::new();

    for (pi, page) in pages.iter().enumerate() {
        let mut page_md = page
            .get("markdown")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .to_string();

        if let Some(imgs) = page.get("images").and_then(|i| i.as_array()) {
            for img in imgs {
                let Some(id) = img
                    .get("id")
                    .and_then(|x| x.as_str())
                    .filter(|s| !s.is_empty())
                else {
                    continue;
                };
                let Some(data) = img.get("image_base64").and_then(|x| x.as_str()) else {
                    continue;
                };
                let bytes = match decode_data_uri(data) {
                    Ok(b) => b,
                    Err(_) => continue, // skip an undecodable image rather than fail the book
                };
                let name = format!("p{pi}_{}", sanitize_name(id));
                // Re-point this page's links (`](id)` / `](./id)`) at the file.
                page_md = page_md
                    .replace(&format!("]({id})"), &format!("](./{name})"))
                    .replace(&format!("](./{id})"), &format!("](./{name})"));
                images.push((name, bytes));
            }
        }

        page_md_list.push(page_md);
    }

    Ok(OcrResult {
        pages: page_md_list,
        images,
    })
}

/// Decode an image that may be a bare base64 string or a `data:…;base64,…` URI.
fn decode_data_uri(s: &str) -> Result<Vec<u8>> {
    let b64 = match s.strip_prefix("data:") {
        Some(rest) => rest
            .split_once(',')
            .map(|(_, data)| data)
            .context("malformed data URI")?,
        None => s,
    };
    let cleaned: String = b64.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(cleaned.as_bytes())
        .context("invalid base64 image data")
}

/// Flatten an image id into a safe, path-component-free filename.
fn sanitize_name(id: &str) -> String {
    let base = id.rsplit(['/', '\\']).next().unwrap_or(id);
    let cleaned: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "image".to_string()
    } else {
        cleaned
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1x1 transparent PNG, base64-encoded (standard alphabet).
    const PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGNgAAAAAgABc3UBGAAAAABJRU5ErkJggg==";

    #[test]
    fn decodes_plain_and_data_uri_base64() {
        let plain = decode_data_uri(PNG_B64).unwrap();
        let uri = decode_data_uri(&format!("data:image/png;base64,{PNG_B64}")).unwrap();
        assert_eq!(plain, uri);
        assert_eq!(&plain[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn sanitizes_ids() {
        assert_eq!(sanitize_name("img-0.jpeg"), "img-0.jpeg");
        assert_eq!(sanitize_name("../../etc/passwd"), "passwd");
        assert_eq!(sanitize_name("a b/c?.png"), "c_.png");
    }

    #[test]
    fn parses_pages_and_rewrites_image_links() {
        let resp = serde_json::json!({
            "pages": [
                {
                    "index": 0,
                    "markdown": "# Title\n\n![img-0.jpeg](img-0.jpeg)\n\nHello.",
                    "images": [
                        { "id": "img-0.jpeg", "image_base64": format!("data:image/png;base64,{PNG_B64}") }
                    ]
                },
                {
                    "index": 1,
                    "markdown": "More on page two.",
                    "images": []
                }
            ]
        });
        let out = parse_ocr_response(&resp).unwrap();
        // Page link rewritten to the namespaced filename, both pages kept.
        assert_eq!(out.pages.len(), 2);
        let stitched = out.stitched();
        assert!(stitched.contains("](./p0_img-0.jpeg)"));
        assert!(stitched.contains("# Title"));
        assert!(stitched.contains("More on page two."));
        assert_eq!(out.images.len(), 1);
        assert_eq!(out.images[0].0, "p0_img-0.jpeg");
        assert_eq!(&out.images[0].1[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn chunks_group_whole_pages() {
        let pages = vec![
            "a".repeat(10),
            String::new(), // blank page dropped
            "b".repeat(10),
            "c".repeat(30), // doesn't fit with a+b under the cap
        ];
        let chunks = chunk_pages(&pages, 25);
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0],
            format!("{}\n\n{}", "a".repeat(10), "b".repeat(10))
        );
        assert_eq!(chunks[1], "c".repeat(30)); // oversized page stands alone
    }

    #[test]
    fn parses_chat_response() {
        let resp = serde_json::json!({
            "choices": [{
                "finish_reason": "stop",
                "message": { "role": "assistant", "content": "Cleaned text." }
            }]
        });
        let (finish, content) = parse_chat_response(&resp).unwrap();
        assert_eq!(finish, "stop");
        assert_eq!(content, "Cleaned text.");
        assert!(parse_chat_response(&serde_json::json!({"choices": []})).is_err());
    }

    #[test]
    fn strips_code_fences() {
        assert_eq!(strip_code_fence("plain"), "plain");
        assert_eq!(strip_code_fence("```markdown\n# Hi\n```"), "# Hi");
        assert_eq!(strip_code_fence("```\ntext\n```"), "text");
        // An unterminated fence is left alone.
        assert_eq!(strip_code_fence("```\ntext"), "```\ntext");
    }
}
