//! Generation of an EPUB 3 XHTML Navigation Document (`nav.xhtml`) from a
//! legacy `toc.ncx`, including a `landmarks` block built from the OPF `<guide>`.

use crate::opf::GuideRef;
use anyhow::{Context, Result};
use regex::Regex;
use std::fs;
use std::path::Path;

/// Maximum `<navPoint>` nesting depth we honour; deeper levels are dropped so a
/// hostile NCX cannot overflow the stack via recursion.
const MAX_NAV_DEPTH: usize = 64;

/// Remove a `<!DOCTYPE ...>` declaration so the strict, DTD-rejecting XML parser
/// (roxmltree) will accept the document. NCX files commonly carry a DOCTYPE.
fn strip_doctype(xml: &str) -> String {
    let re = Regex::new(r"(?is)<!DOCTYPE[^>]*>").unwrap();
    re.replace(xml, "").into_owned()
}

/// Escape untrusted text for inclusion in XHTML element content or attribute
/// values, dropping control chars illegal in XML 1.0. (Local copy of
/// `epub_writer::esc`, which is feature-gated and not always compiled in.)
fn esc(s: &str) -> String {
    s.chars()
        // Drop control chars that are illegal in XML 1.0 (keep tab/newline/CR).
        .filter(|&c| c >= ' ' || c == '\t' || c == '\n' || c == '\r')
        .map(|c| match c {
            '&' => "&amp;".into(),
            '<' => "&lt;".into(),
            '>' => "&gt;".into(),
            '"' => "&quot;".into(),
            other => other.to_string(),
        })
        .collect()
}

/// A node in the navigation tree parsed from `toc.ncx`.
struct NavPoint {
    title: String,
    href: String,
    children: Vec<NavPoint>,
}

/// Parse `toc.ncx` and write a valid EPUB 3 Navigation Document to `out_path`.
pub fn generate_nav_xhtml(ncx_path: &Path, out_path: &Path, guide_refs: &[GuideRef]) -> Result<()> {
    let raw = fs::read_to_string(ncx_path).context("Failed to read toc.ncx")?;
    let xml = strip_doctype(&raw);
    // roxmltree's parser recurses per nesting level; bound the depth before
    // parsing so a crafted NCX can't overflow the stack (uncatchable).
    if !crate::util::xml_nesting_ok(&xml, 256) {
        anyhow::bail!("toc.ncx is nested too deeply to be a real table of contents");
    }
    let doc = roxmltree::Document::parse(&xml).context("Failed to parse toc.ncx")?;

    // Document title.
    let title_text = doc
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "docTitle")
        .and_then(|d| {
            d.children()
                .find(|n| n.is_element() && n.tag_name().name() == "text")
        })
        .and_then(|t| t.text())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Table of Contents".to_string());

    // Navigation tree.
    let toc_items = doc
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "navMap")
        .map(|nav_map| parse_nav_points(nav_map, 0))
        .unwrap_or_default();

    let mut out: Vec<String> = Vec::new();
    out.push(r#"<?xml version="1.0" encoding="utf-8"?>"#.to_string());
    out.push("<!DOCTYPE html>".to_string());
    out.push(r#"<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">"#.to_string());
    out.push("<head>".to_string());
    out.push(format!("  <title>{}</title>", esc(&title_text)));
    out.push(r#"  <meta charset="utf-8" />"#.to_string());
    out.push("  <style>".to_string());
    out.push("    body { font-family: sans-serif; margin: 2em; }".to_string());
    out.push("    nav ol { list-style-type: none; padding-left: 1.5em; }".to_string());
    out.push("    nav ol li { margin: 0.5em 0; }".to_string());
    out.push("    a { text-decoration: none; color: #1a73e8; }".to_string());
    out.push("    a:hover { text-decoration: underline; }".to_string());
    out.push("    h1 { color: #333333; }".to_string());
    out.push("  </style>".to_string());
    out.push("</head>".to_string());
    out.push("<body>".to_string());
    out.push(r#"  <nav epub:type="toc" id="toc">"#.to_string());
    out.push(format!("    <h1>{}</h1>", esc(&title_text)));

    if toc_items.is_empty() {
        out.push("    <p>No table of contents available.</p>".to_string());
    } else {
        render_ol(&toc_items, 2, 0, &mut out);
    }

    out.push("  </nav>".to_string());

    // Landmarks derived from the legacy <guide>.
    if !guide_refs.is_empty() {
        out.push("\n  <nav epub:type=\"landmarks\" id=\"landmarks\" hidden=\"\">".to_string());
        out.push("    <h2>Guide Landmarks</h2>".to_string());
        out.push("    <ol>".to_string());
        for r in guide_refs {
            let href = r.href.replace('\\', "/");
            let mapped = map_landmark_type(&r.ref_type);
            out.push(format!(
                "      <li><a epub:type=\"{}\" href=\"{}\">{}</a></li>",
                mapped,
                esc(&href),
                esc(&r.title)
            ));
        }
        out.push("    </ol>".to_string());
        out.push("  </nav>".to_string());
    }

    out.push("</body>".to_string());
    out.push("</html>".to_string());

    fs::write(out_path, out.join("\n"))?;
    Ok(())
}

/// Recursively collect `<navPoint>` children of `parent`, ignoring anything
/// nested deeper than `MAX_NAV_DEPTH` levels.
fn parse_nav_points(parent: roxmltree::Node, depth: usize) -> Vec<NavPoint> {
    let mut items = Vec::new();
    for p in parent
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "navPoint")
    {
        let title = p
            .children()
            .find(|n| n.is_element() && n.tag_name().name() == "navLabel")
            .and_then(|label| {
                label
                    .children()
                    .find(|n| n.is_element() && n.tag_name().name() == "text")
            })
            .and_then(|t| t.text())
            .unwrap_or("Untitled")
            .to_string();

        let href = p
            .children()
            .find(|n| n.is_element() && n.tag_name().name() == "content")
            .and_then(|c| c.attribute("src"))
            .unwrap_or("")
            .to_string();

        // Skip children beyond the depth cap so untrusted input cannot drive
        // unbounded recursion.
        let children = if depth < MAX_NAV_DEPTH {
            parse_nav_points(p, depth + 1)
        } else {
            Vec::new()
        };
        items.push(NavPoint {
            title,
            href,
            children,
        });
    }
    items
}

/// Render a nested ordered list mirroring the navigation tree, ignoring
/// anything nested deeper than `MAX_NAV_DEPTH` levels.
fn render_ol(items: &[NavPoint], level: usize, depth: usize, out: &mut Vec<String>) {
    let indent = "  ".repeat(level);
    out.push(format!("{}<ol>", indent));
    for item in items {
        let href = item.href.replace('\\', "/");
        out.push(format!("{}  <li>", indent));
        if href.is_empty() {
            out.push(format!("{}    <span>{}</span>", indent, esc(&item.title)));
        } else {
            out.push(format!(
                "{}    <a href=\"{}\">{}</a>",
                indent,
                esc(&href),
                esc(&item.title)
            ));
        }
        if !item.children.is_empty() && depth < MAX_NAV_DEPTH {
            render_ol(&item.children, level + 2, depth + 1, out);
        }
        out.push(format!("{}  </li>", indent));
    }
    out.push(format!("{}</ol>", indent));
}

/// Map EPUB 2 guide types to EPUB 3 landmark semantics.
///
/// The `epub:type` value is an attribute token, so an unrecognised type is
/// passed through only if it consists solely of safe token characters;
/// anything else falls back to a default.
fn map_landmark_type(ref_type: &str) -> String {
    match ref_type {
        "text" => "bodymatter",
        "title-page" => "titlepage",
        "acknowledgements" => "acknowledgments",
        "cover" => "cover",
        "toc" => "toc",
        other
            if !other.is_empty()
                && other
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')) =>
        {
            other
        }
        _ => "bodymatter",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_titles_and_hrefs_in_toc() {
        let items = vec![NavPoint {
            title: "Cats & <Dogs>".to_string(),
            href: "ch1.xhtml?a=1&b=\"2\"".to_string(),
            children: Vec::new(),
        }];
        let mut out = Vec::new();
        render_ol(&items, 0, 0, &mut out);
        let html = out.join("\n");
        assert!(html.contains("Cats &amp; &lt;Dogs&gt;"), "html: {html}");
        assert!(
            html.contains("href=\"ch1.xhtml?a=1&amp;b=&quot;2&quot;\""),
            "html: {html}"
        );
        assert!(!html.contains("Cats & <Dogs>"));
    }

    #[test]
    fn deeply_nested_nav_points_are_capped() {
        // Build an NCX navMap nested past the cap; parsing must neither
        // overflow the stack nor keep levels beyond MAX_NAV_DEPTH. (Kept
        // modest because roxmltree's own parser is also recursive and the
        // libtest thread stack is small.)
        let deep = 100;
        let mut xml = String::from("<navMap>");
        for i in 0..deep {
            xml.push_str(&format!(
                "<navPoint><navLabel><text>t{i}</text></navLabel><content src=\"c{i}.xhtml\"/>"
            ));
        }
        xml.push_str(&"</navPoint>".repeat(deep));
        xml.push_str("</navMap>");

        let doc = roxmltree::Document::parse(&xml).unwrap();
        let items = parse_nav_points(doc.root_element(), 0);

        let mut depth = 0;
        let mut cur = &items;
        while let Some(first) = cur.first() {
            depth += 1;
            cur = &first.children;
        }
        assert_eq!(depth, MAX_NAV_DEPTH + 1);

        // Rendering the capped tree must also complete without overflowing.
        let mut out = Vec::new();
        render_ol(&items, 0, 0, &mut out);
    }

    #[test]
    fn landmark_type_rejects_unsafe_tokens() {
        assert_eq!(map_landmark_type("text"), "bodymatter");
        assert_eq!(map_landmark_type("loi"), "loi");
        assert_eq!(map_landmark_type("a\" onclick=\"x"), "bodymatter");
        assert_eq!(map_landmark_type(""), "bodymatter");
    }
}
