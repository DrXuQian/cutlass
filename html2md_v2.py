#!/usr/bin/env python3
"""Convert Hexo HTML blog posts to Markdown for Obsidian - Version 2."""

import os
import re
from pathlib import Path
from html import unescape


def extract_content(html: str) -> tuple[str, str]:
    """Extract title and article content from HTML."""
    # Extract title
    title_match = re.search(r'<h1 class="title[^"]*">([^<]+)</h1>', html)
    title = title_match.group(1) if title_match else ""

    # Extract content between <div class="content"> and </div><div class="article-licensing
    content_match = re.search(
        r'<div class="content">(.*?)</div><div class="article-licensing',
        html, re.DOTALL
    )
    if not content_match:
        return title, ""

    content = content_match.group(1)
    return title, content


def html_to_markdown(html: str) -> str:
    """Convert HTML content to Markdown."""
    md = html

    # Remove the introductory paragraph before <!-- more -->
    md = re.sub(r'^<p>.*?</p>\s*<span id="more"></span>\s*', '', md, flags=re.DOTALL)

    # Remove blockquote with example code link at the beginning
    md = re.sub(r'<blockquote>\s*<p><strong>示例代码</strong>.*?</blockquote>\s*', '', md, flags=re.DOTALL)

    # Convert headers
    for i in range(6, 0, -1):
        # Handle headers with anchors
        md = re.sub(
            rf'<h{i}[^>]*><a[^>]*></a>([^<]+)</h{i}>',
            lambda m: '\n' + '#' * i + ' ' + m.group(1).strip() + '\n',
            md
        )
        md = re.sub(
            rf'<h{i}[^>]*>([^<]+)</h{i}>',
            lambda m: '\n' + '#' * i + ' ' + m.group(1).strip() + '\n',
            md
        )

    # Convert code blocks (figure with highlight class)
    def convert_code_block(match):
        classes = match.group(1)
        code_html = match.group(2)

        # Determine language
        lang = ""
        for cls in classes.split():
            if cls != "highlight":
                lang = cls
                break

        # Extract code lines
        lines = []
        for line_match in re.finditer(r'<span class="line">([^<]*(?:<[^>]+>[^<]*)*)</span>', code_html):
            line = line_match.group(1)
            # Remove HTML tags but keep content
            line = re.sub(r'<[^>]+>', '', line)
            line = unescape(line)
            lines.append(line)

        code = '\n'.join(lines)
        return f'\n```{lang}\n{code}\n```\n'

    md = re.sub(
        r'<figure class="highlight ([^"]+)"[^>]*>.*?<pre>(.*?)</pre>.*?</figure>',
        convert_code_block,
        md, flags=re.DOTALL
    )

    # Convert tables
    def convert_table(match):
        table_html = match.group(0)

        # Extract headers
        headers = []
        header_match = re.search(r'<thead>(.*?)</thead>', table_html, re.DOTALL)
        if header_match:
            for th in re.finditer(r'<th[^>]*>(.*?)</th>', header_match.group(1), re.DOTALL):
                text = re.sub(r'<[^>]+>', '', th.group(1))
                headers.append(unescape(text.strip()))

        # Extract rows
        rows = []
        tbody_match = re.search(r'<tbody>(.*?)</tbody>', table_html, re.DOTALL)
        if tbody_match:
            for tr in re.finditer(r'<tr>(.*?)</tr>', tbody_match.group(1), re.DOTALL):
                row = []
                for td in re.finditer(r'<td[^>]*>(.*?)</td>', tr.group(1), re.DOTALL):
                    text = re.sub(r'<[^>]+>', '', td.group(1))
                    row.append(unescape(text.strip()))
                if row:
                    rows.append(row)

        if not headers:
            return ""

        # Build markdown table
        result = '\n| ' + ' | '.join(headers) + ' |\n'
        result += '|' + '|'.join(['---'] * len(headers)) + '|\n'
        for row in rows:
            # Pad row if needed
            while len(row) < len(headers):
                row.append('')
            result += '| ' + ' | '.join(row) + ' |\n'

        return result

    md = re.sub(r'<table>.*?</table>', convert_table, md, flags=re.DOTALL)

    # Convert inline code
    md = re.sub(r'<code>([^<]+)</code>', r'`\1`', md)

    # Convert bold
    md = re.sub(r'<strong>([^<]+)</strong>', r'**\1**', md)
    md = re.sub(r'<b>([^<]+)</b>', r'**\1**', md)

    # Convert italic
    md = re.sub(r'<em>([^<]+)</em>', r'*\1*', md)
    md = re.sub(r'<i>([^<]+)</i>', r'*\1*', md)

    # Convert links
    md = re.sub(r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', r'[\2](\1)', md)

    # Convert lists
    md = re.sub(r'<ul>\s*', '\n', md)
    md = re.sub(r'</ul>\s*', '\n', md)
    md = re.sub(r'<ol>\s*', '\n', md)
    md = re.sub(r'</ol>\s*', '\n', md)
    md = re.sub(r'<li>([^<]+)</li>', r'- \1\n', md)
    md = re.sub(r'<li>(.*?)</li>', lambda m: '- ' + re.sub(r'<[^>]+>', '', m.group(1)) + '\n', md, flags=re.DOTALL)

    # Convert blockquotes
    def convert_blockquote(match):
        content = match.group(1)
        content = re.sub(r'<p>', '', content)
        content = re.sub(r'</p>', '', content)
        content = re.sub(r'<[^>]+>', '', content)
        lines = [f'> {line.strip()}' for line in content.strip().split('\n') if line.strip()]
        return '\n' + '\n'.join(lines) + '\n'

    md = re.sub(r'<blockquote>(.*?)</blockquote>', convert_blockquote, md, flags=re.DOTALL)

    # Convert paragraphs
    md = re.sub(r'<p>([^<]*)</p>', r'\n\1\n', md)
    md = re.sub(r'<p>(.*?)</p>', lambda m: '\n' + re.sub(r'<[^>]+>', '', m.group(1)) + '\n', md, flags=re.DOTALL)

    # Remove remaining HTML tags
    md = re.sub(r'<br\s*/?>', '\n', md)
    md = re.sub(r'<[^>]+>', '', md)

    # Unescape HTML entities
    md = unescape(md)

    # Clean up whitespace
    md = re.sub(r'\n{3,}', '\n\n', md)
    md = re.sub(r' +', ' ', md)
    md = md.strip()

    return md


def convert_html_to_md(html_path: Path, output_dir: Path):
    """Convert a single HTML file to Markdown."""
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    title, content_html = extract_content(html_content)
    md_content = html_to_markdown(content_html)

    # Get filename from directory name
    dir_name = html_path.parent.name
    md_filename = f"{dir_name}.md"

    # Create frontmatter
    frontmatter = f"""---
title: "{title}"
date: 2024-12-24
categories:
  - CUTLASS
tags:
  - CUTLASS
  - CuTe
---

"""

    full_content = frontmatter + md_content

    output_path = output_dir / md_filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_content)

    print(f"Converted: {html_path.parent.name} -> {output_path.name}")
    return output_path


def main():
    html_base_24 = Path("/home/qianxu/drqianxu.github.io/.deploy_git/2024/12/24")
    html_base_28 = Path("/home/qianxu/drqianxu.github.io/.deploy_git/2024/12/28")
    output_dir = Path("/mnt/c/Users/DrQianXu/Documents/Obsidian Vault")

    # Convert all 0x* directories from Dec 24
    for dir_path in sorted(html_base_24.iterdir()):
        if dir_path.is_dir() and dir_path.name.startswith("0x"):
            html_file = dir_path / "index.html"
            if html_file.exists():
                convert_html_to_md(html_file, output_dir)

    # Skip 0x12 from Dec 28 since we already have the original markdown


if __name__ == "__main__":
    main()
