#!/usr/bin/env python3
"""Convert Hexo HTML blog posts to Markdown for Obsidian."""

import os
import re
from pathlib import Path
from html.parser import HTMLParser
from html import unescape

class HTMLToMarkdown(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self.in_content = False
        self.in_code = False
        self.in_pre = False
        self.in_table = False
        self.in_thead = False
        self.in_tbody = False
        self.in_tr = False
        self.in_th = False
        self.in_td = False
        self.in_blockquote = False
        self.in_ul = False
        self.in_ol = False
        self.in_li = False
        self.in_strong = False
        self.in_em = False
        self.in_a = False
        self.in_h = 0
        self.in_p = False
        self.code_lang = ""
        self.current_link = ""
        self.table_row = []
        self.table_header = []
        self.skip_until_more = True  # Skip until we pass the "more" marker
        self.title = ""
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag == 'h1' and 'title' in attrs_dict.get('class', ''):
            self.in_title = True
            return

        if tag == 'span' and attrs_dict.get('id') == 'more':
            self.skip_until_more = False
            return

        if self.skip_until_more:
            return

        if tag == 'div' and 'content' in attrs_dict.get('class', ''):
            self.in_content = True

        if not self.in_content:
            return

        if tag == 'figure' and 'highlight' in attrs_dict.get('class', ''):
            classes = attrs_dict.get('class', '').split()
            for c in classes:
                if c != 'highlight':
                    self.code_lang = c
                    break
            self.in_pre = True
            self.result.append(f"\n```{self.code_lang}\n")
        elif tag == 'pre':
            if not self.in_pre:
                self.in_pre = True
                self.result.append("\n```\n")
        elif tag == 'code':
            if not self.in_pre:
                self.result.append("`")
                self.in_code = True
        elif tag == 'table':
            self.in_table = True
            self.result.append("\n")
        elif tag == 'thead':
            self.in_thead = True
        elif tag == 'tbody':
            self.in_tbody = True
        elif tag == 'tr':
            self.in_tr = True
            self.table_row = []
        elif tag == 'th':
            self.in_th = True
        elif tag == 'td':
            self.in_td = True
        elif tag == 'blockquote':
            self.in_blockquote = True
            self.result.append("\n> ")
        elif tag == 'ul':
            self.in_ul = True
            self.result.append("\n")
        elif tag == 'ol':
            self.in_ol = True
            self.result.append("\n")
        elif tag == 'li':
            self.in_li = True
            if self.in_ul:
                self.result.append("- ")
            elif self.in_ol:
                self.result.append("1. ")
        elif tag == 'strong' or tag == 'b':
            self.in_strong = True
            self.result.append("**")
        elif tag == 'em' or tag == 'i':
            self.in_em = True
            self.result.append("*")
        elif tag == 'a':
            self.in_a = True
            self.current_link = attrs_dict.get('href', '')
            self.result.append("[")
        elif tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = int(tag[1])
            self.in_h = level
            self.result.append("\n" + "#" * level + " ")
        elif tag == 'p':
            self.in_p = True
            if not self.in_li and not self.in_blockquote:
                self.result.append("\n")
        elif tag == 'br':
            self.result.append("\n")

    def handle_endtag(self, tag):
        if tag == 'h1' and self.in_title:
            self.in_title = False
            return

        if self.skip_until_more:
            return

        if tag == 'figure':
            if self.in_pre:
                self.result.append("```\n")
                self.in_pre = False
                self.code_lang = ""
        elif tag == 'pre':
            pass  # Handled by figure
        elif tag == 'code':
            if self.in_code:
                self.result.append("`")
                self.in_code = False
        elif tag == 'table':
            self.in_table = False
        elif tag == 'thead':
            self.in_thead = False
        elif tag == 'tbody':
            self.in_tbody = False
        elif tag == 'tr':
            self.in_tr = False
            if self.table_row:
                self.result.append("| " + " | ".join(self.table_row) + " |\n")
                if self.in_thead:
                    self.result.append("|" + "|".join(["---"] * len(self.table_row)) + "|\n")
        elif tag == 'th':
            self.in_th = False
        elif tag == 'td':
            self.in_td = False
        elif tag == 'blockquote':
            self.in_blockquote = False
            self.result.append("\n")
        elif tag == 'ul':
            self.in_ul = False
            self.result.append("\n")
        elif tag == 'ol':
            self.in_ol = False
            self.result.append("\n")
        elif tag == 'li':
            self.in_li = False
            self.result.append("\n")
        elif tag == 'strong' or tag == 'b':
            self.in_strong = False
            self.result.append("**")
        elif tag == 'em' or tag == 'i':
            self.in_em = False
            self.result.append("*")
        elif tag == 'a':
            self.in_a = False
            self.result.append(f"]({self.current_link})")
            self.current_link = ""
        elif tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            self.in_h = 0
            self.result.append("\n")
        elif tag == 'p':
            self.in_p = False
            self.result.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title = data.strip()
            return

        if self.skip_until_more:
            return

        if not self.in_content:
            return

        # Clean up data
        text = data

        if self.in_pre:
            # For code blocks, preserve exact content but clean line numbers
            if text.strip().isdigit():
                return
            self.result.append(text)
        elif self.in_th or self.in_td:
            self.table_row.append(text.strip())
        else:
            # For regular text, normalize whitespace
            text = text.replace('\n', ' ')
            if text.strip():
                self.result.append(text)

    def get_markdown(self):
        md = ''.join(self.result)
        # Clean up multiple newlines
        md = re.sub(r'\n{3,}', '\n\n', md)
        # Clean up spaces before punctuation
        md = re.sub(r' +([.,;:!?])', r'\1', md)
        return md.strip()


def convert_html_to_md(html_path, output_dir):
    """Convert a single HTML file to Markdown."""
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    parser = HTMLToMarkdown()
    parser.feed(html_content)

    # Get the directory name as the filename
    dir_name = html_path.parent.name
    md_filename = f"{dir_name}.md"

    # Create frontmatter
    title = parser.title or dir_name
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

    md_content = frontmatter + parser.get_markdown()

    output_path = output_dir / md_filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"Converted: {html_path} -> {output_path}")
    return output_path


def main():
    html_base = Path("/home/qianxu/drqianxu.github.io/.deploy_git/2024/12/24")
    output_dir = Path("/mnt/c/Users/DrQianXu/Documents/Obsidian Vault")

    # Find all 0x* directories
    for dir_path in sorted(html_base.iterdir()):
        if dir_path.is_dir() and dir_path.name.startswith("0x"):
            html_file = dir_path / "index.html"
            if html_file.exists():
                convert_html_to_md(html_file, output_dir)

    # Also convert 0x12 from different date
    html_28 = Path("/home/qianxu/drqianxu.github.io/.deploy_git/2024/12/28")
    for dir_path in sorted(html_28.iterdir()):
        if dir_path.is_dir() and dir_path.name.startswith("0x"):
            html_file = dir_path / "index.html"
            if html_file.exists():
                convert_html_to_md(html_file, output_dir)


if __name__ == "__main__":
    main()
