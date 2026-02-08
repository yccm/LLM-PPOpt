from __future__ import annotations

import json
import re
from pathlib import Path

INPUT_JSON = Path("test_persona_conversations.json")
OUTPUT_TEX = Path("appendix.tex")
INDENT_UNIT_EM = 1.2


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def render_inline_markdown(text: str) -> str:
    token_map: dict[str, str] = {}
    token_index = 0

    def stash(value: str) -> str:
        nonlocal token_index
        token = f"@@TOK{token_index}@@"
        token_index += 1
        token_map[token] = value
        return token

    rendered = text
    rendered = re.sub(
        r"`([^`]+)`",
        lambda m: stash(rf"\texttt{{{escape_latex(m.group(1))}}}"),
        rendered,
    )

    while True:
        bold_match = re.search(r"\*\*(.+?)\*\*", rendered)
        if not bold_match:
            break
        inner = render_inline_markdown(bold_match.group(1))
        rendered = (
            rendered[: bold_match.start()]
            + stash(rf"\textbf{{{inner}}}")
            + rendered[bold_match.end() :]
        )

    while True:
        italic_match = re.search(r"(?<!\*)\*([^*]+)\*(?!\*)", rendered)
        if not italic_match:
            break
        inner = render_inline_markdown(italic_match.group(1))
        rendered = (
            rendered[: italic_match.start()]
            + stash(rf"\textit{{{inner}}}")
            + rendered[italic_match.end() :]
        )

    rendered = escape_latex(rendered)
    for token, value in token_map.items():
        rendered = rendered.replace(token, value)

    return rendered


def quote_depth_and_content(line: str) -> tuple[int, str]:
    leading_ws_match = re.match(r"^(\s*)", line)
    leading_ws = leading_ws_match.group(1) if leading_ws_match else ""
    stripped = line[len(leading_ws) :]
    depth = 0
    while stripped.startswith(">"):
        depth += 1
        stripped = stripped[1:]
        if stripped.startswith(" "):
            stripped = stripped[1:]
    return depth, leading_ws + stripped


def indent_prefix(level: int) -> str:
    if level <= 0:
        return ""
    return rf"\hspace*{{{level * INDENT_UNIT_EM:.1f}em}}"


def convert_markdown_line_to_latex(line: str) -> tuple[str, bool]:
    if line.strip() == "":
        return "", False

    depth, core = quote_depth_and_content(line.rstrip())
    core = core.rstrip()
    core_stripped = core.strip()

    if core_stripped in {"---", "***", "___"}:
        # Keep separators as readable text instead of forcing horizontal rules.
        return indent_prefix(depth + 1) + "---", False

    if core_stripped.startswith("|") and core_stripped.endswith("|"):
        cells = [cell.strip() for cell in core_stripped[1:-1].split("|")]
        if cells and all(re.fullmatch(r"[:\-]+", cell or "-") for cell in cells):
            return "", False
        rendered_cells = [render_inline_markdown(cell) for cell in cells]
        return indent_prefix(depth + 1) + " | ".join(rendered_cells), False

    heading_match = re.match(r"^(#{1,6})\s+(.*)$", core)
    if heading_match:
        level = len(heading_match.group(1))
        heading_text = render_inline_markdown(heading_match.group(2).strip())
        return indent_prefix(depth + level) + rf"\textbf{{{heading_text}}}", True

    unordered_match = re.match(r"^(\s*)[-*+]\s+(.*)$", core)
    if unordered_match:
        spaces = len(unordered_match.group(1).replace("\t", "    "))
        list_level = spaces // 2
        item_text = render_inline_markdown(unordered_match.group(2).strip())
        return indent_prefix(depth + 1 + list_level) + r"\textbullet{} " + item_text, False

    ordered_match = re.match(r"^(\s*)(\d+)[\.\)]\s+(.*)$", core)
    if ordered_match:
        spaces = len(ordered_match.group(1).replace("\t", "    "))
        list_level = spaces // 2
        item_number = ordered_match.group(2)
        item_text = render_inline_markdown(ordered_match.group(3).strip())
        return indent_prefix(depth + 1 + list_level) + f"{item_number}. " + item_text, False

    return indent_prefix(depth + 1) + render_inline_markdown(core), False


def render_persona_two_columns(persona: dict) -> list[str]:
    items = [(escape_latex(str(k)), escape_latex(str(v))) for k, v in persona.items()]
    split_idx = (len(items) + 1) // 2
    left_items = items[:split_idx]
    right_items = items[split_idx:]

    out: list[str] = [
        r"\noindent\begin{minipage}[t]{0.48\textwidth}",
        r"\raggedright",
    ]
    for key, value in left_items:
        out.append(rf"\textbf{{{key}}}: {value}\par")
    out.extend(
        [
            r"\end{minipage}\hfill",
            r"\begin{minipage}[t]{0.48\textwidth}",
            r"\raggedright",
        ]
    )
    for key, value in right_items:
        out.append(rf"\textbf{{{key}}}: {value}\par")
    out.append(r"\end{minipage}")
    return out


def render_turn(role: str, turn_number: int, content: str) -> list[str]:
    role_l = (role or "").strip().lower()
    if role_l == "user":
        role_label = "User"
    elif role_l == "assistant":
        role_label = "Assistant"
    else:
        role_label = (role or "Message").strip().title() or "Message"

    out: list[str] = [
        rf"\noindent\textbf{{{escape_latex(role_label)} (Turn {turn_number}):}}",
        r"\vspace{2pt}",
    ]

    lines = content.splitlines()
    if not lines:
        out.append(r"\hspace*{1.2em}\par")
    else:
        pending_gap = False
        for line in lines:
            if line.strip() == "":
                pending_gap = True
                continue

            converted, is_heading = convert_markdown_line_to_latex(line)
            if converted == "":
                continue

            if pending_gap:
                out.append(r"\vspace{2pt}")
                pending_gap = False

            if is_heading:
                out.append(rf"\noindent {converted}\par")
            else:
                out.append(rf"{converted}\par")

    out.append(r"\vspace{6pt}")
    return out


def render_example(example_idx: int, example: dict) -> list[str]:
    persona = example.get("persona", {})
    full_conversation = example.get("fullconversation", [])

    out: list[str] = [
        r"\begin{tcolorbox}[",
        r"    enhanced,",
        r"    breakable,",
        r"    colback=white,",
        r"    colframe=black,",
        r"    arc=10pt,",
        r"    boxrule=2pt,",
        r"    left=10pt,",
        r"    right=10pt,",
        r"    top=10pt,",
        r"    bottom=10pt",
        r"]",
        "",
        rf"\noindent\colorbox{{sectionbg}}{{\parbox{{\dimexpr\textwidth-2\fboxsep}}{{\textbf{{\large Example {example_idx}: Persona}}}}}}",
        "",
        r"\vspace{8pt}",
        r"\small",
    ]

    out.extend(render_persona_two_columns(persona))

    out.extend(
        [
            "",
            r"\vspace{12pt}",
            r"\noindent\colorbox{sectionbg}{\parbox{\dimexpr\textwidth-2\fboxsep}{\textbf{\large Full Conversation}}}",
            "",
            r"\vspace{8pt}",
            r"\setlength{\parindent}{0pt}",
        ]
    )

    for turn_idx, turn in enumerate(full_conversation, start=1):
        out.extend(
            render_turn(
                role=str(turn.get("role", "Message")),
                turn_number=turn_idx,
                content=str(turn.get("content", "")),
            )
        )

    out.extend([r"\normalsize", r"\end{tcolorbox}"])
    return out


def main() -> None:
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))

    all_lines: list[str] = []
    for i, example in enumerate(data, start=1):
        all_lines.extend(render_example(i, example))
        if i != len(data):
            all_lines.extend(["", r"\vspace{10pt}", ""])

    OUTPUT_TEX.write_text("\n".join(all_lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    print(f"Rendered {len(data)} examples -> {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
