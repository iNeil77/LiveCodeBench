"""Helpers for printing evaluation metrics as a clean, bordered table."""


def _fmt_pct(value: float) -> str:
    """Format a pass rate as a percentage. Values are treated as fractions in
    [0, 1] unless they already look like a percentage (> 1)."""
    pct = value * 100 if value <= 1.0 else value
    return f"{pct:6.2f}%"


def render_metrics_table(
    pass_at_k: dict[str, float],
    title: str | None = None,
    num_problems: int | None = None,
) -> str:
    """Render a {"pass@k": value, ...} mapping as a bordered ASCII table.

    Non ``pass@k`` keys (e.g. the nested ``"detail"`` entry) are ignored.
    """
    rows = [
        (k, v)
        for k, v in pass_at_k.items()
        if k.startswith("pass@") and isinstance(v, (int, float))
    ]
    # sort by the numeric k
    rows.sort(key=lambda kv: int(kv[0].split("@")[1]))

    metric_w = max([len("Metric")] + [len(k) for k, _ in rows])
    score_w = max([len("Score")] + [len(_fmt_pct(v)) for _, v in rows])

    # Width of a data row's inner content (between the outer "| " and " |").
    data_inner = metric_w + score_w + 3  # " | " column separator

    # Header banner lines (title / subtitle) may be wider than the data rows;
    # grow the box to fit the widest line so all borders line up.
    banners = []
    if title:
        banners.append(title)
    if num_problems is not None:
        banners.append(f"{num_problems} problems")

    inner = max([data_inner] + [len(b) for b in banners])
    border = "+" + "-" * (inner + 2) + "+"

    def row(content_builder):
        return "| " + content_builder + " |"

    lines = [border]
    for b in banners:
        lines.append(row(b.center(inner)))
    if banners:
        lines.append(border)

    # Pad the metric column so metric_w + 3 + score_w == inner.
    pad_metric_w = inner - score_w - 3
    lines.append(row(f"{'Metric'.ljust(pad_metric_w)} | {'Score'.rjust(score_w)}"))
    lines.append(border)
    for k, v in rows:
        lines.append(row(f"{k.ljust(pad_metric_w)} | {_fmt_pct(v).rjust(score_w)}"))
    lines.append(border)
    return "\n".join(lines)


def print_metrics_table(
    pass_at_k: dict[str, float],
    title: str | None = None,
    num_problems: int | None = None,
) -> None:
    print(render_metrics_table(pass_at_k, title=title, num_problems=num_problems))


def render_grid(
    headers: list[str],
    rows: list[list[str]],
    title: str | None = None,
) -> str:
    """Render a generic bordered table with multiple columns.

    ``headers`` is the column header row; ``rows`` is a list of equal-length
    string rows. The first column is left-aligned, the rest right-aligned.
    """
    ncol = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(ncol):
            widths[i] = max(widths[i], len(str(r[i])))

    def fmt_row(cells):
        parts = []
        for i, c in enumerate(cells):
            c = str(c)
            parts.append(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]))
        return "| " + " | ".join(parts) + " |"

    inner = sum(widths) + 3 * (ncol - 1)  # " | " separators between columns
    border = "+" + "-" * (inner + 2) + "+"

    lines = [border]
    if title:
        lines.append("| " + title.center(inner) + " |")
        lines.append(border)
    lines.append(fmt_row(headers))
    lines.append(border)
    for r in rows:
        lines.append(fmt_row(r))
    lines.append(border)
    return "\n".join(lines)
