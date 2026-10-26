"""
PyClean: a beginner-friendly CSV cleaning app.

Run it with:  streamlit run app.py

This file has two parts:
  1. Cleaning logic: plain Pandas functions with no Streamlit code,
     so they can be tested on their own.
  2. The app: the Streamlit page that calls those functions.
"""

import io
from pathlib import Path

import pandas as pd
import streamlit as st


# =============================================================================
# Part 1: Cleaning logic
# =============================================================================


def text_columns(df: pd.DataFrame) -> list:
    """Columns that Pandas stored as text."""
    return [
        col
        for col in df.columns
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
    ]


def numeric_columns(df: pd.DataFrame) -> list:
    """Columns that Pandas read as numbers (True/False columns are left out)."""
    return [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
    ]


def has_extra_spaces(value) -> bool:
    """True for text with spaces at the start or end, like '  Alice ' (but not '   ')."""
    return isinstance(value, str) and value.strip() != "" and value != value.strip()


def is_whitespace_only(value) -> bool:
    """True for text that is empty or only spaces, like '   '."""
    return isinstance(value, str) and value.strip() == ""


def format_number(x) -> str:
    """68500.0 -> '68,500'   37.5 -> '37.5'"""
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def read_csv_file(raw_bytes: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes. Tries UTF-8 first, then Latin-1 (common in Excel exports)."""
    try:
        return pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1")


def diagnose(df: pd.DataFrame):
    """Build a per-column health report and overall totals. Does not change df."""
    numeric = set(numeric_columns(df))
    text = set(text_columns(df))

    rows = []
    for col in df.columns:
        values = df[col].tolist()
        if col in numeric:
            col_type = "number"
        elif col in text:
            col_type = "text"
        else:
            col_type = str(df[col].dtype)
        rows.append(
            {
                "Column": col,
                "Type": col_type,
                "Missing": int(df[col].isna().sum()),
                "Extra spaces": sum(has_extra_spaces(v) for v in values),
                "Whitespace-only": sum(is_whitespace_only(v) for v in values),
            }
        )
    report = pd.DataFrame(rows)

    # A row is "fully empty" when every cell is missing.
    # Rows made only of missing and space-only cells become fully empty after trimming.
    empty_rows = int(df.isna().all(axis=1).sum())
    blank_or_missing = df.isna() | df.map(is_whitespace_only)
    empty_after_trim = int(blank_or_missing.all(axis=1).sum()) - empty_rows

    totals = {
        "missing": int(df.isna().sum().sum()),
        "empty_rows": empty_rows,
        "empty_after_trim": empty_after_trim,
        "duplicates": int(df.duplicated().sum()),
        "extra_spaces": int(report["Extra spaces"].sum()),
        "whitespace_only": int(report["Whitespace-only"].sum()),
    }
    return report, totals


def clean_data(
    original: pd.DataFrame,
    trim_whitespace: bool = False,
    remove_empty_rows: bool = False,
    remove_duplicates: bool = False,
    fill_numeric_columns=(),
    fill_text_columns=(),
    text_fill_value: str = "",
):
    """Return (cleaned_copy, summary). The original DataFrame is never modified.

    Steps always run in this order:
      1. Trim whitespace; cells with only spaces become missing
      2. Remove fully empty rows (every column missing)
      3. Remove duplicate rows
      4. Fill missing values: numbers with the median (calculated now, after
         rows were removed), text with the value the user typed
    """
    text_fill_value = text_fill_value.strip()
    if fill_text_columns and not text_fill_value:
        raise ValueError("The text replacement value can't be blank.")

    df = original.copy(deep=True)
    summary = {
        "rows_before": len(df),
        "cells_trimmed": 0,
        "whitespace_to_missing": 0,
        "empty_rows_removed": 0,
        "duplicates_removed": 0,
        "numbers_filled": 0,
        "text_filled": 0,
        "medians": {},  # column -> median used
        "filled": [],   # one row per filled column, for the review table
        "skipped": [],  # columns that couldn't be filled, with the reason
    }

    # Step 1: trim spaces in text columns. Whitespace-only cells become missing.
    if trim_whitespace:
        for col in text_columns(df):
            new_values = []
            for value in df[col].tolist():
                if is_whitespace_only(value):
                    new_values.append(float("nan"))
                    summary["whitespace_to_missing"] += 1
                elif has_extra_spaces(value):
                    new_values.append(value.strip())
                    summary["cells_trimmed"] += 1
                else:
                    new_values.append(value)
            # Keep the column's type: text stays text, nothing is converted to numbers.
            df[col] = pd.Series(new_values, index=df.index, dtype=df[col].dtype)

    # Step 2: remove rows where every column is missing.
    if remove_empty_rows:
        empty = df.isna().all(axis=1)
        summary["empty_rows_removed"] = int(empty.sum())
        df = df[~empty]

    # Step 3: remove rows that are exact copies of an earlier row.
    if remove_duplicates:
        rows_before_dedupe = len(df)
        df = df.drop_duplicates()
        summary["duplicates_removed"] = rows_before_dedupe - len(df)

    df = df.reset_index(drop=True)

    # Step 4a: fill missing numbers with the median of what's left.
    numeric_now = numeric_columns(df)
    for col in fill_numeric_columns:
        if col not in numeric_now:
            summary["skipped"].append(
                {"Column": col, "Reason": "It isn't a numeric column, so it was left as is."}
            )
            continue
        if df[col].notna().sum() == 0:
            summary["skipped"].append(
                {
                    "Column": col,
                    "Reason": "Every value is missing, so there's no median to calculate. "
                    "The column was left as is.",
                }
            )
            continue
        median = df[col].median()
        missing = int(df[col].isna().sum())
        df[col] = df[col].fillna(median)
        summary["medians"][col] = median
        summary["numbers_filled"] += missing
        summary["filled"].append(
            {"Column": col, "Type": "number", "Filled with": f"Median: {format_number(median)}", "Cells filled": missing}
        )

    # Step 4b: fill missing text with the user's value. Existing values are never touched.
    text_now = text_columns(df)
    for col in fill_text_columns:
        if col not in text_now:
            summary["skipped"].append(
                {"Column": col, "Reason": "It isn't a text column, so it was left as is."}
            )
            continue
        missing = int(df[col].isna().sum())
        df[col] = df[col].fillna(text_fill_value)
        summary["text_filled"] += missing
        summary["filled"].append(
            {"Column": col, "Type": "text", "Filled with": f'"{text_fill_value}"', "Cells filled": missing}
        )

    summary["rows_after"] = len(df)
    return df, summary


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Turn a DataFrame into CSV bytes for downloading. Missing values become empty cells."""
    return df.to_csv(index=False).encode("utf-8")


# =============================================================================
# Part 2: The Streamlit app
# =============================================================================

PREVIEW_ROWS = 20
TABLE_HEIGHT = 280  # keeps previews compact; the table scrolls inside

# Palette: navy text, teal accent, light-gray surfaces.
NAVY = "#1B2A41"
TEAL = "#0F766E"
GRAY_BG = "#F3F5F8"
BORDER = "#DCE2EA"
MUTED = "#5B6B7F"

# Streamlit's toolbar is a fixed 3.75rem strip at the top of the page. The page's
# top padding must clear it, or the header slides underneath and looks clipped.
TOOLBAR_HEIGHT = "3.75rem"

CSS = f"""
<style>
[data-testid="stMainBlockContainer"], .block-container {{
    padding-top: calc({TOOLBAR_HEIGHT} + 1rem); padding-bottom: 2rem; max-width: 1200px;
}}
[data-testid="stVerticalBlock"] {{ gap: 0.65rem; }}

.pc-header {{
    background: {NAVY}; border-left: 6px solid {TEAL}; border-radius: 10px;
    padding: 0.9rem 1.2rem; overflow: visible;
}}
.pc-title {{
    display: flex; align-items: center; flex-wrap: wrap; gap: 0.45rem;
    color: #FFFFFF; font-size: clamp(1.3rem, 4vw, 1.6rem); font-weight: 700;
    line-height: 1.3; margin: 0; padding: 0;
}}
.pc-subtitle {{ color: #D6E4F0; font-size: 0.95rem; line-height: 1.4; margin-top: 0.2rem; }}

.pc-section {{
    display: flex; align-items: center; gap: 0.55rem;
    color: {NAVY}; font-size: 1.1rem; font-weight: 650;
}}
.pc-num {{
    background: {TEAL}; color: #FFFFFF; border-radius: 50%;
    width: 1.55rem; height: 1.55rem; font-size: 0.85rem;
    display: inline-flex; align-items: center; justify-content: center;
}}
.pc-group {{ color: {NAVY}; font-weight: 600; font-size: 0.95rem; }}

[data-testid="stMetric"] {{
    background: {GRAY_BG}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 0.5rem 0.8rem;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.8rem; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; font-size: 1.35rem; }}
</style>
"""

# Everything remembered about the current file. Cleared when a new file is uploaded.
FILE_STATE_KEYS = ["original", "load_error", "result"]
OPTION_KEYS = [
    "opt_trim", "opt_empty", "opt_dedupe",
    "opt_fill_num", "opt_fill_num_cols",
    "opt_fill_text", "opt_fill_text_cols", "opt_text_value",
]


def start_fresh(file_key):
    """Forget the previous file's data, results, and cleaning choices."""
    for key in FILE_STATE_KEYS + OPTION_KEYS:
        st.session_state.pop(key, None)
    st.session_state["file_key"] = file_key


def load_file(uploaded):
    """Read the uploaded file once and store it in session state."""
    try:
        st.session_state["original"] = read_csv_file(uploaded.getvalue())
    except pd.errors.EmptyDataError:
        st.session_state["load_error"] = "This file is empty, so there's nothing to clean."
    except (pd.errors.ParserError, ValueError) as err:
        st.session_state["load_error"] = (
            f"PyClean couldn't read this file as a CSV. Check that it's a valid CSV and try again.\n\nDetails: {err}"
        )


def section(number, title):
    st.markdown(
        f'<div class="pc-section"><span class="pc-num">{number}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def metric_cards(items, per_row=4):
    """Show (label, value) pairs as compact cards, up to `per_row` per line."""
    for start in range(0, len(items), per_row):
        chunk = items[start : start + per_row]
        for column, (label, value) in zip(st.columns(per_row), chunk):
            column.metric(label, value)


def show_upload_and_preview():
    """Section 1. Returns (uploaded_file, original_df), or None if there's nothing to work with."""
    with st.container(border=True):
        section(1, "Upload")
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")

        if uploaded is None:
            start_fresh(None)
            st.caption("Upload a CSV file to get started. Your original data is never changed.")
            return None

        # A new file means a clean slate: old results and choices are thrown away.
        file_key = getattr(uploaded, "file_id", None) or f"{uploaded.name}-{uploaded.size}"
        if st.session_state.get("file_key") != file_key:
            start_fresh(file_key)
            load_file(uploaded)

        if st.session_state.get("load_error"):
            st.error(st.session_state["load_error"])
            return None

        original = st.session_state["original"]
        if original.empty:
            st.warning("This file has column headers but no data rows, so there's nothing to clean.")
            return None

        st.caption(
            f"**{uploaded.name}** · {len(original):,} rows × {len(original.columns):,} columns · "
            f"showing the first {min(PREVIEW_ROWS, len(original))} rows"
        )
        st.dataframe(original.head(PREVIEW_ROWS), height=TABLE_HEIGHT)
        return uploaded, original


def show_quality_check(df):
    with st.container(border=True):
        section(2, "Data Quality Check")
        report, totals = diagnose(df)

        metric_cards(
            [
                ("Missing values", totals["missing"]),
                ("Fully empty rows", totals["empty_rows"]),
                ("Duplicate rows", totals["duplicates"]),
                ("Cells with extra spaces", totals["extra_spaces"]),
                ("Whitespace-only cells", totals["whitespace_only"]),
            ],
            per_row=5,
        )

        notes = []
        if totals["empty_after_trim"]:
            notes.append(
                f"{totals['empty_after_trim']} more row(s) contain only spaces and blanks; "
                "they become fully empty after trimming."
            )
        if totals["duplicates"] or totals["extra_spaces"]:
            notes.append(
                "Duplicates are exact matches. Rows that differ only by extra spaces are "
                "caught when trimming and duplicate removal are both selected."
            )
        if not any(totals.values()):
            st.success("No missing values, empty rows, duplicates, or extra whitespace found.")
        for note in notes:
            st.caption(note)

        with st.expander("Details by column"):
            st.dataframe(report, hide_index=True)


def show_cleaning_actions(df):
    """Section 3. Returns the user's current choices."""
    with st.container(border=True):
        section(3, "Cleaning Actions")
        st.caption(
            "All actions are optional. They run in this order: trim → empty rows → duplicates → fill."
        )
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown('<div class="pc-group">Fix rows and text</div>', unsafe_allow_html=True)
            trim = st.checkbox(
                "Trim extra whitespace",
                key="opt_trim",
                help="Removes spaces at the start and end of text. Cells with only spaces become missing.",
            )
            drop_empty = st.checkbox(
                "Remove fully empty rows",
                key="opt_empty",
                help="Removes a row only if every column is missing. Runs after trimming.",
            )
            dedupe = st.checkbox(
                "Remove duplicate rows",
                key="opt_dedupe",
                help="Keeps the first copy of each row and removes exact repeats.",
            )

        with right:
            st.markdown('<div class="pc-group">Fill missing values</div>', unsafe_allow_html=True)

            fill_num = st.checkbox(
                "Numbers: fill with the median",
                key="opt_fill_num",
                help="The median is calculated after trimming and row removal.",
            )
            fill_num_cols = []
            if fill_num:
                num_cols = numeric_columns(df)
                if num_cols:
                    fill_num_cols = st.multiselect("Numeric columns", num_cols, key="opt_fill_num_cols")
                    st.caption(
                        "Only columns read as numbers are listed. Text columns are never converted to numbers."
                    )
                else:
                    st.warning("This file has no numeric columns.")

            fill_text = st.checkbox(
                "Text: fill with a value",
                key="opt_fill_text",
                help="Replaces only missing text cells. Existing values are never changed.",
            )
            fill_text_cols, text_value = [], ""
            if fill_text:
                txt_cols = text_columns(df)
                if txt_cols:
                    fill_text_cols = st.multiselect("Text columns", txt_cols, key="opt_fill_text_cols")
                    text_value = st.text_input(
                        "Replacement value", value="Unknown", key="opt_text_value"
                    ).strip()
                else:
                    st.warning("This file has no text columns.")

    return {
        "trim": trim,
        "drop_empty": drop_empty,
        "dedupe": dedupe,
        "fill_num": fill_num,
        "fill_num_cols": tuple(fill_num_cols),
        "fill_text": fill_text,
        "fill_text_cols": tuple(fill_text_cols),
        "text_value": text_value,
    }


def problems_with(settings):
    """Reasons the current choices can't be applied yet."""
    problems = []
    if settings["fill_num"] and not settings["fill_num_cols"]:
        problems.append("Pick at least one numeric column, or untick number filling.")
    if settings["fill_text"] and not settings["fill_text_cols"]:
        problems.append("Pick at least one text column, or untick text filling.")
    if settings["fill_text"] and settings["fill_text_cols"] and not settings["text_value"]:
        problems.append("Enter a replacement value for text (it can't be blank).")
    return problems


def show_review(original, result):
    settings, summary, cleaned = result["settings"], result["summary"], result["cleaned"]

    items = [("Rows", f"{summary['rows_before']:,} → {summary['rows_after']:,}")]
    if settings["trim"]:
        items.append(("Cells trimmed", summary["cells_trimmed"]))
        items.append(("Whitespace-only → missing", summary["whitespace_to_missing"]))
    if settings["drop_empty"]:
        items.append(("Empty rows removed", summary["empty_rows_removed"]))
    if settings["dedupe"]:
        items.append(("Duplicate rows removed", summary["duplicates_removed"]))
    if settings["fill_num_cols"]:
        items.append(("Numbers filled", summary["numbers_filled"]))
    if settings["fill_text_cols"]:
        items.append(("Text cells filled", summary["text_filled"]))
    metric_cards(items)

    if summary["filled"]:
        st.markdown('<div class="pc-group">Filled values by column</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(summary["filled"]), hide_index=True)
    for skipped in summary["skipped"]:
        st.warning(f"**{skipped['Column']}** was skipped. {skipped['Reason']}")
    if summary["whitespace_to_missing"]:
        st.caption(
            "Missing counts can rise after trimming, because cells with only spaces now count as missing."
        )

    with st.expander("Missing values by column (before vs. after)"):
        missing = pd.DataFrame(
            {"Before": original.isna().sum(), "After": cleaned.isna().sum()}
        ).rename_axis("Column").reset_index()
        st.dataframe(missing, hide_index=True)

    st.markdown('<div class="pc-group">Cleaned data preview</div>', unsafe_allow_html=True)
    st.dataframe(cleaned.head(PREVIEW_ROWS), height=TABLE_HEIGHT)
    st.caption(f"Showing the first {min(PREVIEW_ROWS, len(cleaned))} of {len(cleaned):,} rows.")


def main():
    st.set_page_config(page_title="PyClean", page_icon="🧹", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="pc-header"><div class="pc-title"><span>🧹</span><span>PyClean</span></div>'
        '<div class="pc-subtitle">Inspect, clean, and export your data.</div></div>',
        unsafe_allow_html=True,
    )

    loaded = show_upload_and_preview()
    if loaded is None:
        return
    uploaded, original = loaded

    show_quality_check(original)
    settings = show_cleaning_actions(original)

    # Apply lives with the actions it applies.
    nothing_chosen = not (
        settings["trim"] or settings["drop_empty"] or settings["dedupe"]
        or settings["fill_num"] or settings["fill_text"]
    )
    problems = problems_with(settings)
    for problem in problems:
        st.info(problem)
    if st.button("Apply cleaning", type="primary", disabled=nothing_chosen or bool(problems)):
        cleaned, summary = clean_data(
            original,
            trim_whitespace=settings["trim"],
            remove_empty_rows=settings["drop_empty"],
            remove_duplicates=settings["dedupe"],
            fill_numeric_columns=list(settings["fill_num_cols"]),
            fill_text_columns=list(settings["fill_text_cols"]),
            text_fill_value=settings["text_value"],
        )
        st.session_state["result"] = {"settings": settings, "cleaned": cleaned, "summary": summary}

    result = st.session_state.get("result")
    is_current = result is not None and result["settings"] == settings

    with st.container(border=True):
        section(4, "Review Results")
        if result is None:
            st.caption("Choose at least one action, then click **Apply cleaning** to see the changes.")
        elif not is_current:
            st.warning(
                "Your cleaning choices changed. Click **Apply cleaning** again to update the results."
            )
        else:
            show_review(original, result)

    with st.container(border=True):
        section(5, "Download")
        if is_current:
            st.download_button(
                "Download cleaned CSV",
                data=to_csv_bytes(result["cleaned"]),
                file_name=f"{Path(uploaded.name).stem}_cleaned.csv",
                mime="text/csv",
                type="primary",
            )
            st.caption("Missing values are saved as empty cells.")
        else:
            st.caption("Available after you apply your current cleaning choices.")


if __name__ == "__main__":
    main()
