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
    totals = {
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "extra_spaces": int(report["Extra spaces"].sum()),
        "whitespace_only": int(report["Whitespace-only"].sum()),
    }
    return report, totals


def clean_data(original: pd.DataFrame, trim_whitespace: bool, remove_duplicates: bool, fill_columns: list):
    """Return (cleaned_copy, summary). The original DataFrame is never modified.

    Steps always run in this order:
      1. Trim whitespace, so '  Ali' and 'Ali' become the same value
      2. Remove duplicate rows
      3. Fill missing numbers with the median, calculated AFTER steps 1 and 2
    """
    df = original.copy(deep=True)
    summary = {
        "rows_before": len(df),
        "cells_trimmed": 0,
        "whitespace_to_missing": 0,
        "duplicates_removed": 0,
        "filled": [],   # columns that were filled: name, median, how many cells
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
            # Keep the column's original type: text stays text, nothing is converted to numbers.
            df[col] = pd.Series(new_values, index=df.index, dtype=df[col].dtype)

    # Step 2: remove rows that are exact copies of an earlier row.
    if remove_duplicates:
        rows_before_dedupe = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        summary["duplicates_removed"] = rows_before_dedupe - len(df)

    # Step 3: fill missing values in the chosen numeric columns with that column's median.
    numeric_now = numeric_columns(df)
    for col in fill_columns:
        if col not in numeric_now:
            summary["skipped"].append(
                {"Column": col, "Reason": "It isn't a numeric column, so it was left as is."}
            )
            continue
        if df[col].notna().sum() == 0:
            summary["skipped"].append(
                {
                    "Column": col,
                    "Reason": "Every value in this column is missing, so there is no median "
                    "to calculate. The column was left as is.",
                }
            )
            continue
        median = df[col].median()
        missing = int(df[col].isna().sum())
        df[col] = df[col].fillna(median)
        summary["filled"].append({"Column": col, "Median used": median, "Cells filled": missing})

    summary["rows_after"] = len(df)
    return df, summary


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Turn a DataFrame into CSV bytes for downloading. Missing values become empty cells."""
    return df.to_csv(index=False).encode("utf-8")


# =============================================================================
# Part 2: The Streamlit app
# =============================================================================

PREVIEW_ROWS = 20

# Everything we remember about the current file. Cleared when a new file is uploaded.
FILE_STATE_KEYS = ["original", "load_error", "result"]
OPTION_KEYS = ["opt_trim", "opt_dedupe", "opt_fill", "opt_fill_cols"]


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


def show_preview(df):
    st.subheader("1. Preview")
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    st.dataframe(df.head(PREVIEW_ROWS))
    st.caption(f"Showing the first {min(PREVIEW_ROWS, len(df))} of {len(df):,} rows.")


def show_health_check(df):
    st.subheader("2. Data health check")
    report, totals = diagnose(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Missing values", totals["missing"])
    c2.metric("Duplicate rows", totals["duplicates"])
    c3.metric("Cells with extra spaces", totals["extra_spaces"])
    c4.metric("Whitespace-only cells", totals["whitespace_only"])

    if sum(totals.values()) == 0:
        st.success("No missing values, duplicate rows, or extra whitespace found.")
    else:
        st.caption(
            "Duplicates here are exact matches. Rows that differ only by extra spaces "
            "are caught when trimming and duplicate removal are both selected."
        )

    with st.expander("Details by column"):
        st.dataframe(report, hide_index=True)


def choose_actions(df):
    """Show the cleaning options and return the user's current choices."""
    st.subheader("3. Choose cleaning actions")

    trim = st.checkbox(
        "Trim extra whitespace",
        key="opt_trim",
        help="Removes spaces at the start and end of text cells. "
        "Cells that contain only spaces become missing values.",
    )
    dedupe = st.checkbox(
        "Remove duplicate rows",
        key="opt_dedupe",
        help="Keeps the first copy of each row and removes exact repeats. Runs after trimming.",
    )
    fill = st.checkbox(
        "Fill missing numbers with the median",
        key="opt_fill",
        help="Replaces missing values in the columns you pick with that column's median. "
        "The median is calculated after trimming and duplicate removal.",
    )

    fill_cols = []
    if fill:
        num_cols = numeric_columns(df)
        if num_cols:
            fill_cols = st.multiselect("Numeric columns to fill", num_cols, key="opt_fill_cols")
            st.caption(
                "Only columns Pandas read as numbers are listed. Text columns are never "
                "converted to numbers, so a number column with a stray text value won't appear here."
            )
        else:
            st.warning("This file has no numeric columns, so there's nothing to fill.")

    return {"trim": trim, "dedupe": dedupe, "fill": fill, "fill_cols": tuple(fill_cols)}


def show_review(original, result):
    st.subheader("4. Review changes")
    settings, summary, cleaned = result["settings"], result["summary"], result["cleaned"]

    items = [("Rows", f"{summary['rows_before']:,} → {summary['rows_after']:,}")]
    if settings["trim"]:
        items.append(("Cells trimmed", summary["cells_trimmed"]))
        items.append(("Whitespace-only → missing", summary["whitespace_to_missing"]))
    if settings["dedupe"]:
        items.append(("Duplicate rows removed", summary["duplicates_removed"]))
    if settings["fill_cols"]:
        items.append(("Missing numbers filled", sum(f["Cells filled"] for f in summary["filled"])))

    for column, (label, value) in zip(st.columns(len(items)), items):
        column.metric(label, value)

    if summary["filled"]:
        st.markdown("**Median fill**")
        st.dataframe(pd.DataFrame(summary["filled"]), hide_index=True)
    for skipped in summary["skipped"]:
        st.warning(f"**{skipped['Column']}** was skipped. {skipped['Reason']}")

    if summary["whitespace_to_missing"]:
        st.caption(
            "Missing values can go up after trimming, because cells that contained only "
            "spaces now count as missing."
        )

    with st.expander("Missing values by column (before vs. after)"):
        missing = pd.DataFrame(
            {"Before": original.isna().sum(), "After": cleaned.isna().sum()}
        ).rename_axis("Column").reset_index()
        st.dataframe(missing, hide_index=True)

    st.markdown("**Cleaned data preview**")
    st.dataframe(cleaned.head(PREVIEW_ROWS))
    st.caption(f"Showing the first {min(PREVIEW_ROWS, len(cleaned))} of {len(cleaned):,} rows.")


def main():
    st.set_page_config(page_title="PyClean", page_icon="🧹", layout="wide")
    st.title("🧹 PyClean")
    st.caption(
        "Upload a CSV, check it for common problems, choose what to fix, and download a "
        "clean copy. Your original data is never changed."
    )

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        start_fresh(None)
        st.info("Upload a CSV file to get started.")
        return

    # A new file means a clean slate: old results and choices are thrown away.
    file_key = getattr(uploaded, "file_id", None) or f"{uploaded.name}-{uploaded.size}"
    if st.session_state.get("file_key") != file_key:
        start_fresh(file_key)
        load_file(uploaded)

    if st.session_state.get("load_error"):
        st.error(st.session_state["load_error"])
        return

    original = st.session_state["original"]
    if original.empty:
        st.warning("This file has column headers but no data rows, so there's nothing to clean.")
        return

    show_preview(original)
    show_health_check(original)
    settings = choose_actions(original)

    nothing_chosen = not (settings["trim"] or settings["dedupe"] or settings["fill"])
    fill_incomplete = settings["fill"] and not settings["fill_cols"]
    if fill_incomplete:
        st.info("Pick at least one column to fill, or untick the median option.")

    if st.button("Apply cleaning", type="primary", disabled=nothing_chosen or fill_incomplete):
        cleaned, summary = clean_data(
            original, settings["trim"], settings["dedupe"], list(settings["fill_cols"])
        )
        st.session_state["result"] = {"settings": settings, "cleaned": cleaned, "summary": summary}

    result = st.session_state.get("result")
    if result is None:
        st.caption("Choose at least one action, then click **Apply cleaning** to see the changes.")
        return
    if result["settings"] != settings:
        st.warning(
            "Your cleaning choices changed. Click **Apply cleaning** again to update the "
            "results before downloading."
        )
        return

    show_review(original, result)

    st.subheader("5. Download")
    st.download_button(
        "Download cleaned CSV",
        data=to_csv_bytes(result["cleaned"]),
        file_name=f"{Path(uploaded.name).stem}_cleaned.csv",
        mime="text/csv",
        type="primary",
    )
    st.caption("Missing values are saved as empty cells.")


if __name__ == "__main__":
    main()
