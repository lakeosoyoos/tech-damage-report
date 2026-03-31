"""
Tech Damage Report Generator — Streamlit App
=============================================
Generate automated OTDR damage reports from SOR files.

Launch:  streamlit run app.py
"""

import os
import sys
import tempfile
import io
from contextlib import redirect_stdout

import streamlit as st
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tech_damage_report import (
    load_all_fibers, discover_splices, collect_eof_positions,
    detect_breaks_by_ribbon, detect_bend_zones, detect_trace_bend_zones,
    analyze_fiber, apply_ribbon_consensus_to_missed, write_xlsx,
    RIBBON_SIZE,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tech Damage Report",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background-color: #4BA82E !important;
        border-color: #4BA82E !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover {
        background-color: #3D8C24 !important;
        border-color: #3D8C24 !important;
    }
    .stButton > button,
    .stDownloadButton > button {
        border-color: #4BA82E !important;
        color: #4BA82E !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #3D8C24 !important;
        color: #3D8C24 !important;
    }
    .stProgress > div > div > div > div {
        background-color: #4BA82E !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white !important;
    }
    .stRadio [role="radiogroup"] label[data-checked="true"],
    .stRadio [role="radiogroup"] label:has(input:checked) {
        background-color: #4BA82E !important;
        border-color: #4BA82E !important;
        color: white !important;
    }
    .stRadio [role="radiogroup"] label[data-checked="true"] p,
    .stRadio [role="radiogroup"] label:has(input:checked) p {
        color: white !important;
    }
    .stRadio [role="radiogroup"] label {
        border-color: #4BA82E !important;
    }
    a { color: #4BA82E !important; }
</style>
""", unsafe_allow_html=True)

st.title("Tech Damage Report Generator")
st.caption("Automated OTDR damage report — breaks, splices, bends, and B-direction fill")


# ── Password protection ──────────────────────────────────────────────────────

def check_password():
    """Return True if user entered the correct password."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    try:
        correct = st.secrets["passwords"]["app_password"]
    except (KeyError, FileNotFoundError):
        return True
    pwd = st.text_input("Enter password", type="password", key="pwd_input")
    if pwd:
        if pwd == correct:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False

if not check_password():
    st.stop()


# ── Session state ────────────────────────────────────────────────────────────

for key in ["xlsx_bytes", "xlsx_name", "summary", "log_output", "done"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload SOR Files")

    input_method = st.radio(
        "Input method",
        ["Upload ZIP", "Browse files", "Folder path"],
        index=0,
        horizontal=True,
    )

    uploaded_a = None
    uploaded_b = None
    zip_a = None
    zip_b = None
    folder_a = None
    folder_b = None

    if input_method == "Upload ZIP":
        zip_a = st.file_uploader(
            "A-direction ZIP",
            type=["zip"],
            accept_multiple_files=False,
            key=f"zip_a_{st.session_state.upload_key}",
        )
        if zip_a:
            st.caption(f"A: {zip_a.name} ({zip_a.size / 1024:.0f} KB)")
        zip_b = st.file_uploader(
            "B-direction ZIP",
            type=["zip"],
            accept_multiple_files=False,
            key=f"zip_b_{st.session_state.upload_key}",
        )
        if zip_b:
            st.caption(f"B: {zip_b.name} ({zip_b.size / 1024:.0f} KB)")
    elif input_method == "Browse files":
        uploaded_a = st.file_uploader(
            "A-direction SOR files",
            type=["sor"],
            accept_multiple_files=True,
            key=f"upload_a_{st.session_state.upload_key}",
        )
        uploaded_b = st.file_uploader(
            "B-direction SOR files",
            type=["sor"],
            accept_multiple_files=True,
            key=f"upload_b_{st.session_state.upload_key}",
        )
    else:
        folder_a = st.text_input(
            "A-direction folder path",
            value=st.session_state.get("folder_a", ""),
            placeholder="/Users/you/Desktop/A Direction/",
        )
        if folder_a:
            folder_a = folder_a.strip().strip("'\"")
            st.session_state.folder_a = folder_a
            if os.path.isdir(folder_a):
                n = len([f for f in os.listdir(folder_a) if f.lower().endswith('.sor')])
                st.caption(f"Found {n} .sor files")
            else:
                st.warning("Folder not found")

        folder_b = st.text_input(
            "B-direction folder path",
            value=st.session_state.get("folder_b", ""),
            placeholder="/Users/you/Desktop/B Direction/",
        )
        if folder_b:
            folder_b = folder_b.strip().strip("'\"")
            st.session_state.folder_b = folder_b
            if os.path.isdir(folder_b):
                n = len([f for f in os.listdir(folder_b) if f.lower().endswith('.sor')])
                st.caption(f"Found {n} .sor files")
            elif folder_b.strip():
                st.warning("Folder not found")

    if st.button("Clear All", use_container_width=True):
        old_key = st.session_state.upload_key
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.upload_key = old_key + 1
        st.rerun()

    st.divider()
    st.subheader("Settings")

    site_a = st.text_input("Site A name", value="A")
    site_b = st.text_input("Site B name", value="B")
    threshold = st.number_input("Unidir threshold (dB)", value=0.15,
                                format="%.3f", step=0.01)
    bidi_threshold = st.number_input("Bidir threshold (dB)", value=0.30,
                                     format="%.3f", step=0.01)
    bend_threshold = st.number_input("Bend threshold (dB/km)", value=0.05,
                                      format="%.3f", step=0.01)
    ribbon_size = st.number_input("Fibers per ribbon", value=RIBBON_SIZE,
                                   min_value=1, max_value=24, step=1)

    has_a = (bool(uploaded_a) or bool(zip_a) or
             (folder_a and os.path.isdir(folder_a)))
    has_b = (bool(uploaded_b) or bool(zip_b) or
             (folder_b and os.path.isdir(folder_b)))
    run_button = st.button("Generate Report", type="primary",
                           use_container_width=True, disabled=not (has_a and has_b))
    if has_a and not has_b:
        st.warning("Both A and B directions are required")


# ── Helpers ──────────────────────────────────────────────────────────────────

def stage_files(uploaded, prefix="sor_"):
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    for uf in uploaded:
        fp = os.path.join(tmpdir, uf.name)
        with open(fp, 'wb') as f:
            f.write(uf.getbuffer())
    return tmpdir


def stage_zip(uploaded_zip, prefix="sor_zip_"):
    """Extract SOR files from a ZIP to a temp directory."""
    import zipfile
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.getbuffer()), 'r') as zf:
        for name in zf.namelist():
            if name.lower().endswith('.sor') and not name.startswith('__MACOSX'):
                basename = os.path.basename(name)
                if not basename:
                    continue
                fp = os.path.join(tmpdir, basename)
                with zf.open(name) as src, open(fp, 'wb') as dst:
                    dst.write(src.read())
    return tmpdir


# ── Run ──────────────────────────────────────────────────────────────────────

if run_button and has_a and has_b:
    # Get directories
    if folder_a and os.path.isdir(folder_a):
        dir_a = folder_a
        dir_b = folder_b if (folder_b and os.path.isdir(folder_b)) else None
    elif zip_a:
        progress = st.progress(0.0, text="Extracting A-direction ZIP...")
        dir_a = stage_zip(zip_a, "td_a_")
        progress.progress(0.03, text="Extracting B-direction ZIP...")
        dir_b = stage_zip(zip_b, "td_b_") if zip_b else None
        progress.progress(0.05, text="Files extracted.")
        progress.empty()
    else:
        progress = st.progress(0.0, text="Staging A-direction files...")
        dir_a = stage_files(uploaded_a, "td_a_")
        progress.progress(0.03, text="Staging B-direction files...")
        dir_b = stage_files(uploaded_b, "td_b_") if uploaded_b else None
        progress.progress(0.05, text="Files staged.")
        progress.empty()

    bar = st.progress(0.05, text="Loading SOR files (event table + RawSamples)...")
    log_buf = io.StringIO()

    with redirect_stdout(log_buf):
        fibers_a, fibers_b = load_all_fibers(dir_a, dir_b)

    n_fibers = max(max(fibers_a.keys(), default=0),
                   max(fibers_b.keys(), default=0))
    trace_a = sum(1 for r in fibers_a.values() if r['has_trace'])
    trace_b = sum(1 for r in fibers_b.values() if r['has_trace'])

    bar.progress(0.20, text=f"Loaded {len(fibers_a)} A ({trace_a} traces) + {len(fibers_b)} B ({trace_b} traces)...")

    # Discover splices
    with redirect_stdout(log_buf):
        splices = discover_splices(fibers_a, n_fibers)
    n_real_splices = sum(1 for sp in splices if not sp['is_bend'])
    n_bend_cols = sum(1 for sp in splices if sp['is_bend'])

    # Span
    eof_vals = []
    for r in fibers_a.values():
        for e in r['events']:
            if e['is_end']:
                eof_vals.append(e['dist_km'])
                break
    span_km = round(float(np.median(eof_vals)) if eof_vals else 0, 2)

    bar.progress(0.30, text=f"Found {n_real_splices} splices + {n_bend_cols} bend columns, span {span_km} km...")

    # Break detection
    bar.progress(0.35, text="Detecting breaks by ribbon EOL comparison...")
    with redirect_stdout(log_buf):
        eof_a_all = collect_eof_positions(fibers_a)
        raw_breaks = detect_breaks_by_ribbon(eof_a_all, ribbon_size)

    bar.progress(0.40, text=f"Found {len(raw_breaks)} break(s). Detecting bend zones...")

    # Bend detection — segment slope
    with redirect_stdout(log_buf):
        all_bends = detect_bend_zones(fibers_a, splices, ribbon_size,
                                      bend_threshold, known_breaks=raw_breaks)

    bar.progress(0.50, text="Scanning RawSamples for localized bends...")

    # Bend detection — trace scan
    splice_kms = [sp['position_km'] for sp in splices]
    with redirect_stdout(log_buf):
        trace_bends, trace_bend_cols = detect_trace_bend_zones(
            fibers_a, ribbon_size, bend_threshold,
            known_event_kms=splice_kms, known_breaks=raw_breaks,
            fibers_b=fibers_b, span_km=span_km)

    # Merge trace bend columns
    new_cols_added = 0
    for bc in trace_bend_cols:
        bc_km = bc['position_km']
        if any(abs(sp['position_km'] - bc_km) < 0.8 for sp in splices):
            continue
        splices.append(bc)
        new_cols_added += 1
    if new_cols_added:
        splices.sort(key=lambda s: s['position_km'])
        snum = 0
        for sp in splices:
            if not sp['is_bend']:
                snum += 1
                sp['splice_num'] = snum

    # Merge trace bend data into all_bends
    for fnum, blist in trace_bends.items():
        if fnum not in all_bends:
            all_bends[fnum] = []
        existing_kms = {round(b.get('start_km', b.get('km', 0)), 1)
                        for b in all_bends[fnum]}
        for b in blist:
            if round(b['km'], 1) not in existing_kms:
                all_bends[fnum].append({
                    'start_km':     b['km'] - 0.5,
                    'end_km':       b['km'] + 0.5,
                    'excess_db_km': b['excess'],
                })

    bar.progress(0.60, text=f"Analyzing {n_fibers} fibers...")

    # Per-fiber analysis
    all_results = {}
    all_breaks = {}
    all_missed = {}
    fiber_list = sorted(set(list(fibers_a.keys()) + list(fibers_b.keys())))
    total = len(fiber_list)

    for i, fnum in enumerate(fiber_list):
        ra = fibers_a.get(fnum)
        rb = fibers_b.get(fnum)
        pre_break = raw_breaks.get(fnum)

        splice_results, break_a, noise_km, missed_a = analyze_fiber(
            fnum, ra, rb, splices, span_km,
            threshold,
            precomputed_break_km=pre_break)

        for rec in splice_results:
            all_results[(fnum, rec['splice_idx'])] = rec
        if break_a:
            all_breaks[fnum] = break_a
        if missed_a:
            all_missed[fnum] = missed_a

        # Update progress every 50 fibers
        if (i + 1) % 50 == 0 or (i + 1) == total:
            pct = 0.60 + 0.25 * ((i + 1) / total)
            bar.progress(pct, text=f"Analyzing fiber {fnum}/{n_fibers}...")

    n_flagged = sum(1 for r in all_results.values() if r['flagged'])
    n_bfill = sum(1 for r in all_results.values() if r['source'] == 'B_fill' and r['primary'])
    n_bend_fibs = len(all_bends)

    bar.progress(0.88, text="Applying ribbon consensus to missed events...")
    all_missed = apply_ribbon_consensus_to_missed(
        all_missed, ribbon_size=ribbon_size, min_fibers=2)
    n_missed_total = sum(len(v) for v in all_missed.values())

    bar.progress(0.92, text="Writing Excel report...")
    xlsx_tmpdir = tempfile.mkdtemp(prefix="td_xlsx_")
    xlsx_path = os.path.join(xlsx_tmpdir, "tech_damage_report.xlsx")
    with redirect_stdout(log_buf):
        write_xlsx(all_results, all_breaks, all_bends, all_missed, splices, n_fibers,
                   xlsx_path, site_a, site_b, span_km,
                   threshold, ribbon_size)

    with open(xlsx_path, 'rb') as f:
        st.session_state.xlsx_bytes = f.read()
    st.session_state.xlsx_name = f"tech_damage_{site_a}_{site_b}.xlsx"

    # Build summary
    summary_lines = [
        f"**Fibers:** {len(fibers_a)} A ({trace_a} with RawSamples), "
        f"{len(fibers_b)} B ({trace_b} with RawSamples)",
        f"**Splice closures:** {n_real_splices} + {n_bend_cols + new_cols_added} bend columns",
        f"**Span:** {span_km} km ({span_km * 3280.84:,.0f} ft)",
        f"**Thresholds:** unidir {threshold} dB, bidir {bidi_threshold} dB, bend {bend_threshold} dB/km",
        "",
        f"**Breaks detected:** {len(all_breaks)}",
    ]
    if all_breaks:
        for fnum in sorted(all_breaks.keys()):
            b = all_breaks[fnum]
            ribbon_idx = (fnum - 1) // ribbon_size
            cls = b.get('break_class', '')
            summary_lines.append(
                f"  - Fiber {fnum} (ribbon {ribbon_idx+1}): "
                f"{b['break_km']:.3f} km ({b['break_km']*3280.84:,.0f} ft) — {cls}"
            )
    summary_lines += [
        "",
        f"**Flagged splice events:** {n_flagged}",
        f"**B-fill entries:** {n_bfill}",
        f"**Fibers with bend zones:** {n_bend_fibs}",
        f"**OTDR-missed events:** {n_missed_total} across {len(all_missed)} fibers",
    ]

    st.session_state.summary = "\n\n".join(summary_lines)
    st.session_state.log_output = log_buf.getvalue()
    st.session_state.done = True

    bar.progress(1.0, text="Done!")
    bar.empty()


# ── Display ──────────────────────────────────────────────────────────────────

if st.session_state.get("done"):
    st.subheader("Report Complete")
    if st.session_state.summary:
        st.markdown(st.session_state.summary)

    st.divider()

    if st.session_state.xlsx_bytes:
        st.download_button(
            "Download Excel Report",
            st.session_state.xlsx_bytes,
            file_name=st.session_state.xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary",
        )

    with st.expander("Analysis Log"):
        st.code(st.session_state.log_output or "No log.", language=None)

else:
    st.info("Upload A and B direction SOR files in the sidebar, then click **Generate Report**.")
    st.markdown("""
    **How it works:**
    1. Upload **A-direction** and **B-direction** SOR files (both required)
    2. Set your site names, thresholds, and ribbon size
    3. Click **Generate Report**
    4. Download the Excel damage report

    **Report contents:**
    - Distance headers in km and feet from both directions (A→B and B→A)
    - Break detection with localization (crush, clean break, stress fracture)
    - Splice loss flagging (bidirectional + unidirectional)
    - B-direction fill past breaks (blue cells)
    - Bend zone detection from RawSamples traces (yellow columns)
    - OTDR-missed events detected by ribbon consensus
    """)
