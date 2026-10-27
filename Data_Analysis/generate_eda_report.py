"""Generate an HTML exploratory data analysis report for the freMTPL2 datasets.

Usage:
    python Data_Analysis/generate_eda_report.py
Output:
    Reports/EDA_Report.html
"""

import base64
import html
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
OUT_DIR = ROOT / "Reports"
OUT_FILE = OUT_DIR / "EDA_Report.html"

FREQ_FILE = DATA_DIR / "freMTPL2freq.csv"
SEV_FILE = DATA_DIR / "freMTPL2sev.csv"

PALETTE = plt.cm.tab20.colors
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})


# ----------------------------------------------------------------------------- helpers
def fig_to_html(fig, caption: str) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{img}" alt="{html.escape(caption)}"/>'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def table_html(df: pd.DataFrame, float_fmt: str = "{:,.4f}", index: bool = True) -> str:
    return df.to_html(
        classes="tbl", border=0, index=index, float_format=lambda x: float_fmt.format(x)
    )


def freq_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True).agg(
        policies=("IDpol", "count"),
        exposure=("Exposure", "sum"),
        claims=("ClaimNb", "sum"),
    )
    g["frequency"] = g["claims"] / g["exposure"]
    return g


def pie(series: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        series.values,
        labels=series.index.astype(str),
        autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
        startangle=90,
        colors=PALETTE[: len(series)],
        wedgeprops={"edgecolor": "white"},
    )
    ax.set_title(title)
    ax.axis("equal")
    return fig


def bar(series: pd.Series, title: str, xlabel: str, ylabel: str, horizontal=False, color="#1f77b4"):
    fig, ax = plt.subplots(figsize=(9, 5 if not horizontal else max(5, 0.3 * len(series))))
    labels = series.index.astype(str)
    if horizontal:
        ax.barh(labels, series.values, color=color)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        ax.bar(labels, series.values, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(title)
    return fig


def freq_exposure_plot(g: pd.DataFrame, title: str, xlabel: str, overall: float):
    """Exposure as bars (left axis) and claim frequency as a line (right axis)."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    labels = g.index.astype(str)
    ax1.bar(labels, g["exposure"], color="#aec7e8", label="Exposure (years)")
    ax1.set_ylabel("Exposure (policy-years)")
    ax1.set_xlabel(xlabel)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(labels, g["frequency"], color="#d62728", marker="o", label="Claim frequency")
    ax2.axhline(overall, color="grey", ls="--", lw=1, label="Portfolio frequency")
    ax2.set_ylabel("Claims per policy-year")
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax1.set_title(title)
    return fig


def hist(values, title: str, xlabel: str, bins=50, color="#1f77b4", log_y=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=bins, color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if log_y:
        ax.set_yscale("log")
    return fig


def section(title: str, body: str, sid: str) -> str:
    return f'<section id="{sid}"><h2>{title}</h2>{body}</section>'


def grid(*items: str) -> str:
    return '<div class="grid">' + "".join(f"<div>{i}</div>" for i in items) + "</div>"


# ----------------------------------------------------------------------------- load
def load():
    freq = pd.read_csv(FREQ_FILE)
    sev = pd.read_csv(SEV_FILE)
    for c in freq.select_dtypes(include="object").columns:
        freq[c] = freq[c].astype(str).str.strip("' ")
    return freq, sev


def overview(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": df.isna().sum(),
            "missing_%": df.isna().mean() * 100,
            "unique": df.nunique(),
        }
    )


# ----------------------------------------------------------------------------- report
def build_report():
    freq, sev = load()
    parts, toc = [], []

    def add(title, body, sid):
        parts.append(section(title, body, sid))
        toc.append(f'<li><a href="#{sid}">{title}</a></li>')

    num_cols = ["ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
    cat_cols = ["Area", "VehBrand", "VehGas", "Region"]

    total_exposure = freq["Exposure"].sum()
    total_claims_freq = freq["ClaimNb"].sum()
    overall_freq = total_claims_freq / total_exposure

    # Linking frequency and severity
    sev_agg = sev.groupby("IDpol").agg(n_sev=("ClaimAmount", "size"), total_amount=("ClaimAmount", "sum"))
    merged = freq.merge(sev_agg, on="IDpol", how="left")
    merged[["n_sev", "total_amount"]] = merged[["n_sev", "total_amount"]].fillna(0)
    sev_ids_not_in_freq = (~sev["IDpol"].isin(freq["IDpol"])).sum()
    claimants = merged[merged["ClaimNb"] > 0]
    claimnb_mismatch = (claimants["ClaimNb"] != claimants["n_sev"]).sum()
    sev_feat = sev.merge(freq, on="IDpol", how="inner")

    # ------------------------------------------------------------------ 1. Overview
    body = f"""
    <div class="cards">
      <div class="card"><span>{len(freq):,}</span>Policies (freq file)</div>
      <div class="card"><span>{len(sev):,}</span>Claims (sev file)</div>
      <div class="card"><span>{total_exposure:,.0f}</span>Policy-years of exposure</div>
      <div class="card"><span>{total_claims_freq:,}</span>Claims recorded (ClaimNb)</div>
      <div class="card"><span>{overall_freq:.4f}</span>Claim frequency / year</div>
      <div class="card"><span>{sev['ClaimAmount'].mean():,.0f}</span>Mean claim amount</div>
    </div>
    <h3>freMTPL2freq.csv &ndash; structure</h3>{table_html(overview(freq), "{:,.2f}")}
    <h3>freMTPL2freq.csv &ndash; first rows</h3>{table_html(freq.head(10), "{:,.2f}", index=False)}
    <h3>freMTPL2sev.csv &ndash; structure</h3>{table_html(overview(sev), "{:,.2f}")}
    <h3>freMTPL2sev.csv &ndash; first rows</h3>{table_html(sev.head(10), "{:,.2f}", index=False)}
    """
    add("1. Dataset Overview", body, "overview")

    # ------------------------------------------------------------------ 2. Data quality
    dq = pd.DataFrame(
        {
            "Check": [
                "Duplicate rows (freq)",
                "Duplicate IDpol (freq)",
                "Duplicate rows (sev)",
                "Missing values (freq)",
                "Missing values (sev)",
                "Severity IDpol not found in freq file",
                "Policies with ClaimNb>0 but no severity record",
                "Policies where ClaimNb != number of severity records",
                "Exposure > 1 year",
                "Claim amounts <= 0",
            ],
            "Value": [
                int(freq.duplicated().sum()),
                int(freq["IDpol"].duplicated().sum()),
                int(sev.duplicated().sum()),
                int(freq.isna().sum().sum()),
                int(sev.isna().sum().sum()),
                int(sev_ids_not_in_freq),
                int(((merged["ClaimNb"] > 0) & (merged["n_sev"] == 0)).sum()),
                int(claimnb_mismatch),
                int((freq["Exposure"] > 1).sum()),
                int((sev["ClaimAmount"] <= 0).sum()),
            ],
        }
    )
    add("2. Data Quality Checks", table_html(dq, index=False), "quality")

    # ------------------------------------------------------------------ 3. Descriptive stats
    desc_num = freq[num_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    desc_num["skew"] = freq[num_cols].skew()
    desc_sev = sev[["ClaimAmount"]].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc_sev["skew"] = sev["ClaimAmount"].skew()
    desc_sev["kurtosis"] = sev["ClaimAmount"].kurt()
    desc_cat = pd.DataFrame(
        {
            "levels": [freq[c].nunique() for c in cat_cols],
            "most_frequent": [freq[c].mode()[0] for c in cat_cols],
            "share_%": [freq[c].value_counts(normalize=True).iloc[0] * 100 for c in cat_cols],
        },
        index=cat_cols,
    )
    body = (
        f"<h3>Numerical variables (freq)</h3>{table_html(desc_num, '{:,.2f}')}"
        f"<h3>Claim amount (sev)</h3>{table_html(desc_sev, '{:,.2f}')}"
        f"<h3>Categorical variables (freq)</h3>{table_html(desc_cat, '{:,.2f}')}"
    )
    add("3. Descriptive Statistics", body, "descriptive")

    # ------------------------------------------------------------------ 4. Claim frequency
    cn = freq["ClaimNb"].clip(upper=3).map(lambda x: f"{x}" if x < 3 else "3+").value_counts().sort_index()
    cn_tbl = freq["ClaimNb"].value_counts().sort_index().to_frame("policies")
    cn_tbl["share_%"] = cn_tbl["policies"] / len(freq) * 100
    mean_rate = freq["ClaimNb"].mean()
    var_rate = freq["ClaimNb"].var()
    body = grid(
        fig_to_html(pie(cn, "Policies by number of claims"), "Distribution of ClaimNb (pie)"),
        fig_to_html(_log_bar(cn_tbl["policies"]), "Distribution of ClaimNb (bar, log scale)"),
    )
    body += table_html(cn_tbl, "{:,.3f}")
    body += (
        f"<p>Mean ClaimNb = <b>{mean_rate:.4f}</b>, variance = <b>{var_rate:.4f}</b>, "
        f"dispersion ratio (var/mean) = <b>{var_rate / mean_rate:.3f}</b>. "
        "A ratio above 1 indicates over-dispersion relative to a Poisson model.</p>"
    )
    add("4. Claim Count (Frequency) Distribution", body, "frequency")

    # ------------------------------------------------------------------ 5. Categorical distributions
    body = grid(
        fig_to_html(pie(freq["Area"].value_counts().sort_index(), "Policies by Area"), "Area"),
        fig_to_html(pie(freq["VehGas"].value_counts(), "Policies by fuel type"), "VehGas"),
    )
    brand_counts = freq["VehBrand"].value_counts()
    body += grid(
        fig_to_html(pie(brand_counts, "Policies by vehicle brand"), "VehBrand (pie)"),
        fig_to_html(bar(brand_counts, "Policies by vehicle brand", "VehBrand", "Policies"), "VehBrand (bar)"),
    )
    body += fig_to_html(
        bar(freq["Region"].value_counts(), "Policies by region", "Region", "Policies", horizontal=True, color="#2ca02c"),
        "Region",
    )
    add("5. Portfolio Composition (Categorical Variables)", body, "categorical")

    # ------------------------------------------------------------------ 6. Numerical distributions
    body = grid(
        fig_to_html(hist(freq["Exposure"], "Exposure distribution", "Exposure (years)"), "Exposure"),
        fig_to_html(hist(freq["DrivAge"], "Driver age distribution", "Driver age", bins=83, color="#9467bd"), "DrivAge"),
        fig_to_html(hist(freq["VehAge"].clip(upper=30), "Vehicle age distribution (capped at 30)", "Vehicle age", bins=31, color="#8c564b"), "VehAge"),
        fig_to_html(bar(freq["VehPower"].value_counts().sort_index(), "Vehicle power distribution", "VehPower", "Policies", color="#e377c2"), "VehPower"),
        fig_to_html(hist(freq["BonusMalus"], "Bonus-Malus distribution", "BonusMalus", bins=60, color="#7f7f7f", log_y=True), "BonusMalus (log y-axis)"),
        fig_to_html(hist(np.log(freq["Density"]), "Log population density", "log(Density)", color="#bcbd22"), "log(Density)"),
    )
    add("6. Distributions of Numerical Variables", body, "numerical")

    # ------------------------------------------------------------------ 7. Frequency by risk factors
    f = freq.copy()
    f["DrivAgeBand"] = pd.cut(f["DrivAge"], [18, 21, 26, 31, 41, 51, 61, 71, np.inf], right=False,
                              labels=["18-20", "21-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"])
    f["VehAgeBand"] = pd.cut(f["VehAge"], [0, 1, 3, 6, 11, 16, 21, np.inf], right=False,
                             labels=["0", "1-2", "3-5", "6-10", "11-15", "16-20", "21+"])
    f["BMBand"] = pd.cut(f["BonusMalus"], [50, 51, 61, 81, 101, 151, np.inf], right=False,
                         labels=["50", "51-60", "61-80", "81-100", "101-150", "151+"])
    f["DensityBand"] = pd.qcut(np.log(f["Density"]), 5, labels=["Q1 (rural)", "Q2", "Q3", "Q4", "Q5 (urban)"])

    factors = [
        ("Area", "Area"),
        ("VehGas", "Fuel type"),
        ("VehPower", "Vehicle power"),
        ("VehBrand", "Vehicle brand"),
        ("DrivAgeBand", "Driver age band"),
        ("VehAgeBand", "Vehicle age band"),
        ("BMBand", "Bonus-Malus band"),
        ("DensityBand", "Density quintile"),
    ]
    freq_tables = {c: freq_by(f, c) for c, _ in factors}
    figs = [
        fig_to_html(freq_exposure_plot(freq_tables[c], f"Exposure and claim frequency by {lbl}", lbl, overall_freq), lbl)
        for c, lbl in factors
    ]
    region_freq = freq_by(f, "Region").sort_values("frequency", ascending=False)
    body = grid(*figs)
    body += fig_to_html(
        bar(region_freq["frequency"], "Claim frequency by region", "Region", "Claims per policy-year",
            horizontal=True, color="#d62728"),
        "Claim frequency by Region",
    )
    body += "<h3>One-way frequency tables</h3>" + "".join(
        f"<h4>{lbl}</h4>{table_html(freq_tables[c], '{:,.4f}')}" for c, lbl in factors[:3] + factors[4:]
    )
    add("7. Claim Frequency by Risk Factors (One-way Analysis)", body, "freq_factors")

    # ------------------------------------------------------------------ 8. Severity
    amounts = sev["ClaimAmount"]
    s_sorted = np.sort(amounts.values)
    cum_share = np.cumsum(s_sorted) / s_sorted.sum()
    top1_share = 1 - cum_share[int(len(s_sorted) * 0.99) - 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.linspace(0, 1, len(cum_share)), cum_share, color="#d62728", label="Lorenz curve")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Equality")
    ax.set_xlabel("Cumulative share of claims")
    ax.set_ylabel("Cumulative share of claim amount")
    ax.set_title("Concentration of claim amounts (Lorenz curve)")
    ax.legend()
    lorenz = fig_to_html(fig, "Lorenz curve of claim amounts")

    sev_bins = pd.cut(amounts, [0, 500, 1000, 2000, 5000, 10000, 50000, np.inf],
                      labels=["<500", "500-1k", "1k-2k", "2k-5k", "5k-10k", "10k-50k", "50k+"])
    sev_bin_counts = sev_bins.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    areas = sorted(sev_feat["Area"].unique())
    ax.boxplot([np.log10(sev_feat.loc[sev_feat["Area"] == a, "ClaimAmount"]) for a in areas], showfliers=False)
    ax.set_xticks(range(1, len(areas) + 1), areas)
    ax.set_title("log10(Claim amount) by Area (outliers hidden)")
    ax.set_xlabel("Area")
    ax.set_ylabel("log10(ClaimAmount)")
    box_area = fig_to_html(fig, "Severity by Area")

    sev_feat["DrivAgeBand"] = f.set_index("IDpol").loc[sev_feat["IDpol"], "DrivAgeBand"].values
    sev_feat["BMBand"] = f.set_index("IDpol").loc[sev_feat["IDpol"], "BMBand"].values
    sev_tables = {}
    for col in ["Area", "VehGas", "DrivAgeBand", "BMBand"]:
        sev_tables[col] = sev_feat.groupby(col, observed=True)["ClaimAmount"].agg(["count", "mean", "median"])

    body = grid(
        fig_to_html(hist(np.log10(amounts), "Distribution of log10(Claim amount)", "log10(ClaimAmount)", bins=80, color="#ff7f0e"), "log10 claim amounts"),
        fig_to_html(pie(sev_bin_counts, "Claims by amount band"), "Claim amount bands (pie)"),
        lorenz,
        box_area,
        fig_to_html(bar(sev_tables["DrivAgeBand"]["median"], "Median claim amount by driver age band", "Driver age band", "Median amount", color="#17becf"), "Median severity by driver age"),
        fig_to_html(bar(sev_tables["BMBand"]["median"], "Median claim amount by Bonus-Malus band", "Bonus-Malus band", "Median amount", color="#9467bd"), "Median severity by BonusMalus"),
    )
    body += (
        f"<p>The largest 1% of claims account for <b>{top1_share * 100:.1f}%</b> of the total claim amount. "
        f"Maximum claim = <b>{amounts.max():,.0f}</b>; median = <b>{amounts.median():,.0f}</b>.</p>"
    )
    body += "".join(f"<h4>Severity by {c}</h4>{table_html(t, '{:,.1f}')}" for c, t in sev_tables.items())
    add("8. Claim Severity Analysis", body, "severity")

    # ------------------------------------------------------------------ 9. Frequency-severity relationship
    pol_sev = merged[merged["n_sev"] > 0].copy()
    pol_sev["avg_amount"] = pol_sev["total_amount"] / pol_sev["n_sev"]
    fs = pol_sev.groupby(pol_sev["n_sev"].clip(upper=4).astype(int))["avg_amount"].agg(["count", "mean", "median"])
    fs.index = [str(i) if i < 4 else "4+" for i in fs.index]
    fs.index.name = "claims per policy"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(pol_sev["BonusMalus"], np.log10(pol_sev["avg_amount"]), s=4, alpha=0.2)
    ax.set_xlabel("BonusMalus")
    ax.set_ylabel("log10(average claim amount)")
    ax.set_title("Average claim amount vs Bonus-Malus (claimants)")
    scat = fig_to_html(fig, "Severity vs BonusMalus")

    body = grid(
        fig_to_html(bar(fs["median"], "Median average claim amount by number of claims", "Claims per policy", "Median amount", color="#2ca02c"), "Severity by claim count"),
        scat,
    )
    body += table_html(fs, "{:,.1f}")
    rho = pol_sev[["n_sev", "avg_amount"]].corr(method="spearman").iloc[0, 1]
    body += (
        f"<p>Spearman correlation between claim count and average severity among claimants: <b>{rho:.3f}</b>. "
        "A non-zero value suggests dependence between frequency and severity, motivating joint modelling.</p>"
    )
    add("9. Frequency&ndash;Severity Dependence", body, "dependence")

    # ------------------------------------------------------------------ 10. Correlations
    corr_df = freq[num_cols].copy()
    corr_df["Density"] = np.log(corr_df["Density"])
    corr_df = corr_df.rename(columns={"Density": "logDensity"})
    corr = corr_df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlation matrix (numerical variables)")
    body = fig_to_html(fig, "Correlation heatmap")
    add("10. Correlation Analysis", body, "correlation")

    # ------------------------------------------------------------------ 11. Summary
    zero_share = (freq["ClaimNb"] == 0).mean() * 100
    da = freq_tables["DrivAgeBand"]["frequency"]
    bm = freq_tables["BMBand"]["frequency"]
    ar = freq_tables["Area"]["frequency"]
    dens = freq_tables["DensityBand"]["frequency"]
    gas = freq_tables["VehGas"]["frequency"]
    short_exp = (freq["Exposure"] < 1).mean() * 100
    top_brand = brand_counts.index[0]
    top_region = freq["Region"].value_counts().index[0]

    bullets = [
        f"The portfolio contains <b>{len(freq):,}</b> policies with <b>{total_exposure:,.0f}</b> policy-years of exposure, "
        f"and the severity file holds <b>{len(sev):,}</b> individual claim amounts.",
        f"Claim counts are heavily zero-inflated: <b>{zero_share:.1f}%</b> of policies report no claim, and the maximum is "
        f"<b>{freq['ClaimNb'].max()}</b> claims on one policy.",
        f"The overall claim frequency is <b>{overall_freq:.4f}</b> claims per policy-year with a variance/mean ratio of "
        f"<b>{var_rate / mean_rate:.2f}</b>, i.e. over-dispersion that favours Negative Binomial or zero-inflated count models.",
        f"Exposure is unequal: <b>{short_exp:.1f}%</b> of policies are observed for less than one year (median "
        f"<b>{freq['Exposure'].median():.2f}</b>), so exposure must enter frequency models as an offset.",
        f"Claim severity is highly right-skewed (mean <b>{amounts.mean():,.0f}</b> vs median <b>{amounts.median():,.0f}</b>, "
        f"skewness <b>{amounts.skew():.1f}</b>), indicating a heavy tail suited to Gamma, Log-normal or Pareto-type models.",
        f"Large losses dominate the total cost: the top 1% of claims make up <b>{top1_share * 100:.1f}%</b> of the total "
        f"claim amount, so large-claim capping or a separate tail model should be considered.",
        f"Bonus-Malus is the strongest frequency driver: the frequency in the highest band (<b>{bm.idxmax()}</b>: "
        f"{bm.max():.3f}) is <b>{bm.max() / bm.loc['50']:.1f}x</b> that of drivers at the minimum level 50 ({bm.loc['50']:.3f}).",
        f"Driver age has a U-shaped effect: young drivers (<b>{da.idxmax()}</b>) show the highest frequency "
        f"({da.max():.3f}), while the lowest is in the <b>{da.idxmin()}</b> band ({da.min():.3f}).",
        f"Urbanisation raises risk: frequency rises from <b>{dens.iloc[0]:.3f}</b> in the most rural density quintile to "
        f"<b>{dens.iloc[-1]:.3f}</b> in the most urban; Area {ar.idxmax()} has the highest frequency ({ar.max():.3f}) "
        f"and Area {ar.idxmin()} the lowest ({ar.min():.3f}).",
        f"Data are clean (no missing values), but <b>{sev_ids_not_in_freq:,}</b> severity records lack a matching policy "
        f"and <b>{claimnb_mismatch:,}</b> claiming policies have a ClaimNb inconsistent with the severity file; the "
        f"frequency&ndash;severity Spearman correlation among claimants is <b>{rho:.3f}</b>, supporting a joint modelling approach. "
        f"(Most common brand: {html.escape(top_brand)}; region: {html.escape(top_region)}; diesel vs regular frequency: "
        f"{gas.get('Diesel', np.nan):.3f} vs {gas.get('Regular', np.nan):.3f}.)",
    ]
    body = "<ol class='summary'>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ol>"
    add("11. Key Findings &ndash; 10-Point Summary", body, "summary")

    # ------------------------------------------------------------------ assemble
    css = """
    body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f9;color:#222}
    header{background:#1f3b63;color:#fff;padding:28px 40px}
    header h1{margin:0 0 6px 0;font-size:26px} header p{margin:0;opacity:.85}
    main{max-width:1250px;margin:auto;padding:20px 40px}
    nav{background:#fff;border-radius:8px;padding:12px 24px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
    nav ol{columns:2;margin:6px 0} nav a{color:#1f3b63;text-decoration:none}
    section{background:#fff;border-radius:8px;padding:18px 26px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,.1);overflow-x:auto}
    h2{color:#1f3b63;border-bottom:2px solid #1f3b63;padding-bottom:6px} h3,h4{color:#2d5a8c}
    .tbl{border-collapse:collapse;font-size:13px;margin:8px 0}
    .tbl th,.tbl td{padding:5px 10px;border-bottom:1px solid #e3e3e3;text-align:right}
    .tbl th{background:#eef2f7} .tbl tr:hover td{background:#f8fbff}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px}
    figure{margin:8px 0;text-align:center} figure img{max-width:100%}
    figcaption{font-size:12px;color:#666}
    .cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px}
    .card{flex:1 1 160px;background:#eef2f7;border-radius:8px;padding:12px;text-align:center;font-size:13px}
    .card span{display:block;font-size:22px;font-weight:600;color:#1f3b63}
    .summary li{margin:10px 0;line-height:1.5}
    footer{text-align:center;color:#888;font-size:12px;padding:20px}
    """
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>EDA Report - French Motor TPL (freMTPL2)</title><style>{css}</style></head>
<body>
<header><h1>Exploratory Data Analysis Report &ndash; French Motor Third-Party Liability (freMTPL2)</h1></header>
<main><nav><b>Contents</b><ol>{''.join(toc)}</ol></nav>
{''.join(parts)}
</main><footer>Data: freMTPL2freq.csv, freMTPL2sev.csv</footer></body></html>"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(doc, encoding="utf-8")
    print(f"Report written to {OUT_FILE}")


def _log_bar(counts: pd.Series):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(counts.index.astype(str), counts.values, color="#ff7f0e")
    ax.set_yscale("log")
    ax.set_xlabel("ClaimNb")
    ax.set_ylabel("Policies (log scale)")
    ax.set_title("Policies by number of claims")
    return fig


if __name__ == "__main__":
    build_report()
