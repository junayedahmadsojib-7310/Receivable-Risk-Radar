import re

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Receivable Risk Radar",
    page_icon="📊",
    layout="wide"
)

st.title("Receivable Risk Radar")

st.caption(
    "AR concentration and counterparty review for Danish SMEs"
)

st.info(
    "This dashboard is a receivables screening tool — not a credit rating "
    "or prediction of default."
)


# =========================================================
# BASIC SETTINGS
# =========================================================

API_BASE = "https://api.companydata.dk/v1"

REQUIRED_COLUMNS = [
    "customer",
    "cvr",
    "invoice_no",
    "invoice_date",
    "due_date",
    "outstanding_dkk",
]

REVIEW_ORDER = [
    "Urgent",
    "Review",
    "Watch",
    "No current flag",
]

REVIEW_COLORS = {
    "Urgent": "#B42318",
    "Review": "#E76F51",
    "Watch": "#D4A72C",
    "No current flag": "#2A9D8F",
}


# =========================================================
# HELPER: CURRENT DATE IN DENMARK
# =========================================================

def local_today():

    return pd.Timestamp.now(
        tz="Europe/Copenhagen"
    ).tz_localize(None).normalize()


# =========================================================
# DEMO DATA
#
# Public CVR identifiers are real.
# Invoice relationships / amounts / dates are fictional.
# =========================================================

def make_demo_data():

    today = local_today()

    rows = [
        {
            "customer": "NOVO NORDISK A/S",
            "cvr": "24256790",
            "invoice_no": "DEMO-1001",
            "invoice_date": today - pd.Timedelta(days=78),
            "due_date": today - pd.Timedelta(days=48),
            "outstanding_dkk": 110000,
        },
        {
            "customer": "NOVO NORDISK A/S",
            "cvr": "24256790",
            "invoice_no": "DEMO-1002",
            "invoice_date": today - pd.Timedelta(days=40),
            "due_date": today - pd.Timedelta(days=10),
            "outstanding_dkk": 45000,
        },
        {
            "customer": "DSV A/S",
            "cvr": "58233528",
            "invoice_no": "DEMO-1003",
            "invoice_date": today - pd.Timedelta(days=16),
            "due_date": today + pd.Timedelta(days=14),
            "outstanding_dkk": 95000,
        },
        {
            "customer": "LEGO A/S",
            "cvr": "54562519",
            "invoice_no": "DEMO-1004",
            "invoice_date": today - pd.Timedelta(days=55),
            "due_date": today - pd.Timedelta(days=25),
            "outstanding_dkk": 70000,
        },
        {
            "customer": "LEGO A/S",
            "cvr": "54562519",
            "invoice_no": "DEMO-1005",
            "invoice_date": today - pd.Timedelta(days=8),
            "due_date": today + pd.Timedelta(days=22),
            "outstanding_dkk": 35000,
        },
        {
            "customer": "Ingrediensen ApS",
            "cvr": "44385058",
            "invoice_no": "DEMO-1006",
            "invoice_date": today - pd.Timedelta(days=100),
            "due_date": today - pd.Timedelta(days=70),
            "outstanding_dkk": 60000,
        },
    ]

    return pd.DataFrame(rows)


# =========================================================
# CSV TEMPLATE
# =========================================================

def csv_template():

    template = pd.DataFrame(
        columns=REQUIRED_COLUMNS
    )

    return template.to_csv(
        index=False
    ).encode("utf-8")


# =========================================================
# MONEY PARSER
#
# Handles ordinary numbers plus common comma / dot formats.
# =========================================================

def parse_money(value):

    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    text = re.sub(
        r"[^\d,.\-]",
        "",
        text
    )

    if not text:
        return None

    # Both separators exist.
    # Assume the final separator is decimal.
    if "," in text and "." in text:

        if text.rfind(",") > text.rfind("."):

            text = text.replace(".", "")
            text = text.replace(",", ".")

        else:

            text = text.replace(",", "")

    # Multiple commas = most likely thousands separators
    elif text.count(",") > 1:

        text = text.replace(",", "")

    # Single comma
    elif "," in text:

        right_side = text.split(",")[-1]

        if len(right_side) in [1, 2]:

            text = text.replace(",", ".")

        else:

            text = text.replace(",", "")

    # Multiple dots = thousands separators
    elif text.count(".") > 1:

        text = text.replace(".", "")

    try:

        return float(text)

    except ValueError:

        return None


# =========================================================
# VALIDATE + CLEAN AR FILE
# =========================================================

def prepare_ar_data(raw_df):

    df = raw_df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing_columns)
        )

    df = df[REQUIRED_COLUMNS].copy()

    # Customer
    df["customer"] = (
        df["customer"]
        .astype(str)
        .str.strip()
    )

    # CVR
    df["cvr"] = (
        df["cvr"]
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
        .str.replace(
            r"\D",
            "",
            regex=True
        )
    )

    # Invoice number
    df["invoice_no"] = (
        df["invoice_no"]
        .astype(str)
        .str.strip()
    )

    # Dates
    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"],
        errors="coerce"
    )

    df["due_date"] = pd.to_datetime(
        df["due_date"],
        errors="coerce"
    )

    # Money
    df["outstanding_dkk"] = (
        df["outstanding_dkk"]
        .map(parse_money)
    )

    # Remove fully unusable rows
    df = df.dropna(
        subset=[
            "customer",
            "invoice_no",
            "invoice_date",
            "due_date",
            "outstanding_dkk",
        ]
    )

    # CVR must be exactly 8 digits
    df = df[
        df["cvr"].str.fullmatch(
            r"\d{8}",
            na=False
        )
    ].copy()

    # Only open positive receivables
    df = df[
        df["outstanding_dkk"] > 0
    ].copy()

    # Avoid accidental duplicate invoices
    df = df.drop_duplicates(
        subset=[
            "cvr",
            "invoice_no",
        ],
        keep="first"
    )

    if df.empty:

        raise ValueError(
            "No usable AR rows remained after validation."
        )

    return df


# =========================================================
# CALCULATE INVOICE AGING
# =========================================================

def add_invoice_aging(df):

    view = df.copy()

    today = local_today()

    view["days_overdue"] = (
        today - view["due_date"]
    ).dt.days.clip(lower=0)

    view["is_overdue"] = (
        view["due_date"] < today
    )

    view["overdue_outstanding"] = (
        view["outstanding_dkk"]
        .where(
            view["is_overdue"],
            0
        )
    )

    view["age_bucket"] = pd.cut(
        view["days_overdue"],
        bins=[
            -1,
            0,
            30,
            60,
            90,
            float("inf")
        ],
        labels=[
            "Current",
            "1–30",
            "31–60",
            "61–90",
            "90+",
        ]
    )

    return view


# =========================================================
# CUSTOMER SUMMARY
# =========================================================

def build_customer_summary(invoice_df):

    total_ar = (
        invoice_df["outstanding_dkk"]
        .sum()
    )

    summary = (
        invoice_df
        .groupby(
            "cvr",
            as_index=False
        )
        .agg(
            customer=(
                "customer",
                "first"
            ),

            total_outstanding=(
                "outstanding_dkk",
                "sum"
            ),

            overdue_amount=(
                "overdue_outstanding",
                "sum"
            ),

            invoice_count=(
                "invoice_no",
                "nunique"
            ),

            overdue_invoice_count=(
                "is_overdue",
                "sum"
            ),

            max_days_overdue=(
                "days_overdue",
                "max"
            ),
        )
    )

    summary["ar_share"] = (
        summary["total_outstanding"]
        / total_ar
    )

    summary["customer_aging"] = pd.cut(
        summary["max_days_overdue"],
        bins=[
            -1,
            0,
            30,
            60,
            90,
            float("inf")
        ],
        labels=[
            "Current",
            "1–30",
            "31–60",
            "61–90",
            "90+",
        ]
    ).astype(str)

    return summary


# =========================================================
# API UTILITIES
#
# The provider confirms the endpoints and Bearer
# authentication. These parsing helpers are deliberately
# defensive so the dashboard does not crash if fields are
# nested differently in a response.
# =========================================================

def normalise_key(value):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower()
    )


def unwrap_value(value):

    if isinstance(
        value,
        (str, int, float, bool)
    ) or value is None:

        return value

    if isinstance(value, dict):

        preferred = [
            "value",
            "amount",
            "text",
            "label",
            "name",
            "code",
        ]

        for preferred_key in preferred:

            for key, inner_value in value.items():

                if (
                    normalise_key(key)
                    == normalise_key(
                        preferred_key
                    )
                ):

                    if not isinstance(
                        inner_value,
                        (dict, list)
                    ):

                        return inner_value

    return None


def deep_find(obj, aliases):

    aliases = {
        normalise_key(alias)
        for alias in aliases
    }

    if isinstance(obj, dict):

        # Look at this level first
        for key, value in obj.items():

            if normalise_key(key) in aliases:

                unwrapped = unwrap_value(
                    value
                )

                if unwrapped is not None:

                    return unwrapped

        # Then recurse
        for value in obj.values():

            found = deep_find(
                value,
                aliases
            )

            if found is not None:

                return found

    elif isinstance(obj, list):

        for item in obj:

            found = deep_find(
                item,
                aliases
            )

            if found is not None:

                return found

    return None


def iter_dicts(obj):

    if isinstance(obj, dict):

        yield obj

        for value in obj.values():

            yield from iter_dicts(
                value
            )

    elif isinstance(obj, list):

        for item in obj:

            yield from iter_dicts(
                item
            )


def numeric_value(value):

    if value is None:
        return None

    return parse_money(value)


# =========================================================
# EXTRACT COMPANY MASTER DATA
# =========================================================

def extract_company_info(payload):

    public_name = deep_find(
        payload,
        [
            "company_name",
            "companyName",
            "legal_name",
            "legalName",
            "name",
        ]
    )

    status = deep_find(
        payload,
        [
            "company_status",
            "companyStatus",
            "current_status",
            "currentStatus",
            "status",
        ]
    )

    explicit_active = deep_find(
        payload,
        [
            "is_active",
            "isActive",
            "active",
        ]
    )

    active = None

    if isinstance(
        explicit_active,
        bool
    ):

        active = explicit_active

    elif status is not None:

        text = (
            str(status)
            .strip()
            .lower()
        )

        inactive_words = [
            "inactive",
            "dissolved",
            "ceased",
            "bankrupt",
            "bankruptcy",
            "opløst",
            "oploest",
            "konkurs",
            "tvangsopløst",
            "tvangsoploest",
        ]

        active_words = [
            "active",
            "aktiv",
            "normal",
        ]

        if any(
            word in text
            for word in inactive_words
        ):

            active = False

        elif any(
            word in text
            for word in active_words
        ):

            active = True

    if status is None:

        if active is True:
            status = "Active"

        elif active is False:
            status = "Not active"

    return {
        "public_name": public_name,
        "company_status": status,
        "company_active": active,
    }


# =========================================================
# EXTRACT LATEST FINANCIAL STATEMENT
# =========================================================

YEAR_KEYS = [
    "year",
    "financial_year",
    "financialYear",
    "report_year",
    "reportYear",
    "accounting_year",
    "accountingYear",
]

DATE_KEYS = [
    "period_end",
    "periodEnd",
    "end_date",
    "endDate",
    "closing_date",
    "closingDate",
]

EQUITY_KEYS = [
    "equity",
    "total_equity",
    "totalEquity",
    "shareholders_equity",
    "shareholdersEquity",
    "egenkapital",
]

RESULT_KEYS = [
    "profit_loss",
    "profitLoss",
    "profit_or_loss",
    "profitOrLoss",
    "net_result",
    "netResult",
    "net_income",
    "netIncome",
    "net_profit",
    "netProfit",
    "annual_result",
    "annualResult",
    "profit",
]


def extract_year(obj):

    raw_year = deep_find(
        obj,
        YEAR_KEYS
    )

    if raw_year is not None:

        match = re.search(
            r"(19|20)\d{2}",
            str(raw_year)
        )

        if match:

            return int(
                match.group()
            )

    raw_date = deep_find(
        obj,
        DATE_KEYS
    )

    if raw_date is not None:

        match = re.search(
            r"(19|20)\d{2}",
            str(raw_date)
        )

        if match:

            return int(
                match.group()
            )

    return None


def extract_latest_financials(payload):

    candidates = []

    period_keys = {
        normalise_key(k)
        for k in (
            YEAR_KEYS
            + DATE_KEYS
            + [
                "period",
                "accounting_period",
                "accountingPeriod",
            ]
        )
    }

    for item in iter_dicts(
        payload
    ):

        keys_here = {
            normalise_key(key)
            for key in item.keys()
        }

        if not (
            keys_here
            & period_keys
        ):

            continue

        year = extract_year(
            item
        )

        equity = numeric_value(
            deep_find(
                item,
                EQUITY_KEYS
            )
        )

        result = numeric_value(
            deep_find(
                item,
                RESULT_KEYS
            )
        )

        if (
            year is not None
            and (
                equity is not None
                or result is not None
            )
        ):

            candidates.append(
                {
                    "financial_year":
                        year,

                    "equity":
                        equity,

                    "net_result":
                        result,
                }
            )

    if candidates:

        return max(
            candidates,
            key=lambda x:
                x["financial_year"]
        )

    # Fallback if the response is flatter
    return {
        "financial_year":
            extract_year(payload),

        "equity":
            numeric_value(
                deep_find(
                    payload,
                    EQUITY_KEYS
                )
            ),

        "net_result":
            numeric_value(
                deep_find(
                    payload,
                    RESULT_KEYS
                )
            ),
    }


# =========================================================
# API REQUEST
# =========================================================

def api_get(path, api_key):

    response = requests.get(
        f"{API_BASE}{path}",
        headers={
            "Authorization":
                f"Bearer {api_key}"
        },
        timeout=15,
    )

    if response.status_code == 429:

        raise RuntimeError(
            "Companydata API rate limit reached."
        )

    if response.status_code == 404:

        return {}

    response.raise_for_status()

    return response.json()


# =========================================================
# CACHED COMPANY LOOKUP
# =========================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def fetch_company(
    cvr,
    _api_key
):

    return api_get(
        f"/companies/{cvr}",
        _api_key
    )


@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def fetch_financials(
    cvr,
    _api_key
):

    return api_get(
        f"/companies/{cvr}/financials",
        _api_key
    )


# =========================================================
# ENRICH TOP CUSTOMER EXPOSURES
# =========================================================

def enrich_customers(
    summary,
    api_key,
    max_companies=4
):

    out = summary.copy()

    out["public_name"] = None
    out["company_status"] = None
    out["company_active"] = None
    out["financial_year"] = None
    out["equity"] = None
    out["net_result"] = None
    out["api_enriched"] = False
    out["api_note"] = None

    targets = (
        out
        .nlargest(
            max_companies,
            "total_outstanding"
        )
        ["cvr"]
        .tolist()
    )

    progress = st.progress(
        0,
        text="Enriching largest customer exposures..."
    )

    total = len(targets)

    for i, cvr in enumerate(
        targets,
        start=1
    ):

        company_payload = {}
        financial_payload = {}
        notes = []

        try:

            company_payload = (
                fetch_company(
                    cvr,
                    api_key
                )
            )

        except Exception as error:

            notes.append(
                f"Company lookup: {error}"
            )

        try:

            financial_payload = (
                fetch_financials(
                    cvr,
                    api_key
                )
            )

        except Exception as error:

            notes.append(
                f"Financials: {error}"
            )

        company_info = (
            extract_company_info(
                company_payload
            )
            if company_payload
            else {}
        )

        financial_info = (
            extract_latest_financials(
                financial_payload
            )
            if financial_payload
            else {}
        )

        mask = (
            out["cvr"] == cvr
        )

        for key, value in (
            company_info.items()
        ):

            out.loc[
                mask,
                key
            ] = value

        for key, value in (
            financial_info.items()
        ):

            out.loc[
                mask,
                key
            ] = value

        out.loc[
            mask,
            "api_enriched"
        ] = bool(
            company_payload
            or financial_payload
        )

        if notes:

            out.loc[
                mask,
                "api_note"
            ] = " | ".join(
                notes
            )

        progress.progress(
            i / total,
            text=(
                f"Enriching public data "
                f"({i}/{total})"
            )
        )

    progress.empty()

    return out


# =========================================================
# REVIEW RULES
# =========================================================

def create_review_signals(
    summary
):

    out = summary.copy()

    out["aging_signal"] = (
        out["max_days_overdue"]
        >= 31
    )

    out["concentration_signal"] = (
        out["ar_share"]
        >= 0.15
    )

    out["negative_equity_signal"] = (
        pd.to_numeric(
            out["equity"],
            errors="coerce"
        )
        < 0
    )

    out["loss_signal"] = (
        pd.to_numeric(
            out["net_result"],
            errors="coerce"
        )
        < 0
    )

    out["inactive_signal"] = (
        out["company_active"]
        .eq(False)
    )

    out["signal_count"] = (
        out[
            [
                "aging_signal",
                "concentration_signal",
                "negative_equity_signal",
                "loss_signal",
            ]
        ]
        .fillna(False)
        .sum(axis=1)
    )

    levels = []
    reasons = []

    for _, row in out.iterrows():

        row_reasons = []

        if row[
            "max_days_overdue"
        ] > 90:

            row_reasons.append(
                "90+ days overdue"
            )

        elif row[
            "max_days_overdue"
        ] >= 31:

            row_reasons.append(
                "31+ days overdue"
            )

        if row[
            "concentration_signal"
        ]:

            row_reasons.append(
                "15%+ of total AR"
            )

        if row[
            "negative_equity_signal"
        ]:

            row_reasons.append(
                "Negative latest equity"
            )

        if row[
            "loss_signal"
        ]:

            row_reasons.append(
                "Latest reported loss"
            )

        if row[
            "inactive_signal"
        ]:

            row_reasons.append(
                "Company not active"
            )

        # Priority rule
        if (
            row["inactive_signal"]
            or row[
                "max_days_overdue"
            ] > 90
        ):

            level = "Urgent"

        elif (
            row["signal_count"]
            >= 2
        ):

            level = "Review"

        elif (
            row["signal_count"]
            == 1
        ):

            level = "Watch"

        else:

            level = (
                "No current flag"
            )

        levels.append(level)

        reasons.append(
            "; ".join(row_reasons)
            if row_reasons
            else "No defined signal"
        )

    out["review_level"] = levels
    out["review_reason"] = reasons

    return out


# =========================================================
# API KEY
# =========================================================

def get_api_key():

    try:

        return st.secrets[
            "COMPANYDATA_API_KEY"
        ]

    except Exception:

        return None


# =========================================================
# SIDEBAR — DATA SOURCE
# =========================================================

with st.sidebar:

    st.header("Controls")

    source = st.radio(
        "AR data",
        [
            "Demo data",
            "Upload CSV",
        ]
    )

    st.download_button(
        "Download CSV template",
        data=csv_template(),
        file_name="ar_template.csv",
        mime="text/csv",
    )


# =========================================================
# LOAD AR DATA
# =========================================================

if source == "Demo data":

    raw_df = make_demo_data()

    st.warning(
        "Demo mode uses real public CVR identifiers only for "
        "live company-data enrichment. All invoices, balances, "
        "dates and customer relationships shown in the demo "
        "are fictional."
    )

else:

    uploaded_file = (
        st.sidebar.file_uploader(
            "Upload AR aging CSV",
            type=["csv"]
        )
    )

    if uploaded_file is None:

        st.info(
            "Upload an AR aging CSV to continue."
        )

        st.stop()

    try:

        raw_df = pd.read_csv(
            uploaded_file,
            dtype={
                "cvr": str
            }
        )

    except Exception as error:

        st.error(
            f"Could not read the CSV: {error}"
        )

        st.stop()


# =========================================================
# CLEAN + CALCULATE ACCOUNTING DATA
# =========================================================

try:

    ar_df = prepare_ar_data(
        raw_df
    )

except ValueError as error:

    st.error(str(error))
    st.stop()


invoice_view = add_invoice_aging(
    ar_df
)

customer_summary = (
    build_customer_summary(
        invoice_view
    )
)


# =========================================================
# LIVE COMPANY ENRICHMENT
# =========================================================

api_key = get_api_key()

if api_key:

    customer_summary = (
        enrich_customers(
            customer_summary,
            api_key,
            max_companies=4
        )
    )

    st.caption(
        "Public-company enrichment is applied to the four "
        "largest customer exposures in this assignment MVP "
        "to stay within the free API rate limit."
    )

else:

    # Create columns so rest of app works
    for column in [
        "public_name",
        "company_status",
        "company_active",
        "financial_year",
        "equity",
        "net_result",
        "api_enriched",
        "api_note",
    ]:

        customer_summary[
            column
        ] = None

    customer_summary[
        "api_enriched"
    ] = False

    st.warning(
        "Companydata API enrichment is currently disabled. "
        "Add COMPANYDATA_API_KEY to Streamlit secrets to "
        "activate live company status and financial data."
    )


# =========================================================
# APPLY REVIEW RULES
# =========================================================

customer_summary = (
    create_review_signals(
        customer_summary
    )
)


# =========================================================
# SIDEBAR — FILTERS
# =========================================================

with st.sidebar:

    st.divider()

    aging_filter = st.selectbox(
        "Aging status",
        [
            "All",
            "Current",
            "1–30",
            "31–60",
            "61–90",
            "90+",
        ]
    )

    review_filter = st.selectbox(
        "Review level",
        [
            "All",
            "Urgent",
            "Review",
            "Watch",
            "No current flag",
        ]
    )

    min_exposure = st.number_input(
        "Minimum exposure (DKK)",
        min_value=0,
        value=0,
        step=10000,
    )

    selected_customers = (
        st.multiselect(
            "Customers",
            options=sorted(
                customer_summary[
                    "customer"
                ].unique()
            ),
        )
    )


# =========================================================
# FILTERED VIEW
# =========================================================

view = customer_summary.copy()

if aging_filter != "All":

    view = view[
        view["customer_aging"]
        == aging_filter
    ]

if review_filter != "All":

    view = view[
        view["review_level"]
        == review_filter
    ]

view = view[
    view["total_outstanding"]
    >= min_exposure
]

if selected_customers:

    view = view[
        view["customer"]
        .isin(
            selected_customers
        )
    ]


if view.empty:

    st.info(
        "No customers match the selected filters."
    )

    st.stop()


# =========================================================
# KPIs
# =========================================================

total_outstanding = (
    view[
        "total_outstanding"
    ].sum()
)

overdue_amount = (
    view[
        "overdue_amount"
    ].sum()
)

overdue_pct = (
    overdue_amount
    / total_outstanding
    * 100
    if total_outstanding
    else 0
)

largest_concentration = (
    view[
        "ar_share"
    ].max()
    * 100
)

review_amount = (
    view.loc[
        view["review_level"]
        .isin(
            [
                "Urgent",
                "Review"
            ]
        ),
        "total_outstanding"
    ]
    .sum()
)


c1, c2, c3, c4 = (
    st.columns(4)
)


c1.metric(
    "Outstanding AR",
    f"DKK {total_outstanding:,.0f}"
)


c2.metric(
    "Overdue AR",
    f"DKK {overdue_amount:,.0f}",
    f"{overdue_pct:.1f}% of selected AR"
)


c3.metric(
    "Largest customer exposure",
    f"{largest_concentration:.1f}%"
)


c4.metric(
    "Urgent + Review",
    f"DKK {review_amount:,.0f}"
)


# =========================================================
# MAIN VISUALIZATION
# =========================================================

st.subheader(
    "Largest customer exposures"
)

chart_df = (
    view
    .nlargest(
        10,
        "total_outstanding"
    )
    .sort_values(
        "total_outstanding"
    )
    .copy()
)

chart_df["amount_label"] = (
    chart_df[
        "total_outstanding"
    ]
    .map(
        lambda x:
        f"DKK {x:,.0f}"
    )
)

fig = px.bar(
    chart_df,
    x="total_outstanding",
    y="customer",
    orientation="h",
    color="review_level",
    color_discrete_map=(
        REVIEW_COLORS
    ),
    category_orders={
        "review_level":
            REVIEW_ORDER
    },
    text="amount_label",
    labels={
        "customer": "",
        "total_outstanding":
            "Outstanding DKK",
        "review_level":
            "Review level",
    },
    hover_data={
        "total_outstanding":
            ":,.0f",

        "ar_share":
            ":.1%",

        "max_days_overdue":
            True,

        "review_reason":
            True,

        "amount_label":
            False,
    }
)

fig.update_traces(
    textposition="outside"
)

fig.update_layout(
    height=450,
    margin=dict(
        l=10,
        r=80,
        t=10,
        b=20
    ),
    legend_title_text="",
)

st.plotly_chart(
    fig,
    width="stretch"
)


# =========================================================
# REVIEW TABLE
# =========================================================

st.subheader(
    "Customer review"
)

display_df = view[
    [
        "customer",
        "cvr",
        "total_outstanding",
        "overdue_amount",
        "ar_share",
        "max_days_overdue",
        "company_status",
        "financial_year",
        "net_result",
        "equity",
        "review_level",
        "review_reason",
    ]
].copy()


display_df = (
    display_df
    .sort_values(
        [
            "review_level",
            "total_outstanding",
        ],
        ascending=[
            True,
            False
        ]
    )
)


st.dataframe(
    display_df,
    hide_index=True,
    width="stretch",
    column_config={

        "customer":
            "Customer",

        "cvr":
            "CVR",

        "total_outstanding":
            st.column_config.NumberColumn(
                "Outstanding DKK",
                format="%.0f"
            ),

        "overdue_amount":
            st.column_config.NumberColumn(
                "Overdue DKK",
                format="%.0f"
            ),

        "ar_share":
            st.column_config.NumberColumn(
                "AR share",
                format="%.1f%%"
            ),

        "max_days_overdue":
            "Oldest overdue (days)",

        "company_status":
            "Company status",

        "financial_year":
            "Latest year",

        "net_result":
            st.column_config.NumberColumn(
                "Latest result",
                format="%.0f"
            ),

        "equity":
            st.column_config.NumberColumn(
                "Latest equity",
                format="%.0f"
            ),

        "review_level":
            "Review",

        "review_reason":
            "Reason",
    }
)


# =========================================================
# SESSION STATE — FOLLOW-UP LIST
# =========================================================

st.subheader(
    "Accounts for follow-up"
)

if (
    "follow_up_cvrs"
    not in st.session_state
):

    st.session_state[
        "follow_up_cvrs"
    ] = []


follow_options = {
    f"{row.customer} · "
    f"DKK {row.total_outstanding:,.0f}":
        row.cvr

    for row in
    view.itertuples()
}


selected_follow = (
    st.selectbox(
        "Customer to review",
        options=list(
            follow_options.keys()
        )
    )
)


left, right, _ = st.columns(
    [1, 1, 4]
)


if left.button(
    "Add to follow-up"
):

    selected_cvr = (
        follow_options[
            selected_follow
        ]
    )

    if (
        selected_cvr
        not in
        st.session_state[
            "follow_up_cvrs"
        ]
    ):

        st.session_state[
            "follow_up_cvrs"
        ].append(
            selected_cvr
        )


if right.button(
    "Clear follow-up"
):

    st.session_state[
        "follow_up_cvrs"
    ] = []


# =========================================================
# DISPLAY FOLLOW-UP LIST
# =========================================================

if st.session_state[
    "follow_up_cvrs"
]:

    follow_df = (
        customer_summary[
            customer_summary[
                "cvr"
            ].isin(
                st.session_state[
                    "follow_up_cvrs"
                ]
            )
        ]
        [
            [
                "customer",
                "cvr",
                "total_outstanding",
                "max_days_overdue",
                "ar_share",
                "review_level",
                "review_reason",
            ]
        ]
        .copy()
    )

    st.dataframe(
        follow_df,
        hide_index=True,
        width="stretch"
    )

    follow_csv = (
        follow_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        "Download follow-up list",
        data=follow_csv,
        file_name=(
            "receivable_follow_up.csv"
        ),
        mime="text/csv"
    )

else:

    st.caption(
        "No customer has been added to the follow-up list yet."
    )


# =========================================================
# DOWNLOAD COMPLETE REVIEW
# =========================================================

with st.expander(
    "Data and downloads"
):

    st.write(
        "Filtered customer review"
    )

    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch"
    )

    review_csv = (
        display_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        "Download filtered AR review",
        data=review_csv,
        file_name=(
            "receivable_risk_review.csv"
        ),
        mime="text/csv"
    )

    st.write(
        "Underlying invoice-level data"
    )

    st.dataframe(
        invoice_view,
        hide_index=True,
        width="stretch"
    )


# =========================================================
# METHODOLOGY
# =========================================================

with st.expander(
    "How review flags work"
):

    st.markdown(
        """
**Urgent**
- Company is identified as not active, **or**
- Oldest open invoice is more than 90 days overdue.

**Review**
- Not already Urgent, and at least two review signals are present.

**Watch**
- Exactly one review signal is present.

**No current flag**
- None of the defined screening signals are present.

**Review signals**
- Oldest invoice is at least 31 days overdue.
- Customer represents at least 15% of total AR.
- Latest available public financial statement shows negative equity.
- Latest available public financial statement shows a loss.

The 15% concentration threshold is a management-review rule chosen
for this demonstration. It is not an accounting standard or a
validated credit-risk threshold.

Missing public financial information does not count as a negative
signal.
        """
    )


# =========================================================
# DATA SOURCE NOTE
# =========================================================

st.caption(
    "Public company information is requested from Companydata.dk. "
    "The dashboard combines bookkeeping data with public company "
    "information for screening and management review only."
)