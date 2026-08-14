"""SmartDemand retail planning dashboard."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.forecasting import (
    MANUAL_MAPPING_FIELDS,
    build_prediction_row,
    inventory_recommendation,
    predict_with_interval,
    prepare_data,
    standardize_dataset,
    train_model,
    validate_dataset,
)

st.set_page_config(page_title="SmartDemand AI", page_icon="📈", layout="wide")

DEFAULT_DATA_PATHS = [
    Path("data/SmartDemandAI_Dataset.xlsx"),
    Path("data/SmartDemandAI_Dataset.csv"),
    Path(r"C:\Users\ASUS\Downloads\Mobile Devices\SmartDemandAI_Dataset.xlsx"),
    Path(r"C:\Users\ASUS\Downloads\Mobile Devices\SmartDemandAI_Dataset.csv"),
]


@st.cache_data(show_spinner=False)
def read_dataset(source: bytes | str, file_name: str) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        if isinstance(source, bytes):
            return pd.read_csv(StringIO(source.decode("utf-8-sig", errors="ignore")))
        return pd.read_csv(source)
    return pd.read_excel(BytesIO(source) if isinstance(source, bytes) else source)


@st.cache_resource(show_spinner="Training forecast engine...")
def fit_cached(frame: pd.DataFrame, trees: int):
    return train_model(prepare_data(frame), n_estimators=trees)


def status_badge(status: str) -> str:
    colors = {"REORDER": "🔴", "OVERSTOCK": "🟠", "SUFFICIENT": "🟢"}
    return f"{colors.get(status, '⚪')} {status.title()}"


def load_source(uploaded_file):
    if uploaded_file is not None:
        return read_dataset(uploaded_file.getvalue(), uploaded_file.name), uploaded_file.name
    for candidate in DEFAULT_DATA_PATHS:
        if candidate.exists():
            return read_dataset(str(candidate), candidate.name), candidate.name
    return None, None


def get_manual_mapping(columns: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    options = ["Auto detect"] + columns
    with st.sidebar.expander("Column mapping", expanded=False):
        st.caption("Map your uploaded headers when they use different names.")
        for field in MANUAL_MAPPING_FIELDS:
            default_index = options.index(field) if field in columns else 0
            selected = st.selectbox(field, options, index=default_index, key=f"map_{field}")
            if selected != "Auto detect":
                mapping[field] = selected
    return mapping


def render_data_readiness(mapped: dict[str, str], generated: list[str], notes: list[str]) -> None:
    with st.expander("Data readiness", expanded=False):
        if mapped:
            mapped_rows = pd.DataFrame(
                [{"Model field": key, "Uploaded column": value} for key, value in sorted(mapped.items())]
            )
            st.dataframe(mapped_rows, use_container_width=True, hide_index=True)
        if generated:
            st.info("Auto-filled retail fields: " + ", ".join(sorted(generated)))
        for note in notes:
            st.caption(note)


def priority_label(advice: dict[str, float | str]) -> str:
    if advice["status"] == "REORDER" and advice["urgency_score"] >= 10:
        return "Approve now"
    if advice["status"] == "REORDER":
        return "Review today"
    if advice["status"] == "OVERSTOCK":
        return "Reduce stock"
    return "Healthy"


def build_action_notes(row: pd.Series, advice: dict[str, float | str], demand_delta: float, unit_price: float) -> list[str]:
    notes: list[str] = []
    if row["Promotion"]:
        notes.append("Promotion is active, which supports a higher demand plan.")
    if row["Discount_Percent"] > 0:
        notes.append(f"Discounting at {row['Discount_Percent']:.0f}% should accelerate sell-through.")
    if row["Holiday"]:
        notes.append("Holiday traffic is built into this scenario.")
    if row["Weekend"]:
        notes.append("Weekend demand behavior is included.")
    if demand_delta > 5:
        notes.append(f"This scenario adds roughly {demand_delta:,.0f} units versus the base case.")
    elif demand_delta < -5:
        notes.append(f"This scenario removes roughly {abs(demand_delta):,.0f} units versus the base case.")
    if advice["shortage_units"] > 0:
        notes.append(
            f"Without action, expected unmet demand is about {advice['shortage_units']:,.0f} units "
            f"(≈ ₹{advice['shortage_units'] * unit_price:,.0f} in sales value)."
        )
    if advice["excess_units"] > 0:
        notes.append(
            f"Inventory is above target by about {advice['excess_units']:,.0f} units "
            f"(≈ ₹{advice['excess_units'] * unit_price:,.0f} tied up in stock)."
        )
    notes.append(f"Suggested action: {priority_label(advice)}.")
    return notes


def build_snapshot_actions(data: pd.DataFrame, pipeline, future_date, lead_time_days: int) -> pd.DataFrame:
    latest_rows = data[data["Date"] == data["Date"].max()].copy()
    records = []
    for _, row in latest_rows.iterrows():
        scenario = build_prediction_row(row.to_dict(), future_date)
        interval = predict_with_interval(pipeline, scenario)
        advice = inventory_recommendation(
            inventory=row["Inventory"],
            reorder_level=row["Reorder_Level"],
            predicted_demand=interval.prediction,
            safety_stock_ratio=0.20,
            lead_time_days=lead_time_days,
        )
        unit_price = float(max(row.get("Unit_Price", 0), 0))
        records.append({
            "Product_Name": row["Product_Name"],
            "Store_ID": row["Store_ID"],
            "Predicted_Demand": interval.prediction,
            "Suggested_Order": advice["recommended_order"],
            "Safety_Stock": advice["safety_stock"],
            "Projected_Balance": advice["projected_balance"],
            "Excess_Units": advice["excess_units"],
            "Urgency_Score": advice["urgency_score"],
            "Priority": priority_label(advice),
            "Revenue_At_Risk": round(advice["shortage_units"] * unit_price, 0),
            "Cash_Tied_Up": round(advice["excess_units"] * unit_price, 0),
            "Projected_Action": advice["status"],
        })
    snapshot = pd.DataFrame(records)
    if snapshot.empty:
        return snapshot
    return snapshot.sort_values(["Urgency_Score", "Revenue_At_Risk", "Suggested_Order"], ascending=False).reset_index(drop=True)


def reliability_label(metrics: dict[str, float]) -> str:
    if metrics["MAPE"] <= 20:
        return "Strong for planning"
    if metrics["MAPE"] <= 35:
        return "Usable with review"
    return "Needs tighter data before automation"


def main() -> None:
    st.title("SmartDemand AI")
    st.caption("Demand forecasting, replenishment planning, and inventory risk control for retail teams")

    with st.sidebar:
        st.header("Planning setup")
        uploaded_file = st.file_uploader("Upload retail sales data (.xlsx or .csv)", type=["xlsx", "csv"])
        trees = 300 #st.slider("Forecast model size", 100, 600, 300, 50)
        lead_time_days = st.slider("Supplier delivery time (days)", 1, 14, 1, 1)
        st.caption("The model learns from older dates and checks itself on later dates before planning.")

    raw, source_name = load_source(uploaded_file)
    if raw is None:
        st.info("Upload a retail sales workbook to start planning.")
        st.stop()

    manual_mapping = get_manual_mapping(list(raw.columns))
    standardized = standardize_dataset(raw, manual_mapping)
    errors = validate_dataset(raw, manual_mapping)
    if errors:
        st.error(" ".join(errors))
        st.caption("Tip: map at least your Date and Demand/Sales columns in the sidebar.")
        st.dataframe(pd.DataFrame({"Uploaded columns": list(raw.columns)}), use_container_width=True, hide_index=True)
        st.stop()

    source_key = f"{source_name}:{len(raw)}:{len(raw.columns)}"
    if st.session_state.get("forecast_source_key") != source_key:
        st.session_state["forecast_source_key"] = source_key
        st.session_state["forecast_ready"] = False

    with st.sidebar:
        if st.button("Start forecasting", type="primary", use_container_width=True):
            st.session_state["forecast_ready"] = True

    st.sidebar.success(f"Loaded {len(raw):,} rows from {source_name}")
    st.sidebar.caption(f"Planning for next day with {lead_time_days}-day supplier delivery time.")
    if standardized.generated_columns:
        st.sidebar.caption("Missing retail fields were filled automatically where needed.")

    if not st.session_state.get("forecast_ready", False):
        st.info("Dataset uploaded successfully. Click **Start forecasting** in the sidebar to begin.")
        st.stop()

    data = prepare_data(raw, manual_mapping)
    result = fit_cached(standardized.frame, trees)
    next_plan_date = (data["Date"].max() + pd.Timedelta(days=1)).date()
    snapshot = build_snapshot_actions(data=data, pipeline=result.pipeline, future_date=next_plan_date, lead_time_days=lead_time_days)

    overview, forecast, action_center, model_tab = st.tabs([
        "Overview", "Scenario Planner", "Action Center", "Forecast Quality"
    ])

    with overview:
        monthly = (
            data.assign(Month_Start=data["Date"].dt.to_period("M").dt.to_timestamp())
            .groupby("Month_Start", as_index=False)["Demand"]
            .sum()
        )
        daily = data.groupby("Date", as_index=False)["Demand"].sum()
        avg_daily_total = daily["Demand"].mean()
        total_predicted = snapshot["Predicted_Demand"].sum() if not snapshot.empty else 0
        revenue_at_risk = snapshot["Revenue_At_Risk"].sum() if not snapshot.empty else 0
        cash_tied_up = snapshot["Cash_Tied_Up"].sum() if not snapshot.empty else 0
        urgent_count = int((snapshot["Projected_Action"] == "REORDER").sum()) if not snapshot.empty else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecast demand for next day", f"{total_predicted:,.0f} units")
        c2.metric("Urgent approvals", urgent_count)
        c3.metric("Sales value at risk", f"₹{revenue_at_risk:,.0f}")
        c4.metric("Excess stock value", f"₹{cash_tied_up:,.0f}")

        render_data_readiness(standardized.column_mapping, standardized.generated_columns, standardized.notes)
        st.caption(
            f"Coverage: {data['Date'].nunique()} days • {data['Product_Name'].nunique()} products • "
            f"{data['Store_ID'].nunique()} stores • average daily demand {avg_daily_total:,.1f} units"
        )

        chart = go.Figure()
        chart.add_scatter(x=monthly["Month_Start"], y=monthly["Demand"], name="Demand", mode="lines+markers")
        chart.update_layout(
            title="Demand trend by month",
            template="plotly_white",
            xaxis_title="Month",
            yaxis_title="Units",
            height=360,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(chart, use_container_width=True)

        left, right = st.columns((3, 2))
        with left:
            st.subheader(f"Priority order queue — plan date {next_plan_date}")
            priority_view = snapshot.loc[:, ["Priority", "Product_Name", "Store_ID", "Predicted_Demand", "Suggested_Order", "Revenue_At_Risk"]].rename(columns={
                "Product_Name": "Product",
                "Store_ID": "Store",
                "Predicted_Demand": "Predicted demand (units)",
                "Suggested_Order": "Suggested order (units)",
                "Revenue_At_Risk": "Revenue at risk (₹)",
            })
            st.dataframe(priority_view.head(10), use_container_width=True, hide_index=True)
        with right:
            st.subheader("Store pressure summary")
            store_summary = (
                snapshot.groupby("Store_ID", as_index=False)[["Suggested_Order", "Revenue_At_Risk", "Cash_Tied_Up"]]
                .sum()
                .sort_values("Suggested_Order", ascending=False)
                .rename(columns={
                    "Store_ID": "Store",
                    "Suggested_Order": "Suggested order (units)",
                    "Revenue_At_Risk": "Revenue at risk (₹)",
                    "Cash_Tied_Up": "Cash tied up (₹)",
                })
            )
            st.dataframe(store_summary, use_container_width=True, hide_index=True)

    with forecast:
        st.subheader("Scenario planner")
        st.caption("Simulate tomorrow’s demand, check the stock position, and decide whether to approve the purchase.")
        reference = data.iloc[-1]
        with st.form("forecast_form"):
            a, b, c, d = st.columns(4)
            selected_product = a.selectbox("Product", sorted(data["Product_Name"].unique()))
            product_rows = data[data["Product_Name"] == selected_product]
            selected_store = b.selectbox("Store", sorted(product_rows["Store_ID"].unique()))
            selected_date = c.date_input("Forecast date", value=next_plan_date)
            promotion = d.toggle("Promotion active", value=bool(reference["Promotion"]))
            e, f, g, h = st.columns(4)
            inventory = e.number_input("Current inventory", min_value=0.0, value=float(max(reference["Inventory"], 0)), step=1.0)
            reorder_level = f.number_input("Reorder level", min_value=0.0, value=float(reference["Reorder_Level"]), step=1.0)
            discount = g.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=float(reference["Discount_Percent"]), step=1.0)
            holiday = h.toggle("Holiday", value=False)
            i, j = st.columns(2)
            temperature = i.number_input("Temperature (°C)", value=float(reference["Temperature"]), step=0.5)
            rainfall = j.number_input("Rainfall (mm)", min_value=0.0, value=float(max(reference["Rainfall"], 0)), step=0.5)
            submitted = st.form_submit_button("Generate plan", type="primary")

        if submitted:
            matching = data[(data["Product_Name"] == selected_product) & (data["Store_ID"] == selected_store)]
            baseline = matching.iloc[-1] if not matching.empty else reference

            base_row = build_prediction_row(baseline.to_dict(), selected_date)
            base_interval = predict_with_interval(result.pipeline, base_row)

            values = baseline.to_dict()
            values.update({
                "Product_Name": selected_product,
                "Store_ID": selected_store,
                "Promotion": int(promotion),
                "Inventory": inventory,
                "Reorder_Level": reorder_level,
                "Discount_Percent": discount,
                "Holiday": int(holiday),
                "Weekend": int(pd.Timestamp(selected_date).dayofweek >= 5),
                "Temperature": temperature,
                "Rainfall": rainfall,
            })
            row = build_prediction_row(values, selected_date)
            interval = predict_with_interval(result.pipeline, row)
            advice = inventory_recommendation(
                inventory=inventory,
                reorder_level=reorder_level,
                predicted_demand=interval.prediction,
                safety_stock_ratio=0.20,
                lead_time_days=lead_time_days,
            )
            demand_delta = interval.prediction - base_interval.prediction
            unit_price = float(max(baseline.get("Unit_Price", 0), 0))

            top = st.columns(4)
            top[0].metric("Base demand", f"{base_interval.prediction:,.0f} units")
            top[1].metric("Planned demand", f"{interval.prediction:,.0f} units", delta=f"{demand_delta:+.0f} units vs base")
            top[2].metric("Forecast range", f"{interval.lower:,.0f} - {interval.upper:,.0f} units")
            top[3].metric("Decision", status_badge(str(advice["status"])))

            second = st.columns(4)
            second[0].metric("Suggested order", f"{advice['recommended_order']:,.0f} units")
            second[1].metric("Stock cover", f"{advice['stock_cover_days']:,.1f} days")
            second[2].metric("Sales value at risk", f"₹{advice['shortage_units'] * unit_price:,.0f}")
            second[3].metric("Excess stock value", f"₹{advice['excess_units'] * unit_price:,.0f}")

            left, right = st.columns((2, 1))
            with left:
                scenario_table = pd.DataFrame([
                    {
                        "Scenario": "Base case",
                        "Demand (units)": round(base_interval.prediction, 1),
                        "Forecast low (units)": round(base_interval.lower, 1),
                        "Forecast high (units)": round(base_interval.upper, 1),
                        "Promotion": int(baseline["Promotion"]),
                        "Discount (%)": float(baseline["Discount_Percent"]),
                        "Inventory (units)": float(max(baseline["Inventory"], 0)),
                    },
                    {
                        "Scenario": "Planned case",
                        "Demand (units)": round(interval.prediction, 1),
                        "Forecast low (units)": round(interval.lower, 1),
                        "Forecast high (units)": round(interval.upper, 1),
                        "Promotion": int(promotion),
                        "Discount (%)": float(discount),
                        "Inventory (units)": float(inventory),
                    },
                ])
                st.dataframe(scenario_table, use_container_width=True, hide_index=True)
                st.info(
                    f"Lead-time demand: **{advice['lead_time_demand']:,.0f} units** • "
                    f"Safety stock: **{advice['safety_stock']:,.0f} units** • "
                    f"Target stock: **{advice['target_stock']:,.0f} units** • "
                    f"Projected closing stock: **{advice['projected_balance']:,.0f} units**"
                )
            with right:
                st.markdown("**Approval panel**")
                st.success(priority_label(advice))
                for note in build_action_notes(row.iloc[0], advice, demand_delta, unit_price):
                    st.write(f"- {note}")

    with action_center:
        st.subheader("Action center")
        top = st.columns(4)
        top[0].metric("Approve now", int((snapshot["Priority"] == "Approve now").sum()) if not snapshot.empty else 0)
        top[1].metric("Review today", int((snapshot["Priority"] == "Review today").sum()) if not snapshot.empty else 0)
        top[2].metric("Reduce stock", int((snapshot["Priority"] == "Reduce stock").sum()) if not snapshot.empty else 0)
        top[3].metric("Recommended order volume", f"{snapshot['Suggested_Order'].sum():,.0f} units" if not snapshot.empty else "0 units")

        approve_now, review_today, reduce_stock = st.tabs(["Approve now", "Review today", "Reduce stock"])
        with approve_now:
            urgent = snapshot[snapshot["Priority"] == "Approve now"].loc[:, ["Product_Name", "Store_ID", "Predicted_Demand", "Suggested_Order", "Revenue_At_Risk", "Urgency_Score"]].rename(columns={
                "Product_Name": "Product",
                "Store_ID": "Store",
                "Predicted_Demand": "Predicted demand (units)",
                "Suggested_Order": "Suggested order (units)",
                "Revenue_At_Risk": "Revenue at risk (₹)",
                "Urgency_Score": "Urgency score",
            })
            st.dataframe(urgent, use_container_width=True, hide_index=True)
        with review_today:
            monitor = snapshot[snapshot["Priority"] == "Review today"].loc[:, ["Product_Name", "Store_ID", "Predicted_Demand", "Projected_Balance", "Safety_Stock", "Suggested_Order"]].rename(columns={
                "Product_Name": "Product",
                "Store_ID": "Store",
                "Predicted_Demand": "Predicted demand (units)",
                "Projected_Balance": "Projected balance (units)",
                "Safety_Stock": "Safety stock (units)",
                "Suggested_Order": "Suggested order (units)",
            })
            st.dataframe(monitor, use_container_width=True, hide_index=True)
        with reduce_stock:
            excess = snapshot[snapshot["Priority"] == "Reduce stock"].loc[:, ["Product_Name", "Store_ID", "Excess_Units", "Cash_Tied_Up"]].rename(columns={
                "Product_Name": "Product",
                "Store_ID": "Store",
                "Excess_Units": "Excess units",
                "Cash_Tied_Up": "Cash tied up (₹)",
            })
            st.dataframe(excess, use_container_width=True, hide_index=True)

    with model_tab:
        st.subheader("Forecast quality")
        reliability = reliability_label(result.metrics)
        top = st.columns(5)
        top[0].metric("MAE", f"{result.metrics['MAE']:.2f} units")
        top[1].metric("RMSE", f"{result.metrics['RMSE']:.2f} units")
        top[2].metric("R²", f"{result.metrics['R2']:.3f}")
        top[3].metric("MAPE", f"{result.metrics['MAPE']:.1f}%")
        top[4].metric("Planning status", reliability)
        st.caption(f"Training rows: {result.train_rows:,} • unseen test rows: {result.test_rows:,}")

        actual_pred = result.test_frame.groupby("Date", as_index=False)[["Demand", "Predicted_Demand"]].mean()
        chart = go.Figure()
        chart.add_scatter(x=actual_pred["Date"], y=actual_pred["Demand"], name="Actual demand")
        chart.add_scatter(x=actual_pred["Date"], y=actual_pred["Predicted_Demand"], name="Predicted demand")
        chart.update_layout(
            title="Actual vs predicted demand on unseen dates",
            template="plotly_white",
            yaxis_title="Units",
            height=360,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(chart, use_container_width=True)

        drivers = result.feature_importance.sort_values("Importance", ascending=True)
        driver_chart = go.Figure(go.Bar(x=drivers["Importance"], y=drivers["Feature"], orientation="h"))
        driver_chart.update_layout(
            title="Main demand drivers",
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(driver_chart, use_container_width=True)
        st.warning("This prototype is best used as decision support. Cleaner store history and richer supply inputs improve reliability.")


if __name__ == "__main__":
    main()
