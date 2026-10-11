import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Overrun Risk", layout="wide")

model = joblib.load("delay_model.joblib")
features = ["built_area", "modul_price", "weeks_duration",
            "DETACHED", "COLLECTIVE", "COMMERCIAL", "OTHERS"]


# ---------- Guardrail helpers ----------

def sanity_warnings(data, planned_col, actual_col):
    """Checks that the planned/actual choice and the data look believable."""
    msgs = []
    average = data["percent_over"].mean()
    share_over = data["overrun"].mean()
    if average < -20:
        msgs.append(f"On average these projects finished {abs(average):.0f}% UNDER budget. "
                    "That is unusual in real projects. Check you picked the right planned and actual columns.")
    if average > 50:
        msgs.append(f"On average these projects finished {average:.0f}% OVER budget. "
                    "That is unusual. Check you picked the right planned and actual columns.")
    if share_over < 0.05 or share_over > 0.95:
        msgs.append("Almost every project falls on the same side of its budget, "
                    "so a model has very little to learn from.")
    if len(data) < 100:
        msgs.append(f"Only {len(data)} projects. With so few, any pattern may be coincidence. Aim for 100 or more.")
    return msgs


def leakage_warnings(data, chosen, planned_col, actual_col):
    """Warns when chosen inputs look like they are only known after the project ends."""
    msgs = []
    for col in chosen:
        if any(word in col.lower() for word in ("actual", "final", "outturn")):
            msgs.append(f"'{col}' sounds like a final figure, which is only known after the project ends.")
    check = [c for c in chosen if c != planned_col]
    if check:
        X = data[check].fillna(data[check].median())
        r2 = LinearRegression().fit(X, data[actual_col]).score(X, data[actual_col])
        if r2 > 0.95:
            msgs.append(f"These inputs explain {r2 * 100:.0f}% of the actual cost by themselves. "
                        "They probably contain the answer (for example, cost parts that add up to the final cost). "
                        "Use only things known BEFORE the project starts.")
    return msgs


def verdict(acc, base):
    """Decides whether a trained model is worth using."""
    if acc >= 0.97:
        return "suspicious", ("Accuracy this high is a warning sign, not a success. "
                              "Check that no input contains the answer.")
    if acc > base + 0.02:
        return "useful", "The model beats the baseline, so these inputs carry some real signal."
    return "not_useful", ("The model does not clearly beat the baseline, so these inputs add little. "
                          "Scoring is turned off to avoid showing a meaningless number.")


# ---------- Sidebar ----------

st.sidebar.title("Overrun Risk")
st.sidebar.caption("Early warning for project overruns")
st.warning("Demo link: please do not upload confidential data.")

demo_tab, data_tab = st.tabs(["Demo model", "Your data"])

# ---------- Demo tab ----------

with demo_tab:
    st.title("Construction Delay Risk (Demo)")
    st.warning("Demo built on synthetic data. These are not real predictions.")

    built_area = st.number_input("Built area (m²)", min_value=50.0, value=5000.0, step=100.0)
    modul_price = st.number_input("Cost per m²", min_value=100.0, value=700.0, step=10.0)
    weeks_duration = st.number_input("Planned duration (weeks)", min_value=1, value=50, step=1)
    typology = st.selectbox("Building type", ["DETACHED", "COLLECTIVE", "COMMERCIAL", "OTHERS"])

    if st.button("Check delay risk", type="primary"):
        row = pd.DataFrame({
            "built_area": [built_area],
            "modul_price": [modul_price],
            "weeks_duration": [weeks_duration],
            "DETACHED": [int(typology == "DETACHED")],
            "COLLECTIVE": [int(typology == "COLLECTIVE")],
            "COMMERCIAL": [int(typology == "COMMERCIAL")],
            "OTHERS": [int(typology == "OTHERS")],
        })
        risk = model.predict_proba(row)[0][1]
        st.metric("Delay risk", f"{risk * 100:.1f}%")
        st.progress(float(risk))

    coefs = model.named_steps["logisticregression"].coef_[0]
    st.subheader("What the model weighs most")
    st.caption("Bars above zero push the risk up. Bars below zero push it down.")
    st.bar_chart(pd.Series(coefs, index=features))

# ---------- Your data tab ----------

with data_tab:
    st.header("Analyse your own projects")

    if os.path.exists("demo_company_projects.csv"):
        with open("demo_company_projects.csv", "rb") as f:
            st.download_button("Download a sample CSV to try", f.read(),
                               "demo_company_projects.csv", "text/csv")

    uploaded = st.file_uploader("Upload a CSV of past projects", type="csv")

    if uploaded is None:
        st.info("Upload a CSV to begin. It needs a planned cost column and an actual cost column.")
        st.stop()

    try:
        data = pd.read_csv(uploaded)
    except Exception:
        st.error("That file could not be read as a CSV.")
        st.stop()

    numeric_cols = data.select_dtypes("number").columns.tolist()
    if len(numeric_cols) < 2:
        st.error("The file needs at least two number columns (planned and actual cost).")
        st.stop()

    left, right = st.columns(2)
    planned_col = left.selectbox("Which column is the PLANNED cost?", numeric_cols)
    actual_col = right.selectbox("Which column is the ACTUAL cost?", numeric_cols, index=1)

    if planned_col == actual_col:
        st.warning("Planned and actual are the same column. Pick two different ones.")
        st.stop()

    data = data[data[planned_col] > 0].copy()
    if len(data) == 0:
        st.error("No rows have a planned cost above zero.")
        st.stop()

    data["percent_over"] = (data[actual_col] - data[planned_col]) / data[planned_col] * 100
    data["overrun"] = data["percent_over"] > 0

    st.write("Rows used:", len(data))
    m1, m2, m3 = st.columns(3)
    m1.metric("Projects over budget", f"{data['overrun'].mean() * 100:.1f}%")
    m2.metric("Average overrun", f"{data['percent_over'].mean():.1f}%")
    m3.metric("Worst overrun", f"{data['percent_over'].max():.1f}%")

    for msg in sanity_warnings(data, planned_col, actual_col):
        st.warning(msg)

    st.dataframe(data.head(10))
    st.download_button("Download results (CSV)", data.to_csv(index=False),
                       "overrun_results.csv", "text/csv", type="primary")

    # ----- Where do overruns happen? -----
    group_options = [c for c in data.columns
                     if c not in numeric_cols and c != "overrun"
                     and 2 <= data[c].nunique() <= 20]

    if group_options:
        st.subheader("Where do overruns happen?")
        group_col = st.selectbox("Break down by", group_options)
        summary = (data.groupby(group_col)["percent_over"]
                   .agg(projects="count", average_overrun="mean")
                   .round(1)
                   .sort_values("average_overrun", ascending=False))
        st.bar_chart(summary["average_overrun"])
        st.dataframe(summary)

        reliable = summary[summary["projects"] >= 5]
        if len(reliable) > 0:
            top = reliable.index[0]
            st.info(f"Highest average overrun among groups with 5 or more projects: "
                    f"{top} ({reliable.iloc[0]['average_overrun']:.1f}% over plan, "
                    f"{int(reliable.iloc[0]['projects'])} projects).")
        if (summary["projects"] < 5).any():
            st.caption("Groups with fewer than 5 projects are too small to trust.")

    # ----- Train -----
    st.subheader("Train a model on this data")
    st.caption("Only pick columns known BEFORE a project finishes. Final costs hand the model the answer.")
    options = [c for c in numeric_cols if c != actual_col]
    chosen = st.multiselect("Input columns for the model", options)

    for msg in leakage_warnings(data, chosen, planned_col, actual_col):
        st.error(msg)

    signature = (uploaded.name, uploaded.size, planned_col, actual_col, tuple(chosen))

    if st.button("Train model", type="primary"):
        counts = data["overrun"].value_counts()
        if not chosen:
            st.error("Tick at least one input column first.")
        elif len(data) < 30 or len(counts) < 2 or counts.min() < 5:
            st.error("Need at least 30 projects, with at least 5 over budget and 5 not.")
        else:
            X = data[chosen].fillna(data[chosen].median())
            y = data["overrun"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            new_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            new_model.fit(X_train, y_train)

            acc = accuracy_score(y_test, new_model.predict(X_test))
            base = max(y_test.mean(), 1 - y_test.mean())
            status, message = verdict(acc, base)

            st.session_state["result"] = {
                "sig": signature, "acc": acc, "base": base, "n_test": len(y_test),
                "status": status, "message": message, "model": new_model,
            }

    # ----- Show results (stay visible after other clicks) -----
    result = st.session_state.get("result")
    if result is not None and result["sig"] == signature:
        a1, a2 = st.columns(2)
        a1.metric("Model accuracy on unseen projects", f"{result['acc'] * 100:.1f}%")
        a2.metric("Baseline (always guess the common answer)", f"{result['base'] * 100:.1f}%")
        st.caption(f"Tested on {result['n_test']} projects the model had not seen. "
                   "Gaps of a few points can be luck when the test set is small.")

        if result["status"] == "useful":
            st.success(result["message"])
        elif result["status"] == "suspicious":
            st.error(result["message"])
        else:
            st.warning(result["message"])

        # ----- Score a new project (only for a usable model) -----
        if result["status"] == "useful":
            st.subheader("Score a new project")
            st.caption("Enter the details known before the job starts.")
            new_values = {}
            for col in chosen:
                new_values[col] = st.number_input(
                    col, value=float(data[col].median()), key="new_" + col)

            if st.button("Score this project", type="primary"):
                new_row = pd.DataFrame([new_values])
                p = result["model"].predict_proba(new_row)[0][1]
                st.metric("Chance of going over budget", f"{p * 100:.1f}%")
                st.progress(float(p))
                st.caption("A rough guide based on your past projects, not a guarantee.")