import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from math import pi
from io import BytesIO
from datetime import datetime
import os

from alg_bez_filtr import find_non_dominated_points as alg_no_filter
from alg_filtr import find_non_dominated_points as alg_filter
from alg_pkt_idealny import find_non_dominated_points as alg_ideal_point

# --- Algorithm mapping ---
algorithms = {
    "Algorithm without filtering": alg_no_filter,
    "Algorithm with filtering": alg_filter,
    "Ideal point algorithm": alg_ideal_point,
}

# --- Streamlit setup ---
st.set_page_config(page_title="Non-Dominated Points Optimizer", layout="wide", page_icon="🎯")

with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🎯 Non-Dominated Points Optimization")
st.write("**Authors:** Michał Cynarski & Bartłomiej Barszczak")
st.sidebar.header("Criteria Configuration")

# --- Criteria Editor ---
criteria = []
if "criteria_count" not in st.session_state:
    st.session_state.criteria_count = 2

with st.sidebar.form("criteria_form"):
    criteria_count = st.number_input("Number of criteria", min_value=2, max_value=10, step=1, value=st.session_state.criteria_count)
    st.session_state.criteria_count = criteria_count
    for i in range(criteria_count):
        col1, col2 = st.columns([3, 2])
        with col1:
            criterion_name = st.text_input(f"Criterion name {i + 1}", value=f"Criterion {i + 1}")
        with col2:
            direction = st.selectbox(f"Direction {i + 1}", ["Min", "Max"], index=0)
        criteria.append((criterion_name, direction))
    submit_button = st.form_submit_button(label="Confirm criteria")

# --- Data generation sidebar ---
st.sidebar.header("Data Generation")
data_config = st.sidebar.expander("Data Settings")
with data_config:
    if "data_count" not in st.session_state:
        st.session_state.data_count = 10
    if "data_distribution" not in st.session_state:
        st.session_state.data_distribution = "Uniform"
    if "data_range" not in st.session_state:
        st.session_state.data_range = (10, 50)

    data_count = st.number_input("Number of points", min_value=1, max_value=1000, value=st.session_state.data_count)
    st.session_state.data_count = data_count

    data_distribution = st.selectbox("Data distribution", ["Uniform", "Gaussian", "Exponential", "Poisson"], index=0)
    st.session_state.data_distribution = data_distribution

    data_range = st.slider("Value range", min_value=0, max_value=100, value=st.session_state.data_range)
    st.session_state.data_range = data_range

    lambda_poisson = st.number_input("λ parameter (Poisson)", min_value=0.1, max_value=10.0, value=2.0) if data_distribution == "Poisson" else None
    sigma_gauss = st.number_input("Standard deviation (Gaussian)", min_value=0.1, max_value=20.0, value=10.0) if data_distribution == "Gaussian" else None
    mu_gauss = st.number_input("Mean value (Gaussian)", min_value=0.0, max_value=100.0, value=30.0) if data_distribution == "Gaussian" else None
    lambda_exponential = st.number_input("λ parameter (Exponential)", min_value=0.1, max_value=10.0, value=1.0) if data_distribution == "Exponential" else None

generate_button = st.sidebar.button("Generate random data")

if generate_button:
    if data_distribution == "Uniform":
        data = np.random.uniform(data_range[0], data_range[1], (data_count, criteria_count))
    elif data_distribution == "Gaussian":
        data = np.random.normal(mu_gauss, sigma_gauss, (data_count, criteria_count))
    elif data_distribution == "Exponential":
        data = np.random.exponential(1 / lambda_exponential, (data_count, criteria_count))
        data = data + data_range[0]
    elif data_distribution == "Poisson":
        data = np.random.poisson(lambda_poisson, (data_count, criteria_count))

    data = np.clip(data, data_range[0], data_range[1])
    df = pd.DataFrame(data, columns=[c[0] for c in criteria])
    st.session_state["data"] = df
else:
    df = st.session_state.get("data", pd.DataFrame(np.random.rand(10, criteria_count), columns=[f"Criterion {i+1}" for i in range(criteria_count)]))
    st.session_state["data"] = df

# --- Display data ---
st.subheader("Input Data")
sort_column = st.selectbox("Sort by criterion", [c[0] for c in criteria])
sort_order = st.radio("Sort order", ["Ascending", "Descending"])
if st.button("Sort"):
    df = df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))
st.dataframe(df)

# --- Algorithm selection ---
st.sidebar.header("Algorithm Selection")
selected_algorithm_name = st.sidebar.selectbox("Algorithm", list(algorithms.keys()))
selected_algorithm = algorithms[selected_algorithm_name]

# --- Run algorithm ---
if st.button("Find non-dominated points"):
    points = df.values.tolist()
    start_time = time.time()
    non_dominated_points, num_comparisons = selected_algorithm(points)
    execution_time = time.time() - start_time

    st.session_state["non_dominated_points"] = non_dominated_points
    st.session_state["num_comparisons"] = num_comparisons
    st.session_state["execution_time"] = execution_time

# --- Display results ---
if "non_dominated_points" in st.session_state:
    non_dominated_points = st.session_state["non_dominated_points"]
    num_comparisons = st.session_state["num_comparisons"]
    execution_time = st.session_state["execution_time"]

    st.subheader(f"Non-Dominated Points (Count: {len(non_dominated_points)})")
    formatted_points_html = "<br>".join(
        [f"<span style='color: white;'>{i + 1}: ({', '.join(map(str, p))})</span>" for i, p in
         enumerate(non_dominated_points)]
    )
    st.markdown(formatted_points_html, unsafe_allow_html=True)

    st.write(f"Number of comparisons: {num_comparisons}")
    st.write(f"Algorithm execution time: {execution_time:.4f} seconds")

    st.subheader("Results Visualization")

    # 2D visualization
    if criteria_count == 2:
        fig, ax = plt.subplots()
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], label="All points")
        ax.scatter([p[0] for p in non_dominated_points], [p[1] for p in non_dominated_points], color='red', label="Non-dominated points")
        ax.set_xlabel(f"{criteria[0][0]} ({criteria[0][1]})")
        ax.set_ylabel(f"{criteria[1][0]} ({criteria[1][1]})")
        ax.legend()
        st.pyplot(fig)

    # 3D visualization
    elif criteria_count == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], label="All points")
        ax.scatter([p[0] for p in non_dominated_points], [p[1] for p in non_dominated_points], [p[2] for p in non_dominated_points], color='red', label="Non-dominated points")
        ax.set_xlabel(f"{criteria[0][0]} ({criteria[0][1]})")
        ax.set_ylabel(f"{criteria[1][0]} ({criteria[1][1]})")
        ax.set_zlabel(f"{criteria[2][0]} ({criteria[2][1]})")
        ax.legend()
        st.pyplot(fig)

    # Visualization for N > 3
    else:
        vis_option = st.selectbox("Select visualization method (for N > 3)", ["Dimensionality reduction (PCA)", "Pairplot matrix", "Radar chart"])

        if vis_option == "Dimensionality reduction (PCA)":
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(df)
            fig, ax = plt.subplots()
            ax.scatter(reduced_data[:, 0], reduced_data[:, 1], label="All points")
            ax.scatter(pca.transform(non_dominated_points)[:, 0], pca.transform(non_dominated_points)[:, 1], color='red', label="Non-dominated points")
            ax.set_xlabel("Principal component 1")
            ax.set_ylabel("Principal component 2")
            ax.legend()
            st.pyplot(fig)

        elif vis_option == "Pairplot matrix":
            fig = sns.pairplot(df)
            st.pyplot(fig)

        elif vis_option == "Radar chart":
            num_vars = len(criteria)
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            for point in non_dominated_points:
                values = point + point[:1]
                ax.plot(angles, values, linewidth=1, linestyle='solid')
                ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([c[0] for c in criteria])
            st.pyplot(fig)

# --- Function to save benchmark results ---
def save_to_excel_local(benchmark_df, criteria, data_df, batch_count, data_distribution, data_count, data_range,
                        lambda_poisson, sigma_gauss, mu_gauss, lambda_exponential):
    folder_path = "benchmark_results"
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_path, f"benchmark_{data_distribution}_{batch_count}_{timestamp}.xlsx")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        start_row = 0
        benchmark_df.to_excel(writer, index=False, startrow=start_row, sheet_name="Benchmark & Data")
        start_row += len(benchmark_df) + 3

        criteria_df = pd.DataFrame({
            "Criterion": [c[0] for c in criteria],
            "Direction": [c[1] for c in criteria]
        })
        criteria_df.to_excel(writer, index=False, startrow=start_row, sheet_name="Benchmark & Data")
        start_row += len(criteria_df) + 3

        settings_df = pd.DataFrame({
            "Parameter": ["Batch count", "Distribution", "Point count", "Value range",
                          "λ Poisson", "σ Gaussian", "μ Gaussian", "λ Exponential"],
            "Value": [batch_count, data_distribution, data_count, str(data_range),
                      lambda_poisson, sigma_gauss, mu_gauss, lambda_exponential]
        })
        settings_df.to_excel(writer, index=False, startrow=start_row, sheet_name="Benchmark & Data")
        start_row += len(settings_df) + 3

        data_df.to_excel(writer, index=False, startrow=start_row, sheet_name="Benchmark & Data")

    with open(file_path, "wb") as f:
        f.write(output.getvalue())

    output.seek(0)
    return file_path, output.getvalue()

# --- Benchmark Section ---
st.sidebar.header("Benchmark Testing")
batch_count = st.sidebar.number_input("Number of batches", min_value=1, max_value=1000, value=50)
if st.sidebar.button("Run Benchmark"):
    benchmark_results = []
    for alg_name, alg_func in algorithms.items():
        total_comparisons = 0
        total_execution_time = 0

        for _ in range(batch_count):
            if data_distribution == "Uniform":
                batch_data = np.random.uniform(data_range[0], data_range[1], (data_count, criteria_count))
            elif data_distribution == "Gaussian":
                batch_data = np.random.normal(mu_gauss, sigma_gauss, (data_count, criteria_count))
            elif data_distribution == "Exponential":
                batch_data = np.random.exponential(1 / lambda_exponential, (data_count, criteria_count))
                batch_data = batch_data + data_range[0]
            elif data_distribution == "Poisson":
                batch_data = np.random.poisson(lambda_poisson, (data_count, criteria_count))

            batch_data = np.clip(batch_data, data_range[0], data_range[1])
            points = batch_data.tolist()

            start_time = time.time()
            non_dominated_points, num_comparisons = alg_func(points)
            execution_time = time.time() - start_time

            total_comparisons += num_comparisons
            total_execution_time += execution_time

        benchmark_results.append({
            "Algorithm": alg_name,
            "Non-dominated points (avg)": len(non_dominated_points),
            "Comparisons (avg)": total_comparisons / batch_count,
            "Execution time (s)": total_execution_time / batch_count
        })

    benchmark_df = pd.DataFrame(benchmark_results)
    st.session_state["benchmark_df"] = benchmark_df
    st.subheader("Benchmark Results")
    st.dataframe(benchmark_df)

if "benchmark_df" in st.session_state and not st.session_state["benchmark_df"].empty:
    file_path, excel_data = save_to_excel_local(
        st.session_state["benchmark_df"],
        criteria=criteria,
        data_df=st.session_state["data"],
        batch_count=batch_count,
        data_distribution=data_distribution,
        data_count=data_count,
        data_range=data_range,
        lambda_poisson=lambda_poisson,
        sigma_gauss=sigma_gauss,
        mu_gauss=mu_gauss,
        lambda_exponential=lambda_exponential
    )

    st.download_button(
        label="Download Benchmark Results (Excel)",
        data=excel_data,
        file_name=os.path.basename(file_path),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
