"""Bank capital stress test: VARX(1) + ARDL + bootstrap Monte Carlo.

Historical data and the 12-month stress scenario are stored separately.
ARDL coefficients are taken from the original EViews specification and are
not re-estimated in Python.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# 1) ИСТОРИЧЕСКИЕ ДАННЫЕ
HISTORY_PATH = DATA_DIR / "main_date.xlsx"

# 2) СТРЕСС-ШОКИ НА 12 МЕСЯЦЕВ
SHOCKS_PATH = DATA_DIR / "Forecast_date.xlsx"

OUTPUT_PATH = RESULTS_DIR / "stress_test_results.xlsx"
CHART_PATH = RESULTS_DIR / "terminal_ncap_distribution.png"

N_SIMS = 10_000
RANDOM_SEED = 42

ARDL_PARAMS = {
    "NCAP(-1)": 0.805,
    "R": 0.190,
    "RATE": 0.078,
    "RATE(-1)": 0.138,
    "LCR": 0.001,
    "LCR(-1)": 0.017,
    "LCR(-2)": -0.009,
    "LCR(-3)": -0.010,
    "CURRENCY": 0.009,
    "CURRENCY(-1)": -0.106,
    "IPI": 0.026,
    "IPC": -0.146,
    "C": 1.031,
}

COLUMN_NAMES = {
    "НДК": "NCAP",
    "КАПИТАЛ": "NCAP",
    "РЕНТАБЕЛЬНОСТЬ": "R",
    "СТАВКА": "RATE",
    "ЛИКВИДНОСТЬ": "LCR",
    "КУРС": "CURRENCY",
    "ИПП": "IPI",
    "ИПЦ": "IPC",
    "ДАТА": "DATE",
}

HISTORY_COLUMNS = ["NCAP", "R", "RATE", "LCR", "CURRENCY", "IPI", "IPC"]
SHOCK_COLUMNS = ["IPC", "IPI", "CURRENCY"]


def clean_column_names(df):
    """Standardize Excel column names."""
    df = df.copy()
    df.columns = [str(column).strip().upper() for column in df.columns]
    return df.rename(columns=COLUMN_NAMES)


def prepare_dates(df):
    """Sort by DATE when a complete date column is available."""
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", dayfirst=True)
        if df["DATE"].notna().all():
            df = df.sort_values("DATE")
    return df


def check_columns(df, required, file_label):
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"В {file_label} отсутствуют колонки: " + ", ".join(missing)
        )


def load_history(file_path):
    """Load the historical sample."""
    if not file_path.exists():
        raise FileNotFoundError(f"Не найден файл с историческими данными:\n{file_path}")

    df = prepare_dates(clean_column_names(pd.read_excel(file_path)))
    check_columns(df, HISTORY_COLUMNS, "историческом файле")

    for column in HISTORY_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    history = df.loc[
        df["NCAP"].notna() & (df["NCAP"] > 0), HISTORY_COLUMNS
    ].copy().reset_index(drop=True)

    if history.empty:
        raise ValueError("После загрузки не осталось исторических наблюдений.")

    missing_values = history.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        text = ", ".join(f"{column}: {count}" for column, count in missing_values.items())
        raise ValueError("В исторической выборке есть пропуски: " + text)

    return history


def load_shocks(file_path):
    """Load exactly 12 months of IPC, IPI and CURRENCY stress inputs."""
    if not file_path.exists():
        raise FileNotFoundError(f"Не найден файл со стресс-шоками:\n{file_path}")

    df = prepare_dates(clean_column_names(pd.read_excel(file_path)))
    check_columns(df, SHOCK_COLUMNS, "файле стресс-шоков")

    for column in SHOCK_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    shocks = df[SHOCK_COLUMNS].copy().dropna().reset_index(drop=True)
    if len(shocks) != 12:
        raise ValueError(
            "В файле стресс-шоков должно быть ровно 12 полностью заполненных "
            f"строк. Сейчас найдено: {len(shocks)}."
        )

    return shocks


# -----------------------------------------------------------------------------
# VARX(1): forecast R, LCR and RATE
# -----------------------------------------------------------------------------
def run_varx_forecast(history, shocks):
    """Estimate the reconstructed VARX(1) system and forecast 12 months."""
    endog = ["R", "LCR", "RATE"]
    exog = ["IPC", "IPI", "CURRENCY"]

    diff_df = history[endog].diff().dropna()
    exog_history = history.loc[diff_df.index, exog]

    X = pd.DataFrame(index=diff_df.index)
    for column in endog:
        X[f"d_{column}_lag1"] = diff_df[column].shift(1)
    for column in exog:
        X[column] = exog_history[column]
    X["const"] = 1.0

    valid_index = X.dropna().index
    X_clean = X.loc[valid_index]
    models = {
        column: sm.OLS(diff_df.loc[valid_index, column], X_clean).fit()
        for column in endog
    }

    current_diff = {column: float(diff_df[column].iloc[-1]) for column in endog}
    predicted_diffs = []

    for month in range(len(shocks)):
        row = {f"d_{column}_lag1": current_diff[column] for column in endog}
        row.update({column: float(shocks.loc[month, column]) for column in exog})
        row["const"] = 1.0

        step_X = pd.DataFrame([row])[X_clean.columns]
        next_diff = {
            column: float(models[column].predict(step_X).iloc[0])
            for column in endog
        }
        predicted_diffs.append(next_diff)
        current_diff = next_diff

    predicted_diffs = pd.DataFrame(predicted_diffs)
    predicted_levels = predicted_diffs.cumsum() + history[endog].iloc[-1].values

    full_scenario = pd.concat(
        [predicted_levels.reset_index(drop=True), shocks.reset_index(drop=True)],
        axis=1,
    )
    full_scenario.index = np.arange(1, len(full_scenario) + 1)
    full_scenario.index.name = "MONTH"
    return full_scenario


# -----------------------------------------------------------------------------
# ARDL residuals from fixed EViews coefficients
# -----------------------------------------------------------------------------
def calculate_ardl_residuals(history, params):
    """Calculate centered ARDL residuals without re-estimating coefficients."""
    y = history["NCAP"]
    X = pd.DataFrame(index=history.index)
    X["NCAP(-1)"] = history["NCAP"].shift(1)
    X["R"] = history["R"]
    X["RATE"] = history["RATE"]
    X["RATE(-1)"] = history["RATE"].shift(1)
    X["LCR"] = history["LCR"]
    X["LCR(-1)"] = history["LCR"].shift(1)
    X["LCR(-2)"] = history["LCR"].shift(2)
    X["LCR(-3)"] = history["LCR"].shift(3)
    X["CURRENCY"] = history["CURRENCY"]
    X["CURRENCY(-1)"] = history["CURRENCY"].shift(1)
    X["IPI"] = history["IPI"]
    X["IPC"] = history["IPC"]
    X["C"] = 1.0

    valid_index = X.dropna().index
    X_clean = X.loc[valid_index]
    y_clean = y.loc[valid_index]

    fitted = np.zeros(len(X_clean))
    for variable, coefficient in params.items():
        fitted += coefficient * X_clean[variable].values

    residuals = y_clean.values - fitted
    return residuals - np.mean(residuals)


# -----------------------------------------------------------------------------
# Stress test and bootstrap Monte Carlo
# -----------------------------------------------------------------------------
def simulate_stress_test(
    history,
    full_scenario,
    params,
    residuals,
    n_sims=10_000,
    seed=42,
):
    """Build deterministic NCAP and bootstrap Monte Carlo paths."""
    horizon = len(full_scenario)
    alpha = params["NCAP(-1)"]

    history_tail = history.iloc[-4:].copy().reset_index(drop=True)
    scenario = full_scenario.copy().reset_index(drop=True)
    evaluation = pd.concat([history_tail, scenario], ignore_index=True)
    offset = len(history_tail)

    deterministic_inputs = np.zeros(horizon)
    for month in range(horizon):
        i = offset + month
        deterministic_inputs[month] = (
            params["C"]
            + params["R"] * evaluation.loc[i, "R"]
            + params["RATE"] * evaluation.loc[i, "RATE"]
            + params["RATE(-1)"] * evaluation.loc[i - 1, "RATE"]
            + params["LCR"] * evaluation.loc[i, "LCR"]
            + params["LCR(-1)"] * evaluation.loc[i - 1, "LCR"]
            + params["LCR(-2)"] * evaluation.loc[i - 2, "LCR"]
            + params["LCR(-3)"] * evaluation.loc[i - 3, "LCR"]
            + params["CURRENCY"] * evaluation.loc[i, "CURRENCY"]
            + params["CURRENCY(-1)"] * evaluation.loc[i - 1, "CURRENCY"]
            + params["IPI"] * evaluation.loc[i, "IPI"]
            + params["IPC"] * evaluation.loc[i, "IPC"]
        )

    deterministic_path = np.zeros(horizon)
    current_ncap = float(history["NCAP"].iloc[-1])
    for month in range(horizon):
        current_ncap = alpha * current_ncap + deterministic_inputs[month]
        deterministic_path[month] = current_ncap

    rng = np.random.default_rng(seed)
    random_shocks = rng.choice(residuals, size=(n_sims, horizon), replace=True)
    simulation_paths = np.zeros((n_sims, horizon))
    current_paths = np.full(n_sims, float(history["NCAP"].iloc[-1]))

    for month in range(horizon):
        current_paths = (
            alpha * current_paths
            + deterministic_inputs[month]
            + random_shocks[:, month]
        )
        simulation_paths[:, month] = current_paths

    return deterministic_path, simulation_paths


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def calculate_summary(deterministic_path, simulation_paths):
    terminal_ncap = simulation_paths[:, -1]
    q05 = np.percentile(terminal_ncap, 5)
    median = np.percentile(terminal_ncap, 50)
    q95 = np.percentile(terminal_ncap, 95)

    summary = pd.DataFrame(
        {
            "METRIC": [
                "Minimum deterministic NCAP",
                "Deterministic terminal NCAP",
                "Terminal Q05",
                "Terminal median",
                "Terminal Q95",
                "Q95 - Q05",
            ],
            "VALUE": [
                np.min(deterministic_path),
                deterministic_path[-1],
                q05,
                median,
                q95,
                q95 - q05,
            ],
        }
    )
    return summary, q05, median, q95


def save_results(full_scenario, deterministic_path, simulation_paths, summary):
    scenario_output = full_scenario.copy()
    scenario_output["NCAP_DETERMINISTIC"] = deterministic_path

    terminal_distribution = pd.DataFrame(
        {
            "SIMULATION": np.arange(1, len(simulation_paths) + 1),
            "TERMINAL_NCAP": simulation_paths[:, -1],
        }
    )

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        scenario_output.to_excel(writer, sheet_name="Scenario")
        summary.to_excel(writer, sheet_name="Summary", index=False)
        terminal_distribution.to_excel(writer, sheet_name="Terminal_MC", index=False)


def plot_distribution(simulation_paths, q05, median, q95):
    terminal_ncap = simulation_paths[:, -1]

    plt.figure(figsize=(9, 4.5), dpi=150)
    plt.hist(terminal_ncap, bins=42, alpha=0.85, label="Terminal NCAP")
    plt.axvline(q05, linestyle="--", label=f"Q05 = {q05:.2f}%")
    plt.axvline(median, linestyle="--", label=f"Median = {median:.2f}%")
    plt.axvline(q95, linestyle="--", label=f"Q95 = {q95:.2f}%")
    plt.xlabel("NCAP на 12-й месяц, %")
    plt.ylabel("Количество симуляций")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_PATH, bbox_inches="tight")
    plt.show()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("СТРЕСС-ТЕСТ НОРМАТИВНОГО КАПИТАЛА")
    print("=" * 70)

    history = load_history(HISTORY_PATH)
    shocks = load_shocks(SHOCKS_PATH)
    print(f"История: {len(history)} наблюдений | стресс-сценарий: {len(shocks)} месяцев")

    full_scenario = run_varx_forecast(history, shocks)
    residuals = calculate_ardl_residuals(history, ARDL_PARAMS)
    deterministic_path, simulation_paths = simulate_stress_test(
        history,
        full_scenario,
        ARDL_PARAMS,
        residuals,
        n_sims=N_SIMS,
        seed=RANDOM_SEED,
    )
    summary, q05, median, q95 = calculate_summary(
        deterministic_path, simulation_paths
    )

    print("\nРЕЗУЛЬТАТЫ")
    print(f"Минимальный детерминированный NCAP : {np.min(deterministic_path):.2f}%")
    print(f"NCAP на 12-й месяц                 : {deterministic_path[-1]:.2f}%")
    print(f"Q05 Monte Carlo                    : {q05:.2f}%")
    print(f"Медиана Monte Carlo                : {median:.2f}%")
    print(f"Q95 Monte Carlo                    : {q95:.2f}%")
    print(f"Размах Q95 - Q05                   : {q95 - q05:.2f} п.п.")

    save_results(full_scenario, deterministic_path, simulation_paths, summary)
    plot_distribution(simulation_paths, q05, median, q95)
    print(f"\nРезультаты: {OUTPUT_PATH}")
    print(f"График:     {CHART_PATH}")


if __name__ == "__main__":
    main()
