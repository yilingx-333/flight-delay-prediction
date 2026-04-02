from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FINAL_DATASET_PATH = RESULTS_DIR / "final_dataset.csv"
EDA_DIR = RESULTS_DIR / "eda"


# Utility functions
def ensure_output_dir(path: Path) -> None:
    """Create output directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_plot(path: Path) -> None:
    """Save the current matplotlib figure and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def load_data(path: Path) -> pd.DataFrame:
    """Load the cleaned dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def basic_summary(df: pd.DataFrame) -> str:
    """Return a text summary of the dataset."""
    lines = []
    lines.append("Flight Delay EDA Summary")
    lines.append("=" * 30)
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("")

    lines.append("Columns:")
    lines.append(", ".join(df.columns.astype(str).tolist()))
    lines.append("")

    lines.append("Missing values:")
    na_counts = df.isna().sum().sort_values(ascending=False)
    for col, val in na_counts.items():
        lines.append(f"  {col}: {val}")
    lines.append("")

    lines.append("Numeric summary:")
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        lines.append(num_df.describe().to_string())
    else:
        lines.append("  No numeric columns found.")

    return "\n".join(lines)


def print_and_save_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Print summary to console and save it to a text file."""
    summary_text = basic_summary(df)
    print(summary_text)
    (output_dir / "eda_summary.txt").write_text(summary_text, encoding="utf-8")


# Plot functions
def plot_arrival_delay_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot the distribution of arrival delay."""
    if "arr_delay" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(df["arr_delay"].dropna(), bins=50)
    plt.title("Distribution of Arrival Delay")
    plt.xlabel("Arrival Delay (minutes)")
    plt.ylabel("Count")
    save_plot(output_dir / "01_arrival_delay_distribution.png")


def plot_dep_vs_arr(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot departure delay vs arrival delay."""
    if not {"dep_delay", "arr_delay"}.issubset(df.columns):
        return

    sample = df[["dep_delay", "arr_delay"]].dropna()
    if len(sample) > 8000:
        sample = sample.sample(8000, random_state=42)

    plt.figure(figsize=(7, 5))
    plt.scatter(sample["dep_delay"], sample["arr_delay"], alpha=0.25)
    plt.title("Departure Delay vs Arrival Delay")
    plt.xlabel("Departure Delay (minutes)")
    plt.ylabel("Arrival Delay (minutes)")
    save_plot(output_dir / "02_dep_delay_vs_arr_delay.png")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot correlation matrix for numeric features."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return

    corr = numeric_df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, interpolation="nearest")
    plt.title("Correlation Matrix")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    save_plot(output_dir / "03_correlation_matrix.png")


def plot_delay_by_hour(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot average arrival delay by departure hour."""
    if not {"dep_hour", "arr_delay"}.issubset(df.columns):
        return

    tmp = (
        df.groupby("dep_hour")["arr_delay"]
        .mean()
        .reset_index()
        .sort_values("dep_hour")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(tmp["dep_hour"], tmp["arr_delay"], marker="o")
    plt.title("Average Arrival Delay by Departure Hour")
    plt.xlabel("Departure Hour")
    plt.ylabel("Average Arrival Delay (minutes)")
    save_plot(output_dir / "04_delay_by_dep_hour.png")


def plot_delay_by_month(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot average arrival delay by month."""
    if not {"month", "arr_delay"}.issubset(df.columns):
        return

    tmp = (
        df.groupby("month")["arr_delay"]
        .mean()
        .reset_index()
        .sort_values("month")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(tmp["month"], tmp["arr_delay"], marker="o")
    plt.title("Average Arrival Delay by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Arrival Delay (minutes)")
    plt.xticks(tmp["month"].tolist())
    save_plot(output_dir / "05_delay_by_month.png")


def plot_weekday_vs_weekend(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot average arrival delay on weekday vs weekend."""
    if not {"is_weekend", "arr_delay"}.issubset(df.columns):
        return

    tmp = df.groupby("is_weekend")["arr_delay"].mean().reindex([0, 1])

    plt.figure(figsize=(6, 5))
    plt.bar(["Weekday", "Weekend"], tmp.values)
    plt.title("Average Arrival Delay: Weekday vs Weekend")
    plt.xlabel("Day Type")
    plt.ylabel("Average Arrival Delay (minutes)")
    save_plot(output_dir / "06_weekday_vs_weekend.png")


def plot_delay_by_carrier(df: pd.DataFrame, output_dir: Path, top_n: int = 10) -> None:
    """Plot average arrival delay by carrier."""
    if not {"carrier", "arr_delay"}.issubset(df.columns):
        return

    tmp = (
        df.groupby("carrier")["arr_delay"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(9, 5))
    plt.bar(tmp.index.astype(str), tmp.values)
    plt.title(f"Top {top_n} Carriers by Average Arrival Delay")
    plt.xlabel("Carrier")
    plt.ylabel("Average Arrival Delay (minutes)")
    plt.xticks(rotation=45, ha="right")
    save_plot(output_dir / "07_top_carriers_by_delay.png")


def plot_weather_effects(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot precipitation and temperature versus arrival delay."""
    if "PRCP" in df.columns:
        sample = df[["PRCP", "arr_delay"]].dropna()
        if len(sample) > 8000:
            sample = sample.sample(8000, random_state=42)

        plt.figure(figsize=(7, 5))
        plt.scatter(sample["PRCP"], sample["arr_delay"], alpha=0.25)
        plt.title("Precipitation vs Arrival Delay")
        plt.xlabel("Precipitation")
        plt.ylabel("Arrival Delay (minutes)")
        save_plot(output_dir / "08_prcp_vs_delay.png")

    if "TMAX" in df.columns:
        sample = df[["TMAX", "arr_delay"]].dropna()
        if len(sample) > 8000:
            sample = sample.sample(8000, random_state=42)

        plt.figure(figsize=(7, 5))
        plt.scatter(sample["TMAX"], sample["arr_delay"], alpha=0.25)
        plt.title("Maximum Temperature vs Arrival Delay")
        plt.xlabel("TMAX")
        plt.ylabel("Arrival Delay (minutes)")
        save_plot(output_dir / "09_tmax_vs_delay.png")


def plot_top_routes(df: pd.DataFrame, output_dir: Path, top_n: int = 10) -> None:
    """Plot average arrival delay for the most common routes."""
    if not {"route", "arr_delay"}.issubset(df.columns):
        return

    route_counts = df["route"].value_counts().head(top_n).index
    subset = df[df["route"].isin(route_counts)]

    tmp = (
        subset.groupby("route")["arr_delay"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(tmp.index.astype(str), tmp.values)
    plt.title(f"Top {top_n} Most Frequent Routes by Average Arrival Delay")
    plt.xlabel("Route")
    plt.ylabel("Average Arrival Delay (minutes)")
    plt.xticks(rotation=45, ha="right")
    save_plot(output_dir / "10_top_routes_by_delay.png")


# Extra text analysis helpers
def write_key_findings(df: pd.DataFrame, output_dir: Path) -> None:
    """Write a short insight summary based on the dataset."""
    lines = []
    lines.append("Key Findings")
    lines.append("=" * 20)
    lines.append("")

    if "arr_delay" in df.columns:
        lines.append(f"Arrival delay mean: {df['arr_delay'].mean():.3f}")
        lines.append(f"Arrival delay median: {df['arr_delay'].median():.3f}")
        lines.append(f"Arrival delay std: {df['arr_delay'].std():.3f}")
        lines.append("")

    if {"dep_delay", "arr_delay"}.issubset(df.columns):
        corr = df[["dep_delay", "arr_delay"]].corr().iloc[0, 1]
        lines.append(f"Correlation(dep_delay, arr_delay): {corr:.3f}")

    if {"distance", "great_circle_km"}.issubset(df.columns):
        corr = df[["distance", "great_circle_km"]].corr().iloc[0, 1]
        lines.append(f"Correlation(distance, great_circle_km): {corr:.3f}")

    if {"PRCP", "arr_delay"}.issubset(df.columns):
        corr = df[["PRCP", "arr_delay"]].corr().iloc[0, 1]
        lines.append(f"Correlation(PRCP, arr_delay): {corr:.3f}")

    if {"TMAX", "arr_delay"}.issubset(df.columns):
        corr = df[["TMAX", "arr_delay"]].corr().iloc[0, 1]
        lines.append(f"Correlation(TMAX, arr_delay): {corr:.3f}")

    lines.append("")
    lines.append("Top carriers by average arrival delay:")
    if {"carrier", "arr_delay"}.issubset(df.columns):
        top_carriers = (
            df.groupby("carrier")["arr_delay"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )
        for carrier, value in top_carriers.items():
            lines.append(f"  {carrier}: {value:.3f}")

    (output_dir / "key_findings.txt").write_text("\n".join(lines), encoding="utf-8")


# Main
def main() -> None:
    ensure_output_dir(EDA_DIR)

    df = load_data(FINAL_DATASET_PATH)

    print(f"Shape: {df.shape}")
    print(df.info())
    print(df.select_dtypes(include="number").describe())

    print_and_save_summary(df, EDA_DIR)
    write_key_findings(df, EDA_DIR)

    plot_arrival_delay_distribution(df, EDA_DIR)
    plot_dep_vs_arr(df, EDA_DIR)
    plot_correlation_matrix(df, EDA_DIR)
    plot_delay_by_hour(df, EDA_DIR)
    plot_delay_by_month(df, EDA_DIR)
    plot_weekday_vs_weekend(df, EDA_DIR)
    plot_delay_by_carrier(df, EDA_DIR, top_n=10)
    plot_weather_effects(df, EDA_DIR)
    plot_top_routes(df, EDA_DIR, top_n=10)

    print(f"EDA files saved in: {EDA_DIR}")


if __name__ == "__main__":
    main()