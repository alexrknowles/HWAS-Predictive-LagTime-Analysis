import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta

# ==============================================================================
# 1. CONFIGURATION AND CONSTANTS
# ==============================================================================

# --- File and Path Configuration (UPDATE THESE PATHS) ---
DATA_DIR = './data/'
RAMPART_FILE = 'Rampart2024.csv'
LOWER_STATIONS = {
    'Cadet Area': 'CadetArea2024.csv',
    'Community Center': 'CommCenter2024.csv',
    'Stadium': 'Stadium2024.csv'
}

# --- Meteorological Criteria ---
WIND_THRESHOLD_KT = 35  # Required for Lower Station hit AND Rampart trigger
ROT_THRESHOLD_F = 11  # Required for Delta T trigger
WEST_MIN_DEG = 230.0
WEST_MAX_DEG = 310.0
SEARCH_WINDOW = timedelta(hours=6)
EXCLUDED_MONTHS = [5, 6, 7, 8]

# --- Data Column Mappings ---
COL_TIME = 'Time'
COL_SPEED = 'Wind_speed'
COL_DIR = 'Wind_direction'
COL_TEMP = 'Temp'

# Time Format for Parsing
DATETIME_FORMAT = '%m/%d/%Y %H:%M'


# ==============================================================================
# 2. DATA INGESTION AND PREPROCESSING (Unchanged)
# ==============================================================================

def load_and_preprocess(filepath):
    """
    Loads a data file, cleans columns, handles duplicate timestamps, and sets the time index.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()

    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file type for {filepath}. Use CSV or Excel.")

    df = df.rename(columns=lambda x: x.strip())

    required_cols = [COL_TIME, COL_SPEED, COL_DIR, COL_TEMP]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {os.path.basename(filepath)}: {missing_cols}. Found columns: {list(df.columns)}")

    try:
        df[COL_TIME] = pd.to_datetime(df[COL_TIME], format=DATETIME_FORMAT, errors='coerce')
    except ValueError as e:
        print(f"Error parsing date-time column in {filepath}. Check your DATETIME_FORMAT.")
        raise e

    df = df.dropna(subset=[COL_TIME])

    original_rows = len(df)
    df = df.drop_duplicates(subset=[COL_TIME], keep='first')
    rows_dropped = original_rows - len(df)
    if rows_dropped > 0:
        print(f"Warning: Dropped {rows_dropped} duplicate time entries in {os.path.basename(filepath)}.")

    df = df.set_index(COL_TIME).sort_index()

    for col in [COL_SPEED, COL_DIR, COL_TEMP]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[COL_SPEED, COL_DIR, COL_TEMP])

    print(f"Loaded {os.path.basename(filepath)}: {len(df)} observations from {df.index.min()} to {df.index.max()}")

    return df


# ==============================================================================
# 3. MAIN ANALYSIS LOGIC (DELTA T TRIGGERED + ALL FILTERS)
# ==============================================================================

def run_analysis_delta_t_trigger():
    """
    Performs the wind translation analysis, requiring ALL three conditions at the trigger time:
    1. Delta T >= 11F
    2. Rampart wind direction is West
    3. Rampart wind speed >= 35 KT
    """
    print(f"\n{'=' * 70}\n🔬 Running Analysis: Lag Time from Triple-Filtered Trigger\n{'=' * 70}")

    # --- Load Data ---
    rampart_df = load_and_preprocess(os.path.join(DATA_DIR, RAMPART_FILE))
    if rampart_df.empty: return pd.DataFrame()

    lower_dfs = {}
    for name, file in LOWER_STATIONS.items():
        df = load_and_preprocess(os.path.join(DATA_DIR, file))
        if not df.empty:
            lower_dfs[name] = df

    if not lower_dfs:
        print("No lower station data successfully loaded. Exiting.")
        return pd.DataFrame()

    results_list = []

    for station_name, lower_df in lower_dfs.items():

        # --- 1. Align Data and Calculate Delta T ---
        combined_index = rampart_df.index.intersection(lower_df.index)

        if combined_index.empty: continue

        temp_df = pd.DataFrame({
            'Rampart_T': rampart_df[COL_TEMP].reindex(combined_index),
            'Lower_T': lower_df[COL_TEMP].reindex(combined_index),
            'Lower_S': lower_df[COL_SPEED].reindex(combined_index),
            'Rampart_S': rampart_df[COL_SPEED].reindex(combined_index),
            'Rampart_D': rampart_df[COL_DIR].reindex(combined_index)
        }).dropna()

        if temp_df.empty: continue

        temp_df['Delta_T'] = temp_df['Lower_T'] - temp_df['Rampart_T']

        # --- 2. Identify Triple-Filtered Trigger Events ---

        # Condition 1: Instability (Delta T >= 11F)
        delta_t_hit = (temp_df['Delta_T'] >= ROT_THRESHOLD_F)

        # Condition 2: Mountain Wave Setup (West Wind Direction)
        west_wind_at_trigger = temp_df['Rampart_D'].between(WEST_MIN_DEG, WEST_MAX_DEG)

        # Condition 3: Penetration Force (Rampart Speed >= 35 KT)
        # This is the NEW filter to ensure the force is present when the door opens.
        strong_wind_at_trigger = (temp_df['Rampart_S'] >= WIND_THRESHOLD_KT)

        # Combine all three criteria for a valid trigger start
        valid_trigger = delta_t_hit & west_wind_at_trigger & strong_wind_at_trigger

        # Identify the start time of distinct valid events
        is_start = valid_trigger & (~valid_trigger.shift(1, fill_value=False))
        trigger_start_times = valid_trigger[is_start].index

        # --- FILTER OUT EXCLUDED MONTHS ---
        initial_event_count = len(trigger_start_times)
        trigger_start_times = [
            t for t in trigger_start_times if t.month not in EXCLUDED_MONTHS
        ]
        filtered_count = len(trigger_start_times)
        print(
            f"[{station_name}] Remaining events after all filters: {filtered_count} (started with {initial_event_count}).")
        # --- END FILTERS ---

        if not trigger_start_times:
            print(f"No valid Triple-Filtered events found for {station_name}.")
            continue

        # --- 3. Analysis Loop for Trigger Times ---
        for trigger_time in trigger_start_times:

            trigger_obs = temp_df.loc[trigger_time]

            translation_occurred = False
            lag_time_minutes = np.nan

            # Define the search window
            search_end_time = trigger_time + SEARCH_WINDOW

            # Search for the wind hit in the lower station within the window
            search_window_df = lower_df.loc[trigger_time:search_end_time]

            # Find the first time the lower station hits the wind threshold
            lower_hit_series = search_window_df[search_window_df[COL_SPEED] >= WIND_THRESHOLD_KT]

            if not lower_hit_series.empty:
                # Translation occurred!
                translation_time = lower_hit_series.index[0]
                lag_time = translation_time - trigger_time
                lag_time_minutes = lag_time.total_seconds() / 60.0
                translation_occurred = True

            # Record the event details
            results_list.append({
                'Station': station_name,
                'Trigger_Time': trigger_time,
                'Delta_T_Trigger': trigger_obs['Delta_T'],
                'Lower_T_Trigger': trigger_obs['Lower_T'],
                'Rampart_Wind_at_Trigger': trigger_obs['Rampart_S'],
                'Translation_Occurred': translation_occurred,
                'Lag_Time_Minutes': lag_time_minutes
            })

    return pd.DataFrame(results_list)


# ==============================================================================
# 4. STATISTICAL SUMMARY AND VISUALIZATION (Corrected Print Statements)
# ==============================================================================

def generate_output_delta_t_trigger(results_df):
    """
    Calculates final statistics and generates scatter plots based on the Delta T trigger.
    """
    if results_df.empty:
        print("No results to display.")
        return

    # --- DataFrames for Filtering ---
    total_events_df = results_df.copy()
    successful_df = total_events_df[total_events_df['Translation_Occurred'] == True]

    # ==========================================================================
    # 1. OVERALL SUMMARY
    # ==========================================================================

    print(f"\n{'*' * 70}\n📊 SUMMARY: Overall Analysis (Lag Time from Triple-Filtered Trigger)\n{'*' * 70}")

    total_events = len(total_events_df)
    successful_translations = total_events_df['Translation_Occurred'].sum()
    avg_lag = successful_df['Lag_Time_Minutes'].mean()

    print(
        f"Total Triple-Filtered events analyzed (Delta T >= {ROT_THRESHOLD_F}°F, West Wind, Rampart Wind >= {WIND_THRESHOLD_KT} KT): {total_events}")
    print(f"Total successful wind translations: {successful_translations}")
    print(f"**Overall Wind Translation Success Rate: {(successful_translations / total_events) * 100:.2f}%**")
    print(f"Average Lag Time for ALL successful translations: {avg_lag:.2f} minutes")

    # ==========================================================================
    # 2. SUMMARY by Station
    # ==========================================================================

    print("\n--- Summary by Lower Elevation Station ---")

    station_summary = total_events_df.groupby('Station').agg(
        Total_Triggers=('Trigger_Time', 'size'),
        Successful_Translations=('Translation_Occurred', 'sum'),
        Avg_Rampart_Wind_at_Trigger=('Rampart_Wind_at_Trigger', 'mean')
    )

    successful_summary = successful_df.groupby('Station').agg(
        Avg_Lag_Mins=('Lag_Time_Minutes', 'mean'),
        Avg_Lower_T_Trigger=('Lower_T_Trigger', 'mean')
    )

    station_summary = station_summary.join(successful_summary)
    station_summary['Translation_Percent'] = (station_summary['Successful_Translations'] / station_summary[
        'Total_Triggers']) * 100

    # Clean up column names and formatting
    station_summary = station_summary.rename(columns={'Avg_Lower_T_Trigger': 'Avg_Lower_T'})
    station_summary['Avg_Lag_Mins'] = station_summary['Avg_Lag_Mins'].round(2)
    station_summary['Avg_Rampart_Wind_at_Trigger'] = station_summary['Avg_Rampart_Wind_at_Trigger'].round(2)
    station_summary['Avg_Lower_T'] = station_summary['Avg_Lower_T'].round(2)
    station_summary['Translation_Percent'] = station_summary['Translation_Percent'].round(2)

    print(station_summary.to_markdown(numalign="left", stralign="left"))

    # ==========================================================================
    # 3. VISUALIZATIONS (Lag Time vs. Conditions at Trigger)
    # ==========================================================================

    # --- Scatter Plot 1 (Lag Time vs. Rampart Wind Speed at Delta T Trigger) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    unsuccessful_df = total_events_df[total_events_df['Translation_Occurred'] == False]

    # Plot Unsuccessful Events (Lag = 0)
    ax1.scatter(unsuccessful_df['Rampart_Wind_at_Trigger'], np.zeros_like(unsuccessful_df['Rampart_Wind_at_Trigger']),
                marker='x', color='gray', alpha=0.4, label='Failed Translation')

    # Plot Successful Events
    for station in successful_df['Station'].unique():
        station_df = successful_df[successful_df['Station'] == station]
        ax1.scatter(station_df['Rampart_Wind_at_Trigger'], station_df['Lag_Time_Minutes'],
                    label=f'{station} (Success)', alpha=0.8, s=100)

    # FIXED PLOT TITLE PRINTING
    ax1.set_title(
        f'Plot 1: Translation Lag Time vs. Rampart Wind Speed (at $\Delta T \geq {ROT_THRESHOLD_F}^\circ$F Trigger)',
        fontsize=14)
    ax1.set_xlabel('Rampart Wind Speed at Trigger Time (KT)', fontsize=12)
    ax1.set_ylabel('Lag Time for Wind Translation (Minutes)', fontsize=12)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=1.0)
    ax1.legend(title="Event Outcome", loc='upper right')
    plt.tight_layout()
    plt.show()

    # --- Scatter Plot 2 (Lag Time vs. Absolute Lower Temperature at Trigger) ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    # Plot Successful Events
    for station in successful_df['Station'].unique():
        station_df = successful_df[successful_df['Station'] == station]
        ax2.scatter(station_df['Lower_T_Trigger'], station_df['Lag_Time_Minutes'],
                    label=f'{station} (Success)', alpha=0.8, s=100)

    # Plot Unsuccessful Events
    ax2.scatter(unsuccessful_df['Lower_T_Trigger'], np.zeros_like(unsuccessful_df['Lower_T_Trigger']),
                marker='x', color='gray', alpha=0.4, label='Failed Translation')

    # FIXED PLOT TITLE PRINTING
    ax2.set_title(
        f'Plot 2: Translation Lag Time vs. Absolute Lower Elevation Temperature (at $\Delta T \geq {ROT_THRESHOLD_F}^\circ$F Trigger)',
        fontsize=14)
    ax2.set_xlabel('Lower Elevation Temperature at Trigger Time (°F)', fontsize=12)
    ax2.set_ylabel('Lag Time for Wind Translation (Minutes)', fontsize=12)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=1.0)
    ax2.legend(title="Event Outcome", loc='upper right')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 5. EXECUTION
# ==============================================================================

if __name__ == "__main__":
    results_delta_t_triggered = run_analysis_delta_t_trigger()
    generate_output_delta_t_trigger(results_delta_t_triggered)
