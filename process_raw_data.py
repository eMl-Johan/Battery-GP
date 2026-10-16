from Data_PreProcessing import dfSOC_pre_cumu
import torch
import numpy as np
from pathlib import Path
import pandas as pd


def process_data(df, training_cycles=10, test_cycle=11,save_dir="C:/Users/JohanEkberg/OneDrive - eMaintenanceLAB/Aviation/BatMan/Data"):
    df['SOC_t+1'] = df['SOC'].shift(-1)
    df['voltage_load_t+1'] = df['voltage_load'].shift(-1)
    df['temperature_battery_t+1'] = df['temperature_battery'].shift(-1) 
    df = df[:-1]
    df.loc[df['cycle'] != df['cycle'].shift(-1), ['SOC_t+1', 'voltage_load_t+1', 'temperature_battery_t+1']] = None

    df = df.dropna()

    df = df[df['voltage_load'] != df.groupby('cycle')['voltage_load'].transform('min')]
    df = df[df['voltage_load_t+1'] != df.groupby('cycle')['voltage_load_t+1'].transform('min')]
    df = df[~df['cycle'].isin([12, 37])]

    training_cycles = np.sort(df['cycle'].unique())[0:training_cycles]

    train_data = df[df['cycle'].isin(training_cycles)]
    test_data = df[df['cycle'] == test_cycle] 

    test_data['voltage_load'] = test_data['voltage_load']
    test_data['current_load'] = test_data['current_load']
    test_data['temperature_battery'] = test_data['temperature_battery']
    test_data['cumulative_current_total'] = test_data['cumulative_current_total']
    test_data['cumulative_current'] = test_data['cumulative_current']
    test_data['temperature_battery_t+1'] = test_data['temperature_battery_t+1']
    test_data['voltage_load_t+1'] = test_data['voltage_load_t+1']

    train_data['voltage_load'] = train_data['voltage_load']
    train_data['current_load'] = train_data['current_load']
    train_data['temperature_battery'] = train_data['temperature_battery']
    train_data['cumulative_current_total'] = train_data['cumulative_current_total']
    train_data['cumulative_current'] = train_data['cumulative_current']
    train_data['temperature_battery_t+1'] = train_data['temperature_battery_t+1']
    train_data['voltage_load_t+1'] = train_data['voltage_load_t+1']

    Y_train = train_data[['SOC_t+1','voltage_load_t+1','temperature_battery_t+1']].values \
            - train_data[['SOC','voltage_load','temperature_battery']].values
    Y_test = test_data[['SOC_t+1','voltage_load_t+1','temperature_battery_t+1']].values \
            - test_data[['SOC','voltage_load','temperature_battery']].values
    X_train = train_data[['SOC', 'voltage_load', 'temperature_battery', 'current_load']].values
    X_test = test_data[['SOC', 'voltage_load', 'temperature_battery', 'current_load']].values
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        save_dir / "X_train.csv",
        X_train,
        delimiter=",",
        header="SOC,voltage_load,temperature_battery,current_load",
        comments=""
    )

    np.savetxt(
        save_dir / "X_test.csv",
        X_test,
        delimiter=",",
        header="SOC,voltage_load,temperature_battery,current_load",
        comments=""
    )

    np.savetxt(
        save_dir / "Y_train.csv",
        Y_train,
        delimiter=",",
        header="delta_SOC,delta_voltage_load,delta_temperature_battery",
        comments=""
    )

    np.savetxt(
        save_dir / "Y_test.csv",
        Y_test,
        delimiter=",",
        header="delta_SOC,delta_voltage_load,delta_temperature_battery",
        comments=""
    )

    print(f"Saved CSV files to: {save_dir}")
    return save_dir

if __name__ == "__main__":
    df = dfSOC_pre_cumu(
        "C:/Users/JohanEkberg/OneDrive - eMaintenanceLAB/Aviation/Battery/Datasets/battery_pack_data/battery_alt_dataset/regular_alt_batteries/battery51.csv"
    )

    process_data(
        df,
        training_cycles=10,
        test_cycle=11,
        save_dir="C:/Users/JohanEkberg/OneDrive - eMaintenanceLAB/Aviation/BatMan/Data"
    )