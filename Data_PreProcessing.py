import pandas as pd
import numpy as np

def dfSOC_pre(file_path):
    data = pd.read_csv(file_path)
    current_cycle = 0
    cycle_values = []

    
    for i, row in data.iterrows():
        if row['mode'] == -1:  # Discharge mode
            if i == 0 or data.loc[i - 1, 'mode'] != -1:
                current_cycle += 1
        cycle_values.append(current_cycle)

    data['cycle'] = cycle_values

    discharge_data = data[data['mode'] == -1]

    mean_currents = discharge_data.groupby('cycle')['current_load'].mean()
    valid_cycles = mean_currents[mean_currents >= 5].index
    discharge_data = discharge_data[discharge_data['cycle'].isin(valid_cycles)]
    discharge_data['delta_t'] = discharge_data['time'].diff().fillna(0) / 3600
    discharge_data['SOC'] = 1.0
    discharge_data['cumulative_current'] = 0.0
    cumulative_current_total = 0.0

    for cycle, cycle_data in discharge_data.groupby('cycle'):
        discharge_data.loc[cycle_data.index[0], 'SOC'] = 1.0  
        discharge_data.loc[cycle_data.index[0], 'cumulative_current'] = 0.0
        
        capacity = np.trapezoid(cycle_data['current_load'], cycle_data['time']) / 3600  
        cumulative_current = 0.0  
        for i in range(1, len(cycle_data)):
            idx = cycle_data.index[i]
            prev_idx = cycle_data.index[i - 1]

            current = discharge_data.loc[idx, 'current_load']
            delta_t = discharge_data.loc[idx, 'delta_t']
            cumulative_current += abs(current) * delta_t
            discharge_data.loc[idx, 'cumulative_current'] = cumulative_current
            previous_soc = discharge_data.loc[prev_idx, 'SOC']
            new_soc = previous_soc - (current * delta_t / capacity)

            if new_soc < 0:
                new_soc = 0
            discharge_data.loc[idx, 'SOC'] = new_soc


    additional_rows = []
    last_rows = []
    for cycle, cycle_data in discharge_data.groupby('cycle'):
        first_row = cycle_data.iloc[0]
        last_row = cycle_data.iloc[-1]
        for i in range(5):
            additional_row = first_row.copy()
            additional_row['time'] -= (5 - i) * 1 
            additional_row['current_load'] = 0.0
            additional_row['voltage_load'] = discharge_data['voltage_load'].max()
            additional_row['temperature_battery'] = first_row['temperature_battery']
            additional_row['SOC'] = 1.0
            additional_row['cumulative_current'] = 0.0
            additional_row['cycle'] = first_row['cycle']
            additional_rows.append(additional_row)

            additional_rows_end = last_row.copy()
            additional_rows_end['current_load'] = 0.0
            additional_rows_end['voltage_load'] = last_row['voltage_load']
            additional_rows_end['temperature_battery'] = last_row['temperature_battery']
            additional_rows_end['SOC'] = last_row['SOC']
            additional_rows_end['cumulative_current'] = last_row['cumulative_current']
            additional_rows_end['cycle'] = last_row['cycle']
            additional_rows_end['time'] += i * 1
            last_rows.append(additional_rows_end)

    additional_rows_df = pd.DataFrame(additional_rows)
    last_rows_df = pd.DataFrame(last_rows)
    discharge_data = pd.concat([additional_rows_df, discharge_data, last_rows_df], ignore_index=True)
    discharge_data = discharge_data.sort_values(by=['cycle', 'time']).reset_index(drop=True)

    return discharge_data



def dfSOC_pre_cumu(file_path):
    data = pd.read_csv(file_path)
    current_cycle = 0
    cycle_values = []

    for i, row in data.iterrows():
        if row['mode'] == -1:  # Discharge mode
            if i == 0 or data.loc[i - 1, 'mode'] != -1:
                current_cycle += 1
        cycle_values.append(current_cycle)

    data['cycle'] = cycle_values

    discharge_data = data[data['mode'] == -1]

    
    mean_currents = discharge_data.groupby('cycle')['current_load'].mean()
    valid_cycles = mean_currents[mean_currents >= 5].index
    discharge_data = discharge_data[discharge_data['cycle'].isin(valid_cycles)]

    
    discharge_data['delta_t'] = discharge_data['time'].diff().fillna(0) / 3600
    discharge_data.loc[discharge_data['cycle'] != discharge_data['cycle'].shift(1), 'delta_t'] = 0

    
    discharge_data['SOC'] = 1.0
    discharge_data['cumulative_current'] = 0.0
    discharge_data['cumulative_current_total'] = 0.0

    cumulative_current = 0.0  

    for cycle, cycle_data in discharge_data.groupby('cycle'):
        first_idx = cycle_data.index[0]
        discharge_data.loc[first_idx, 'SOC'] = 1.0

        capacity = np.trapezoid(cycle_data['current_load'], cycle_data['time']) / 3600
        discharge_data.loc[cycle_data.index, 'effective_capacity'] = capacity
        for i in range(len(cycle_data)):
            idx = cycle_data.index[i]

            
            current = discharge_data.loc[idx, 'current_load']
            delta_t = discharge_data.loc[idx, 'delta_t']
            cumulative_current += abs(current) * delta_t
            discharge_data.loc[idx, 'cumulative_current'] = cumulative_current
            discharge_data.loc[idx, 'cumulative_current_total'] = cumulative_current

            if i > 0:
                prev_idx = cycle_data.index[i - 1]
                previous_soc = discharge_data.loc[prev_idx, 'SOC']
                new_soc = previous_soc - (current * delta_t / capacity)
                discharge_data.loc[idx, 'SOC'] = max(new_soc, 0)

    additional_rows = []
    last_rows = []

    for cycle, cycle_data in discharge_data.groupby('cycle'):
        first_row = cycle_data.iloc[0]
        last_row = cycle_data.iloc[-1]

        for i in range(10):
            additional_row = first_row.copy()
            additional_row['time'] -= (10 - i) * 1
            additional_row['current_load'] = 0.0
            additional_row['voltage_load'] = discharge_data['voltage_load'].max()
            additional_row['temperature_battery'] = first_row['temperature_battery']
            additional_row['SOC'] = 1.0
            additional_row['cumulative_current'] = 0.0
            additional_row['cumulative_current_total'] = first_row['cumulative_current_total']
            additional_row['cycle'] = first_row['cycle']
            additional_rows.append(additional_row)
            """for i in range(10):
            additional_row = last_row.copy()
            additional_row['time'] += i * 1
            additional_row['current_load'] = 0.0
            additional_row['voltage_load'] = last_row['voltage_load']
            additional_row['temperature_battery'] = last_row['temperature_battery']
            additional_row['SOC'] = last_row['SOC']
            additional_row['cumulative_current'] = last_row['cumulative_current']
            additional_row['cumulative_current_total'] = last_row['cumulative_current_total']
            additional_row['cycle'] = last_row['cycle']
            last_rows.append(additional_row) """

    additional_rows_df = pd.DataFrame(additional_rows)
    last_rows_df = pd.DataFrame(last_rows)

    discharge_data = pd.concat([additional_rows_df, discharge_data, last_rows_df], ignore_index=True)
    discharge_data = discharge_data.sort_values(by=['cycle', 'time']).reset_index(drop=True)

    return discharge_data



def dfSOC_pre_rest(file_path):
    data = pd.read_csv(file_path)
    current_cycle = 0
    cycle_values = []

    for i, row in data.iterrows():
        if row['mode'] == -1:  # Discharge mode
            if i == 0 or data.loc[i - 1, 'mode'] != -1:
                current_cycle += 1
        cycle_values.append(current_cycle)

    data['cycle'] = cycle_values

    discharge_data = data[data['mode'] == -1]

    
    mean_currents = discharge_data.groupby('cycle')['current_load'].mean()
    valid_cycles = mean_currents[mean_currents >= 5].index
    discharge_data = discharge_data[discharge_data['cycle'].isin(valid_cycles)]

    
    discharge_data['delta_t'] = discharge_data['time'].diff().fillna(0) / 3600
    discharge_data.loc[discharge_data['cycle'] != discharge_data['cycle'].shift(1), 'delta_t'] = 0

    
    discharge_data['SOC'] = 1.0
    discharge_data['SOC_Physics'] = 1.0
    discharge_data['cumulative_current'] = 0.0
    discharge_data['cumulative_current_total'] = 0.0
    discharge_data['cumulative_energy'] = 0.0
    
    cumulative_current = 0.0  
    cumulative_current_cycle = 0.0
    cumulative_energy_total = 0.0
    for cycle, cycle_data in discharge_data.groupby('cycle'):
        first_idx = cycle_data.index[0]
        discharge_data.loc[first_idx, 'SOC'] = 1.0

        capacity = np.trapezoid(cycle_data['current_load'], cycle_data['time']) / 3600
        discharge_data.loc[cycle_data.index, 'effective_capacity'] = capacity
        cumulative_current_cycle = 0
        for i in range(len(cycle_data)):
            idx = cycle_data.index[i]

            
            current = discharge_data.loc[idx, 'current_load']
            delta_t = discharge_data.loc[idx, 'delta_t']
            voltage = discharge_data.loc[idx, 'voltage_load']
            cumulative_current += abs(current) * delta_t
            cumulative_current_cycle += abs(current) * delta_t
            discharge_data.loc[idx, 'cumulative_current'] = cumulative_current_cycle
            discharge_data.loc[idx, 'cumulative_current_total'] = cumulative_current
            
            dE = abs(current) * voltage * delta_t
            cumulative_energy_total += dE
            discharge_data.loc[idx, 'cumulative_energy'] = cumulative_energy_total

            if i > 0:
                prev_idx = cycle_data.index[i - 1]
                previous_soc = discharge_data.loc[prev_idx, 'SOC']
                new_soc = previous_soc - (current * delta_t / capacity)
                discharge_data.loc[idx, 'SOC'] = max(new_soc, 0)

                prev_idx = cycle_data.index[i - 1]
                previous_soc_physics = discharge_data.loc[prev_idx, 'SOC_Physics']
                new_soc_physics = previous_soc_physics - (current * delta_t / 2.5)
                discharge_data.loc[idx, 'SOC_Physics'] = max(new_soc_physics, 0)

    
    additional_rows = []
    #last_rows = []

    for cycle, cycle_data in discharge_data.groupby('cycle'):
        first_row = cycle_data.iloc[0]
        #last_row = cycle_data.iloc[-1]

        mode_transition_idx = data[(data['cycle'] == cycle) & (data['mode'] == -1)].index[-1] + 1

        temp_values = data.loc[mode_transition_idx:mode_transition_idx + 29, 'temperature_battery'].values
        voltage_values = data.loc[mode_transition_idx:mode_transition_idx + 29, 'voltage_charger'].values
        
        cycle_max_vload = cycle_data['voltage_load'].max()
        # If there are fewer than 10 rows available, pad with the last available value
        #if len(temp_values) < 10:
            #temp_values = np.pad(temp_values, (0, 20 - len(temp_values)), constant_values=temp_values[-1])
        #if len(voltage_values) < 10:
            #voltage_values = np.pad(voltage_values, (0, 20 - len(voltage_values)), constant_values=voltage_values[-1])

        # Add the first 10 rows with adjusted values
        for i in range(30):
            additional_row = first_row.copy()
            additional_row['time'] -= (30 - i) * 1
            additional_row['current_load'] = 0.0
            #additional_row['voltage_load'] = discharge_data['voltage_load'].max()
            additional_row['voltage_load'] = cycle_max_vload
            additional_row['temperature_battery'] = first_row['temperature_battery']
            additional_row['SOC'] = 1.0
            additional_row['cumulative_current'] = 0.0
            additional_row['cumulative_current_total'] = first_row['cumulative_current_total']
            additional_row['cycle'] = first_row['cycle']
            additional_row['cumulative_energy'] = first_row['cumulative_energy'] 
            additional_rows.append(additional_row)

        # Add the last 10 rows with adjusted values
        """for i in range(30):
            additional_row = last_row.copy()
            additional_row['time'] += i * 1
            additional_row['current_load'] = 0.0
            additional_row['voltage_load'] = voltage_values[i]
            additional_row['temperature_battery'] = temp_values[i]
            additional_row['SOC'] = last_row['SOC']
            additional_row['cumulative_current'] = last_row['cumulative_current']
            additional_row['cumulative_current_total'] = last_row['cumulative_current_total']
            additional_row['cumulative_energy'] = last_row['cumulative_energy']
            additional_row['cycle'] = last_row['cycle']
            last_rows.append(additional_row) """

    additional_rows_df = pd.DataFrame(additional_rows)
    #last_rows_df = pd.DataFrame(last_rows)

    discharge_data = pd.concat([additional_rows_df, discharge_data], ignore_index=True)
    discharge_data = discharge_data.sort_values(by=['cycle', 'time']).reset_index(drop=True)


    return discharge_data



""" def dfSOC_pre_cumu(file_path):
    data = pd.read_csv(file_path)
    current_cycle = 0
    cycle_values = []

    # Assign cycle numbers
    for i, row in data.iterrows():
        if row['mode'] == -1:  # Discharge mode
            if i == 0 or data.loc[i - 1, 'mode'] != -1:
                current_cycle += 1
        cycle_values.append(current_cycle)

    data['cycle'] = cycle_values

    discharge_data = data[data['mode'] == -1].copy()

    # Keep only cycles with sufficient mean discharge current
    mean_currents = discharge_data.groupby('cycle')['current_load'].mean()
    valid_cycles = mean_currents[mean_currents >= 5].index
    discharge_data = discharge_data[discharge_data['cycle'].isin(valid_cycles)].copy()

    # Δt in hours; reset at cycle boundaries
    discharge_data['delta_t'] = discharge_data['time'].diff().fillna(0) / 3600.0
    discharge_data.loc[discharge_data['cycle'] != discharge_data['cycle'].shift(1), 'delta_t'] = 0

    # Init columns
    discharge_data['SOC'] = 1.0
    discharge_data['cumulative_current'] = 0.0
    discharge_data['cumulative_current_total'] = 0.0

    cumulative_current = 0.0

    for cycle, cycle_data in discharge_data.groupby('cycle'):
        first_idx = cycle_data.index[0]
        discharge_data.loc[first_idx, 'SOC'] = 1.0

        capacity = np.trapezoid(cycle_data['current_load'], cycle_data['time']) / 3600.0

        for i in range(len(cycle_data)):
            idx = cycle_data.index[i]
            current = discharge_data.loc[idx, 'current_load']
            delta_t = discharge_data.loc[idx, 'delta_t']

            cumulative_current += abs(current) * delta_t
            discharge_data.loc[idx, 'cumulative_current'] = cumulative_current
            discharge_data.loc[idx, 'cumulative_current_total'] = cumulative_current

            if i > 0:
                prev_idx = cycle_data.index[i - 1]
                previous_soc = discharge_data.loc[prev_idx, 'SOC']
                new_soc = previous_soc - (current * delta_t / capacity if capacity != 0 else 0.0)
                discharge_data.loc[idx, 'SOC'] = max(new_soc, 0)

    # ---- Add pre- and post-discharge rest rows ----
    additional_rows = []
    last_rows = []

    for cycle, cycle_data in discharge_data.groupby('cycle'):
        first_row = cycle_data.iloc[0]
        last_row = cycle_data.iloc[-1]

        # --- Pre-discharge rest (mode == 0) ---
        start_idx = data[(data['cycle'] == cycle) & (data['mode'] == -1)].index.min()
        pre_slice = data.loc[:start_idx - 1]
        pre_slice = pre_slice[pre_slice['mode'] == 0].tail(10)

        pre_volt = pre_slice['voltage_load'].to_numpy() if not pre_slice.empty else np.full(10, first_row['voltage_load'])
        pre_temp = pre_slice['temperature_battery'].to_numpy() if not pre_slice.empty else np.full(10, first_row['temperature_battery'])

        if len(pre_volt) < 10:
            pre_volt = np.pad(pre_volt, (10 - len(pre_volt), 0), mode='edge')
            pre_temp = np.pad(pre_temp, (10 - len(pre_temp), 0), mode='edge')

        for i in range(10):
            additional_row = first_row.copy()
            additional_row['time'] = first_row['time'] - (10 - i)
            additional_row['current_load'] = 0.0
            additional_row['voltage_load'] = pre_volt[i]
            additional_row['temperature_battery'] = pre_temp[i]
            additional_row['SOC'] = 1.0
            additional_row['cumulative_current'] = 0.0
            additional_row['cumulative_current_total'] = first_row['cumulative_current_total']
            additional_row['cycle'] = cycle
            additional_rows.append(additional_row)

        # --- Post-discharge rest (mode == 0) ---
        end_idx = data[(data['cycle'] == cycle) & (data['mode'] == -1)].index.max()
        post_slice = data.loc[end_idx + 1:]
        post_slice = post_slice[post_slice['mode'] == 0].head(10)

        post_volt = post_slice['voltage_load'].to_numpy() if not post_slice.empty else np.full(10, last_row['voltage_load'])
        post_temp = post_slice['temperature_battery'].to_numpy() if not post_slice.empty else np.full(10, last_row['temperature_battery'])

        if len(post_volt) < 10:
            post_volt = np.pad(post_volt, (0, 10 - len(post_volt)), mode='edge')
            post_temp = np.pad(post_temp, (0, 10 - len(post_temp)), mode='edge')

        for i in range(10):
            additional_row = last_row.copy()
            additional_row['time'] = last_row['time'] + (i + 1)
            additional_row['current_load'] = 0.0
            additional_row['voltage_load'] = post_volt[i]
            additional_row['temperature_battery'] = post_temp[i]
            additional_row['SOC'] = last_row['SOC']
            additional_row['cumulative_current'] = last_row['cumulative_current']
            additional_row['cumulative_current_total'] = last_row['cumulative_current_total']
            additional_row['cycle'] = cycle
            last_rows.append(additional_row)

    discharge_data = pd.concat([additional_rows, discharge_data, last_rows], ignore_index=True)
    discharge_data = discharge_data.sort_values(by=['cycle', 'time']).reset_index(drop=True)

    return discharge_data

 """


def PreProcess(file_path):
    data = pd.read_csv(file_path)

    # 1) Label cycles: increment when CHARGE starts (mode==1)
    cyc = 1
    cycles = []
    for i in range(len(data)):
        if data.loc[i, 'mode'] == 1 and (i == 0 or data.loc[i-1, 'mode'] != 1):
            cyc += 1
        cycles.append(cyc)
    data['cycle'] = cycles

    # 2) Drop charging rows; keep ALL rest + discharge
    data = data[data['mode'] != 1].copy().reset_index(drop=True)
    data = data.sort_values(['cycle','time']).reset_index(drop=True)

    # 3) Keep only cycles that actually have a discharge with |mean I| >= 5 A
    dis = data[data['mode'] == -1]
    valid_cycles = dis.groupby('cycle')['current_load'].mean().abs()
    valid_cycles = valid_cycles[valid_cycles >= 5].index
    data = data[data['cycle'].isin(valid_cycles)].copy()
    data = data.sort_values(['cycle','time']).reset_index(drop=True)

    # 4) Per-row deltas (hours)
    data['delta_t'] = data.groupby('cycle')['time'].diff().fillna(0) / 3600.0

    # 5) Initialize outputs
    data['SOC'] = 1.0
    data['SOC_Physics'] = 1.0
    data['cumulative_current'] = 0.0
    data['cumulative_current_total'] = 0.0
    data['cumulative_energy'] = 0.0
    data['effective_capacity'] = np.nan

    # 6) Compute metrics per cycle
    cum_Ah_total = 0.0
    cum_Wh_total = 0.0

    for cyc, g in data.groupby('cycle', sort=True):
        idxs = g.index.to_list()
        g_dis = g[g['mode'] == -1]

        # capacity (Ah) from discharge only
        if len(g_dis) > 1:
            cap = np.trapz(g_dis['current_load'].to_numpy(), g_dis['time'].to_numpy()) / 3600.0
            cap = float(abs(cap)) if not np.isnan(cap) else np.nan
        else:
            cap = np.nan

        data.loc[idxs, 'effective_capacity'] = cap

        # set initial SOC for cycle
        data.loc[idxs[0], ['SOC','SOC_Physics']] = 1.0, 1.0

        cum_Ah_cycle = 0.0
        for i, idx in enumerate(idxs):
            if i == 0:
                data.loc[idx, 'cumulative_current'] = cum_Ah_cycle
                data.loc[idx, 'cumulative_current_total'] = cum_Ah_total
                data.loc[idx, 'cumulative_energy'] = cum_Wh_total
                continue

            prev = idxs[i-1]
            dt = float(data.loc[idx, 'delta_t'])
            I  = float(data.loc[idx, 'current_load'])
            V  = float(data.loc[idx, 'voltage_load'])

            dAh = abs(I) * dt
            cum_Ah_cycle += dAh
            cum_Ah_total += dAh
            data.loc[idx, 'cumulative_current'] = cum_Ah_cycle
            data.loc[idx, 'cumulative_current_total'] = cum_Ah_total

            dWh = abs(I) * V * dt
            cum_Wh_total += dWh
            data.loc[idx, 'cumulative_energy'] = cum_Wh_total

            # SOC: change only during discharge, hold during rest
            if data.loc[idx, 'mode'] == -1 and cap and cap > 0:
                prev_soc = float(data.loc[prev, 'SOC'])
                data.loc[idx, 'SOC'] = max(prev_soc - (I * dt / cap), 0.0)
            else:
                data.loc[idx, 'SOC'] = data.loc[prev, 'SOC']

            # Physics SOC (2.5 Ah baseline during discharge)
            prev_phys = float(data.loc[prev, 'SOC_Physics'])
            if data.loc[idx, 'mode'] == -1:
                data.loc[idx, 'SOC_Physics'] = max(prev_phys - (I * dt / 2.5), 0.0)
            else:
                data.loc[idx, 'SOC_Physics'] = prev_phys

    return data


