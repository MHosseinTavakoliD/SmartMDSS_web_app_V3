# Virtual Environment
# All the units are in SI system
import numpy as np
#

# Virtual Environment
# All the units are in SI system
import numpy as np


#
#  Material calculation ################################################################################################################################################
def Matrial_on_the_road(Air_temp, Suraface_temp, Precipitation, wind_vel, initial_Snow, initial_water, initial_Salt,
                        initial_ice, plowing):
    # General assumptions:
    Dataset_hour_inc = 1  # *3 had applied in the dataset
    f_t = 0.9  # times when the road is covered with moving vehicle
    g_t = 1 - f_t

    # print ("Air_temp, Suraface_temp, Precipitation, wind_vel,initial_Snow, initial_water, initial_Salt, initial_ice, plowing")
    # print (Air_temp, Suraface_temp, Precipitation, wind_vel,initial_Snow, initial_water, initial_Salt, initial_ice, plowing)

    # convert Precipitation to snow or rain based on the air temp
    if Air_temp < .5:
        Snowfall_amount = Precipitation
        Rainfall_amount = 0
    else:
        Rainfall_amount = Precipitation
        Snowfall_amount = 0

    ###### Heat Ballance ##########################################################################################################
    # Q_csp : Flux of pavement heat
    V_wis = initial_ice + initial_Snow + Snowfall_amount  # Volume of WIS layer (m3*m-2 = m) depth of material on the road
    V_ps = 50e-3  # thikness of pavement surface, generally 25 to 55 mm, here we assume 50mm
    Lambda_wis = 0.8  # thermal conductivity of WIS layer. Assume that is compacted snow
    Lambda_P = 1.5  # Thermal conductivity of pavemetn. 0.8 to 2
    T_wis = (Air_temp + Suraface_temp) / 2  # Temp of wis layer: assumption
    Q_csp = 1 / ((V_wis / (2 * Lambda_wis)) + (V_ps / (2 * Lambda_P))) * (Suraface_temp - T_wis)

    # Q_rn : Flux of net radient heat
    q_rld = 20  # Since we are modeling the winter stroms, the sky radiation shouldn't be high 0-200
    q_rlu = 0.97 * 5.64e-8 * (T_wis + 273.15) ** 4
    q_rsd = 20  # Since we are modeling the winter stroms, the sky radiation shouldn't be high 0-300
    q_rsu = 0.3 * q_rsd
    Q_rn = f_t * q_rld + q_rlu + f_t * q_rsd - q_rsu

    # Q_sn Flux of net radiant heat
    q_sa = (10e4 * wind_vel ** 0.7 + 2.2) * (T_wis - Air_temp)  # snesable heat flux due to wind
    q_sf = (4.184 * (initial_water + Rainfall_amount) + 2.108 * (
                initial_ice + initial_Snow + Snowfall_amount))  # Sensable heat flux due to rainfall and snowfall
    q_sr = 0  # Sensible heat flux of drainage due to road gradient assume = 0
    q_sv = 4.184 * T_wis * 1000 * 0.5 * 0.005 / 3600 * 0.15  # sensible heat flux of water dispersion due to passing vehicle
    Q_sn = q_sa + f_t * q_sf + q_sr + g_t * q_sv

    # Q_ln flux of net latent heat
    # m_wi * L_wi is not considered here, we will change the q_net relation with M_wi later
    m_il = 0  # sublimation flux
    L_i = 2838  # kJkg-1 latent heat of sublimation
    m_wl = 0  # flux of evaporation and condensation
    L_w = 2260  # kJkg-1 latent heat of evaporation and condensation
    m_sl = 3.34e-5  # dissolving flux (0 - 6.67e-5) if salting
    L_s = -66.4  # latent heat of dissolution of salt
    Q_ln = m_il * L_i + m_wl * L_w + m_sl * L_s

    # Q_vn flux of net vehicle heat
    Q_vn = 100  # assumption according to Fujimoto and other citations

    # Q_net
    Q_net = Q_csp + Q_rn + Q_sn + Q_ln + Q_vn
    #################################################################################################################################
    # *******************************************************************************************************************************#

    ## Material creation $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Ice
    Ice_final = initial_ice

    # Snowfall
    m_snowf = Snowfall_amount * Dataset_hour_inc * 100  # 100 kg/m3
    Snow_creat = m_snowf * f_t
    Snow_final = Snow_creat + initial_Snow
    # plowing  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    if plowing == 1:
        Snow_final *= 0.1

    # or rainfall
    m_wf = Rainfall_amount * Dataset_hour_inc * 1000  # 1000 kg/m3
    Rain_creat = m_wf * f_t
    Water_final = Rain_creat + initial_water
    # before mesuring the water to ice, we have to consider the water dispersion
    # Water dispersion
    # Assume we have good drainage system
    Water_final = Water_final * 0.95

    # Salt
    # Salt_final = initial_Salt * 0.5# Maybe need conversion

    ### Material Conversion $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    ## Snow to ice
    Snow_To_Ice_rate = (0.1 if initial_Snow > 0 else 0)  # Assumption: conversion applied to initial snow
    Ice_final += Snow_To_Ice_rate * initial_Snow

    ## water to ice or vise versa *
    L_wi = 334  # kJkg-1 latent heat of melting and freezing
    Water_ice_conv_mass = 5*abs(
        Q_net * 1000 / (L_wi * 999)) if initial_Snow > 0 or initial_water > 0 or initial_ice > 0 else 0
    # We will assume that the water freezing point is just a function of salt concentration
    T_frezzing = -0.025 * initial_Salt - 0.5  # 0.04 * initial_Salt + 1#
    # T_frezzing = (T_frezzing -32)*5/9
    # print("TF",T_frezzing, "initial_Salt", initial_Salt, "T_wis", T_wis  )

    if T_wis >= T_frezzing:
        STATUS = "Melting"
        # print("###############################Melting")  # Melting
        stat_sign = 1
        Ice_final = Ice_final - min(Water_ice_conv_mass, Ice_final)
        Water_final = Water_final + min(Water_ice_conv_mass, Ice_final)
    elif T_wis < T_frezzing and initial_Salt == 0:
        STATUS = "Freezing"
        # print("###############################Freezing")  # Freezing
        Ice_final = Ice_final + min(Water_ice_conv_mass, Water_final)
        Water_final = Water_final - min(Water_ice_conv_mass, Water_final)
    elif T_wis < T_frezzing and initial_Salt > 0:
        STATUS = "Freezing slowly"
        # print("###############################Freezing slowly")  # Freezing
        Ice_final = 0.2 * (Ice_final + min(Water_ice_conv_mass, Water_final))
        Water_final = Water_final - min(Water_ice_conv_mass, Water_final)

    # Snow to water
    if STATUS == "Melting":
        Water_final = Water_final + stat_sign * Water_ice_conv_mass
        Water_final = Water_final * 0.05  # disperssed
        Snow_final = Snow_final - stat_sign * Water_ice_conv_mass
        # print ("Water_ice_conv_mass", Water_ice_conv_mass)
    ## Material dispersion $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Salt dispersion: depands on STATUS
    if STATUS == "Melting":
        Salt_disp_rate = 0.9
    if STATUS == "Freezing" or STATUS == "Freezing slowly":
        Salt_disp_rate = 0.98
    Salt_final = initial_Salt * Salt_disp_rate

    # Check each variable and set to zero if negative
    Water_final = max(Water_final, 0)
    Salt_final = max(Salt_final, 0)
    Ice_final = max(Ice_final, 0)
    Snow_final = max(Snow_final, 0)

    # print ("Water_final, Salt_final, Ice_final, Snow_final", Water_final, Salt_final, Ice_final, Snow_final)
    return Water_final, Salt_final, Ice_final, Snow_final
#  Agent    ############################################################################################################################################################
import numpy as np
import pandas as pd
from collections import deque
import random
from tensorflow.keras.models import Sequential, load_model



class DQNAgent:
    def __init__(self, state_size, action_size, model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.action_list_size = action_size
        self.memory = deque(maxlen=10000)  #
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.00001  # <<<<<<<<<<
        self.Number_ofExploitation = 0
        self.Number_ofExploration = 0
        if model_path:
            self.model = load_model(model_path)
            print("Model loaded from", model_path)
        else:
            self.model = self._build_model()
            print("New model initialized")

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        ## Input layer and first hidden layer
        model.add(Dense(48, input_dim=total_state_size, activation='relu'))
        ## Second hidden layer
        # model.add(Dense(96, activation='relu'))
        # ## Adding Dropout to prevent overfitting
        # model.add(Dropout(0.2))

        ## Third hidden layer
        # model.add(Dense(512, activation='relu'))
        ## Adding Dropout to prevent overfitting
        # model.add(Dropout(0.2))

        ## Fourth hidden layer
        # model.add(Dense(256, activation='relu'))

        model.add(Dense(124, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(124, activation='relu'))
        model.add(Dense(self.action_list_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if True:  # np.random.rand() <= self.epsilon:
            # Biased action selection
            actions_probability = [0.4, 0.2, 0.2, 0.2]  # # Example: 40% do nothing, 20% for each other action
            self.Number_ofExploration += 1
            return np.random.choice(np.arange(self.action_size), p=actions_probability)
        else:
            act_values = self.model.predict(state)
            self.Number_ofExploitation += 1
            return np.argmax(act_values[0])

    def predict_action(self, state):
        """Selects the action with the highest Q-value for the given state."""
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#   Env ############################################################################################################################################################

import numpy as np


class RoadEnv:
    def __init__(self, dataset, unique_window_ids):
        self.dataset = dataset
        self.current_state = None
        self.unique_window_ids = unique_window_ids
        self.current_episode_index = 0
        # self.action_costs = {0: 0, 1: 1, 2: 6, 3: 5}
        self.reset()

    #     def simulate_no_intervention(self, window_data):
    #         no_intervention_results = []
    #         initial_conditions = {'initial_Snow': 0, 'initial_water': 0, 'initial_Salt': 0, 'initial_ice': 0, 'plowing':0}
    #         for index, row in window_data.iterrows():
    #             # Simulate step without interventions
    #             Air_temp = row['Air TemperatureC']
    #             Surface_temp = row['Surface TemperatureC']
    #             Precipitation = row['Precipitation Intensitym/3h']
    #             wind_vel = row['Wind Speed (act)m/s']
    #             water, salt, ice, snow = Matrial_on_the_road(Air_temp, Surface_temp, Precipitation, wind_vel, **initial_conditions)
    #             # print ("water, salt, ice, snow", water, salt, ice, snow)
    #             # Update initial conditions for the next step
    #             initial_conditions.update({'initial_Snow': snow, 'initial_ice': ice, 'initial_water': water, 'initial_Salt': salt, 'plowing': 0})

    #             # Append results for no intervention forecast
    #             no_intervention_results.append((water, salt, ice, snow))
    #             # print("no_intervention_results",no_intervention_results)
    #         return no_intervention_results
    # def reset(self):
    #     # Reset the environment to the initial state of a new episode
    #     self.current_state = self.dataset.iloc[0].copy()
    #     self.plowing_count = 0
    #     self.salting_count = 0
    #
    #     return self._get_state()
    def reset(self, episode_index=0):

        if self.current_state is not None:
            # Initialize the state properties if current_state is properly set
            self.current_state['initial_Snow'] = 0
            self.current_state['initial_water'] = 0
            self.current_state['initial_Salt'] = 0
            self.current_state['initial_ice'] = 0
        # print ("RESET self.current_state" )
        # print (self.current_state)
        # Calculate the start index of the episode in the dataset
        self.start_index = episode_index * 10  # Assuming each episode has 10 states
        self.current_state_index = self.start_index
        window_id = self.unique_window_ids[episode_index]
        # print ("window_id", window_id)
        # Check if the calculated start index is within the bounds of the dataset
        if self.start_index >= 0 and self.start_index < len(self.dataset):
            self.current_state = self.dataset.iloc[self.start_index].copy()
        else:
            # Fallback to the first row if the calculated index is out of bounds
            self.current_state = self.dataset.iloc[0].copy()
            print(f"Warning: Episode index {episode_index} out of bounds, falling back to initial state.")

        # Reset the action counts for the new episode
        self.plowing_count = 0
        self.salting_count = 0

        # Reset environment to the start of a new episode, with modifications to include no intervention forecast
        window_id = unique_window_ids[episode_index]
        episode_data = self.dataset[self.dataset['Window_ID'] == window_id]
        self.Lowest_temp = min(min(episode_data['Surface TemperatureC']), min(episode_data['Air TemperatureC']))
        # print ("self.Lowest_temp", self.Lowest_temp)
        # self.no_intervention_forecast = self.simulate_no_intervention(episode_data)
        # print ("self.no_intervention_forecast", self.no_intervention_forecast)
        # Return the initial state of the new episode
        return self._get_state()

    def step(self, action):
        reward = 0  # 20000

        # Check if the action is allowed based on the resource limitations
        if action == 1 and self.plowing_count >= 3:  # Plowing limit reached
            reward = -1000  # Force 'no action'
        elif action == 3 and self.salting_count >= 2:  # Salting limit reached
            reward = -1000  # Force 'no action'
        elif action == 2:  # Plowing and salting
            if self.plowing_count >= 3 and self.salting_count >= 2:
                reward = -2000  # If either limit is reached, force 'no action'
            elif self.plowing_count >= 3:
                reward = -1000
            elif self.salting_count >= 2:
                reward = -1000

        # Update action counters
        if action == 1:  # Plowing
            self.plowing_count += 1
        elif action == 3:  # Salting
            self.salting_count += 1
        elif action == 2:  # Plowing and salting
            self.plowing_count += 1
            self.salting_count += 1
        # Apply the chosen action to the environment and update the state
        # print ("action", action)

        water, salt, ice, snow = self._apply_action(action)
        # print (" water, salt, ice, snow", water, salt, ice, snow)
        # Update the current state with the new values
        self.current_state['initial_Snow'] = snow
        self.current_state['initial_water'] = water
        self.current_state['initial_Salt'] = salt
        self.current_state['initial_ice'] = ice
        # print ("self.current_state &&&&&Step", self.current_state)
        # Calculate the reward
        reward += self._calculate_reward(action, ice, snow, salt)
        # print ("reward",reward)
        # Move to the next row in the dataset, simulating time progression

        # Dynamically determine if it's the last state in the episode
        current_episode_index = self.dataset.index.get_loc(self.current_state.name)
        # print ("current_episode_index *********************   ", current_episode_index)
        # episode_start_index = current_episode_index - (current_episode_index % 10)  # Assuming episodes start at indices 0, 10, 20, ...
        # next_index = current_episode_index + 1
        # is_last_state_in_episode = (next_index % 10 == 0) or (next_index >= len(self.dataset))
        is_last_state_in_episode = (self.current_state_index - self.start_index == 9)
        if is_last_state_in_episode:
            done = True

        else:
            self.current_state_index = self.current_state_index + 1

            next_state = self.dataset.iloc[self.current_state_index].copy()
            for col in ['MeasureTime', 'Window_ID', 'Air TemperatureC', 'Surface TemperatureC', 'Wind Speed (act)m/s',
                        'Precipitation Intensitym/3h']:
                self.current_state[col] = next_state[col]
            done = False
        # print ("self.current_state_index", self.current_state_index, "self.start_index", self.start_index)
        # print ("SETP self.current_state" )
        # print (self.current_state)

        return self._get_state(), reward, done

    def _get_state(self):
        # Extract the current state from the dataset row
        current_state_values = [self.current_state[col] for col in ['Air TemperatureC', 'Surface TemperatureC',
                                                                    'Wind Speed (act)m/s',
                                                                    'Precipitation Intensitym/3h',
                                                                    'initial_Snow', 'initial_water', 'initial_Salt',
                                                                    'initial_ice']]

        # Prepare no intervention forecast data for the entire episode
        # Assuming no_intervention_forecast is a list of tuples for the entire episode at this point
        # no_intervention_forecast_flat = [value for forecast in self.no_intervention_forecast for value in forecast]

        # Combine the current state values with the no intervention forecast
        # Here, we ensure the state_with_forecast includes the no intervention data for the entire episode
        state_ = np.array(current_state_values)  # + no_intervention_forecast_flat)
        # print ("state_with_forecast", state_with_forecast)
        return state_

    def _apply_action(self, action):
        # Apply the chosen action and update the road conditions
        # For simplicity, this function assumes the existence of a function named 'Matrial_on_the_road'
        # that calculates the final amounts of water, salt, ice, and snow

        air_temp = self.current_state['Air TemperatureC']
        surface_temp = self.current_state['Surface TemperatureC']
        precipitation = self.current_state['Precipitation Intensitym/3h']
        wind_vel = self.current_state['Wind Speed (act)m/s']
        initial_snow = self.current_state['initial_Snow']
        initial_water = self.current_state['initial_water']
        initial_salt = self.current_state['initial_Salt']
        initial_ice = self.current_state['initial_ice']
        plowing = 0
        # print ("self.current_state _apply_action", self.current_state)
        if action == 1 or action == 2:  # Plowing or Plowing and Salting
            plowing = 1

        if action == 2 or action == 3:  # Salting or Plowing and Salting
            # Measuring salt amount based on the Ruled-Base model
            Salt_amount = -25 * self.Lowest_temp + 25  # -32.027 * self.Lowest_temp + 79.24
            # print ("Salt_amount: ", Salt_amount)
            if Salt_amount < 0: Salt_amount = 0
            if Salt_amount > 400: Salt_amount = 400  # here salt unit is lb/mile/lane
            initial_salt = Salt_amount  # Assuming a constant amount of salt is used

        water_final, salt_final, ice_final, snow_final = Matrial_on_the_road(
            air_temp, surface_temp, precipitation, wind_vel,
            initial_snow, initial_water, initial_salt, initial_ice, plowing
        )
        # print (f"action {action}, water_final {water_final}, salt_final {salt_final}, ice_final {ice_final}, snow_final {snow_final}" )
        return water_final, salt_final, ice_final, snow_final

    def _calculate_reward(self, action, ice, snow, salt):
        # Calculate the reward based on the reduction of ice and snow, and the cost of the action
        # cost = self.action_costs[action]
        cost = 0
        if action == 1:  # Plowing
            cost = 12  # 4.3
        if action == 2:  # Plowing and saltin
            cost = 12 + 0.07 * salt
        if action == 3:
            cost = 0.07 * salt  # 4.5 + 0.07*salt
        # Penalties
        # Convert kg/m2 to inch of snow
        snow_convert = snow / 2.5  # 100kg/m3
        ice_convert = ice / 6 / 2.5  # 600 kg/m3

        # function correlating snow accumulation depth to decreased traffic speed
        f_spped_snow = 0
        if snow_convert <= 0.1:
            f_spped_snow = 0
        elif snow_convert < 0.5 and snow_convert > 0.1:
            f_spped_snow = snow_convert * 0.2 * 100
        elif snow_convert >= 0.5:
            f_spped_snow = snow_convert * 0.15 * 100

        f_spped_ice = 0
        if ice_convert <= 0.05:
            f_spped_ice = 0
        elif ice_convert > 0.05:
            f_spped_ice = 5 * ice_convert

        # Cost per 3h/mile of no maintence is 250$
        Cost_speed = min(250, max(f_spped_ice * 250, f_spped_snow * 250))

        # function correlating snow accumulation depth to accident probability
        f_accident_snow = 0
        if snow_convert <= 0.1:
            f_accident_snow = 0
        elif snow_convert < 5 and snow_convert > 0.1:
            f_accident_snow = (snow_convert - 0.1) / (4.9)
        elif snow_convert >= 0.5:
            f_accident_snow = 1

        f_accident_ice = 0
        if ice_convert <= 0.05:
            f_accident_ice = 0
        elif ice_convert > 0.05 and ice_convert < 0.5:
            f_accident_ice = (ice_convert - 0.05) / (0.45)
        elif ice_convert >= 0.5:
            f_accident_ice = 1

        # •	Cost accident is the cost associated with accidents
        Cost_accident = f_accident_snow * 25 + f_accident_ice * 17

        #         print ("Cost_accident", Cost_accident)
        #         print ("Salt", salt)

        reward = - cost - Cost_speed - Cost_accident

        return reward

#  Predict  ############################################################################################################################################################
# Expolite
import pandas as pd
import numpy as np
import os


# Load and preprocess dataset
import pymongo
from pymongo import MongoClient

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')  # Adjust the connection URL as needed
db = client['SmartMDSS_data']
collection = db['Test_winter']

# Fetch data from MongoDB
data = collection.find_one({'forecast': {'$exists': True}})  # Assuming one document format
df = pd.DataFrame(data['forecast'])

# Convert data types and perform unit conversions
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])
df['Air TemperatureC'] = (df['Air TemperatureF'] - 32) * 5.0/9.0  # F to C
df['Surface TemperatureC'] = (df['Surface TemperatureF'] - 32) * 5.0/9.0  # F to C
df['Wind Speed (act)m/s'] = df['Wind Speed (act)mph'] * 0.44704  # mph to m/s
df['Precipitation Intensitym/3h'] = df[['snow', 'rain']].sum(axis=1)  * 25.4 * 3  # inches/h to mm/3h

# Drop the columns that are no longer necessary
df.drop(['Air TemperatureF', 'Surface TemperatureF', 'Wind Speed (act)mph', 'snow', 'rain'], axis=1, inplace=True)

# Set index to MeasureTime
df.set_index('MeasureTime', inplace=True)

# Calculate total duration and determine resampling period
total_duration = (df.index.max() - df.index.min())
step_duration = total_duration / 9  # Dividing by 9 gives 10 intervals

# Resample data into 10 equal intervals
df_resampled = df.resample(step_duration).mean().reset_index()

# Add the window ID (assuming it is constant as 'Appleton_window_1')
df_resampled['Window_ID'] = 'Appleton_window_1'

# Reorder columns to match the desired output
dataset = df_resampled[['MeasureTime', 'Window_ID', 'Air TemperatureC', 'Surface TemperatureC', 'Wind Speed (act)m/s', 'Precipitation Intensitym/3h']]
pd.set_option('display.max_rows', None)  # or set to df_resampled.shape[0]
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
for col in ['initial_Snow', 'initial_water', 'initial_Salt', 'initial_ice']:
    if col not in dataset.columns:
        dataset.loc[:, col] = 0
print(dataset.head(10))

# Initialize environment and agent


action_size = 4
no_intervention_data_size = 0  # 10 * 4  # 10 future states * 4 variables per state
current_state_size = 8  # Current state variables
total_state_size = current_state_size + no_intervention_data_size
count = 0
model_directory = 'DRL_models_soph/'
best_cumulative_reward = -100000
best_model_data = None
models = [file for file in os.listdir(model_directory) if file.endswith('.h5')]
for model_file in models:
    print ("Model ", count)
    count +=1
    model_path = os.path.join(model_directory, model_file)
    agent = DQNAgent(total_state_size, action_size, model_path=model_path)

    # Split dataset into training and validation
    unique_window_ids = dataset['Window_ID'].unique()
    env = RoadEnv(dataset, unique_window_ids)

    print("len(unique_window_ids)", len(unique_window_ids))

    current_episode_data =  dataset[dataset['Window_ID'] == 'Appleton_window_1']

    state = env.reset(0)  # Reset environment and action counters

    episode_actions = []
    episode_info = []
    cumulative_reward = 0
##################################   With Intervention  #########################################
    for index, row in current_episode_data.iterrows():
        state = np.reshape(state, [1, total_state_size])

        # Use predict_action to exploit the learned policy
        action = agent.predict_action(state)

        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, total_state_size])
        cumulative_reward += reward

        episode_actions.append(action)
        episode_info.append({
            'state': state.flatten().tolist(),
            'action': action,
            'reward': reward,
            'next_state': next_state.flatten().tolist(),
            'cumulative_reward': cumulative_reward,
            'done': done
        })

        state = next_state
        if done:
            break
    print ('cumulative_reward', cumulative_reward)

    # Update best model tracking
    if cumulative_reward > best_cumulative_reward:
        print ("best_cumulative_reward", best_cumulative_reward)
        best_cumulative_reward = cumulative_reward
        print("best_cumulative_reward", best_cumulative_reward)
        best_model_data = {
            'model_path': model_path,
            'cumulative_reward': cumulative_reward,
            'episode_info': episode_info
        }
print('final########final########final########final########final########final########final########final########')
print ("best_model_data", best_model_data)
# print("Actions taken:", episode_actions)
# print("Cumulative Reward:", cumulative_reward)
# print ("episode_info", episode_info)
print ("$$$$$$$$$$$$$$^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^$$$$$$$$$$$$$$$$$")
# for info in episode_info:
#     print(info)
# Log the actions for each episode
#     print (f"Episode , Actions taken: {episode_actions}, Cumulative Reward: {cumulative_reward}\n")
# print ("Best model", best_model_data)
# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['SmartMDSS_data']
collection = db['Test_winter']  # New collection for maintenance suggestions

# Check if there are any non-zero actions
if  best_model_data :
    print ("check if best_model_data")
    # Create a list to hold maintenance suggestions for states where actions were taken
    maintenance_suggestions = []
    for index, info in enumerate(best_model_data['episode_info']):
        suggestion = {
            'time': dataset['MeasureTime'].iloc[index].to_pydatetime(),
            'initial_Snow': float(info['state'][4]),
            'initial_water': float(info['state'][5]),
            'initial_Salt': float(info['state'][6]),
            'initial_ice': float(info['state'][7]),
            'action': int(info['action']),
            'reward': float(info['reward']),
            'cumulative_reward': float(info['cumulative_reward']),
            'done': info['done']
        }
        maintenance_suggestions.append(suggestion)

    # Update the single document in the collection
    if maintenance_suggestions:  # Check if there are any suggestions to add
        print ("It is true")
        result = collection.update_one(
            {'forecast': {'$exists': True}},  # Assuming this is how you identify the document
            {'$set': {'Winter_mntnc_sugg': maintenance_suggestions}}
        )
        print(f"Updated {result.matched_count} document(s), Modified {result.modified_count} document(s).")
else:
    print("No non-zero actions taken, no document updated.")

# Ensure the script executes fully and handles MongoDB operations correctly.
##################################   Without Intervention  #########################################
episode_actions = []
episode_info = []
cumulative_reward = 0
state = env.reset(0)  # Reset environment and action counters

##################################   With Intervention  #########################################
for index, row in current_episode_data.iterrows():
    state = np.reshape(state, [1, total_state_size])

    # Use predict_action to exploit the learned policy
    action = 0#agent.predict_action(state)

    next_state, reward, done = env.step(action)
    next_state = np.reshape(next_state, [1, total_state_size])
    cumulative_reward += reward

    episode_actions.append(action)
    episode_info.append({
        'state': state.flatten().tolist(),
        'action': action,
        'reward': reward,
        'next_state': next_state.flatten().tolist(),
        'cumulative_reward': cumulative_reward,
        'done': done
    })

    state = next_state
    if done:
        break

# print(f"\nEpisode for Window ID: {window_id}")
# print("Actions taken:", episode_actions)
# print("Cumulative Reward:", cumulative_reward)
print ("episode_info", episode_info)
print ("$$$$$$$$$$$$$$^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^$$$$$$$$$$$$$$$$$")
for info in episode_info:
    print(info)
# Log the actions for each episode
    print (f"Episode , Actions taken: {episode_actions}, Cumulative Reward: {cumulative_reward}\n")

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['SmartMDSS_data']
collection = db['Test_winter']  # New collection for maintenance suggestions

# Check if there are any non-zero actions
if True: #any(action != 0 for action in episode_actions):
    # Create a list to hold maintenance suggestions for states where actions were taken
    maintenance_suggestions = []
    for index, info in enumerate(episode_info):
        suggestion = {
            'time': dataset['MeasureTime'].iloc[index].to_pydatetime(),
            'initial_Snow': float(info['state'][4]),
            'initial_water': float(info['state'][5]),
            'initial_Salt': float(info['state'][6]),
            'initial_ice': float(info['state'][7]),
            'action': int(info['action']),
            'reward': float(info['reward']),
            'cumulative_reward': float(info['cumulative_reward']),
            'done': info['done']
        }
        maintenance_suggestions.append(suggestion)

    # Update the single document in the collection
    if maintenance_suggestions:  # Check if there are any suggestions to add
        result = collection.update_one(
            {'forecast': {'$exists': True}},  # Assuming this is how you identify the document
            {'$set': {'Winter_cond_no_intrvn': maintenance_suggestions}}
        )
        print(f"Updated {result.matched_count} document(s), Modified {result.modified_count} document(s).")
else:
    print("No non-zero actions taken, no document updated.")
