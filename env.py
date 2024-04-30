'''
This script implements the crop plantation environment based on AquaCrop-OSPy model. 
'''

from aquacrop import InitialWaterContent, Crop, Soil, AquaCropModel 
from aquacrop import IrrigationManagement as IrrMngtClass

import gym
from gym import spaces

import numpy as np 


class CropEnv(gym.Env):
    """
    AquaCrop-OSPy crop environment in openai-gym style.

    Cropping system consists of a single crop on 1 HA of homogenous soil.
    Each episode consits of 1 season of this system

    Config parameters will specify the type of cropping environment.

    Every `days_to_irr` days, the agent will see an observation of the enviornemnt

    The agent will then make an irrigation decision which is passed to the environment

    This proicess continues until season has finished

    the seasonal profit is calculated and apssed to the agent as the reward
    """
 
    def __init__(self,config):

        super(CropEnv, self).__init__()
    
        # unpack config parameters, which define the variations of environment 
        self.gendf = config["gendf"] # this is the weather data of Nebraska, including temperatures, precipitation, and ETo 
        self.days_to_irr=config["days_to_irr"] # this is the frequency of making a decision - how often we change the thresholds 
        self.year1=config["year1"] # this is the start year of the training data 
        self.year2=config["year2"] # this is the end year of the training data  
        self.dayshift = config["dayshift"] # this is the possible shift of planting date of the crop 
        self.max_irr=config["max_irr"] # this is the maximum irrigation depth per day 
        self.init_wc = config["init_wc"] # this is the initial water content of the soil 
        self.CROP_PRICE=config["crop_price"] # this is the price of the crop 
        self.IRRIGATION_COST=config["irrigation_cost"] # this is the cost of irrigation 
        self.FIXED_COST = config["fixed_cost"] # this is the fixed cost of irrigation per year 
        self.planting_month = int(config['planting_date'].split('/')[0]) # this is the month of planting 
        self.planting_day = int(config['planting_date'].split('/')[1]) # this is the day of planting 
        self.max_irr_season=config['max_irr_season'] # this is the maximum irrigation application for the season 
        self.simcalyear=config["simcalyear"] # this is the year of simulation passed into the AquaModel 
        self.observation_set=config["observation_set"] # this is the choice of state definition 
        self.forecast_lead_time=config["forecast_lead_time"] # if we are using weather forecast, this is the number of days ahead 
        self.split = self.year2 # this is the split year between training and testing data 

        # we randomly drawn weather year for simulation 
        self.chosen = np.random.choice([i for i in range(self.year1,self.year2+1)])

        # crop and soil choice 
        self.crop = Crop(config['crop'], planting_date=config['planting_date']) 
        self.soil = Soil(config['soil']) 

        # we record the best profit, yield, and water use for each year 
        self.best_profit = np.ones(self.year2-self.year1+1) * (-500) 
        self.best_yield = np.ones(self.year2-self.year1+1) * (-1) 
        self.best_water = np.ones(self.year2-self.year1+1) * (1000) 

        # the number of trajectories completed 
        self.tsteps=0 

        # obsservation and action sets 
        # since each state definition is different, we need to define the observation space for each case 
        if self.observation_set == 'default':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32) 
        elif self.observation_set == 'forecast': 
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32) 
        elif self.observation_set == 'temperature':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32) 
        elif self.observation_set == 'no eto':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32) 
        
        # the action space is continuous with dimension 4 
        self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)  

        # we record the training curve, yield curve, and water curve, for later access after training 
        self.train_curve = [] 
        self.yield_curve = []    
        self.water_curve = [] 
        self.yields = [] 

        # this is the weight of the reward for yield and water use 
        self.reward_scale = config['reward_scale'] # (yield scale weight, water scale weight) 

                
    def states(self):
        # return observation space 
        return dict(type='float', shape=(self.observation_space.shape[0],))
 
    def actions(self):
        # return action space 
        return dict(type='float', num_values=self.action_space.shape[0])
        
    def reset(self):
        """
        re-initialize model and return first observation
        """ 
        
        # choose a random training year to simulate each time we reset the environment 
        sim_year=int(np.random.choice(np.arange(self.year1,self.year2+1))) 
        # get the weather data for that year 
        self.wdf = self.gendf[self.gendf.simyear==sim_year].drop('simyear',axis=1)
        # record our choice 
        self.chosen=sim_year*1 

        # irrigation cap
        self.chosen_max_irr_season =self.max_irr_season*1.
            
        # create the simulation model of plantation 
        self.model = AquaCropModel(
            f'{self.simcalyear}/{self.planting_month}/{self.planting_day}',f'{self.simcalyear}/12/31',
            self.wdf,self.soil,self.crop,
            irrigation_management=IrrMngtClass(irrigation_method=5,MaxIrrSeason=self.chosen_max_irr_season),
            initial_water_content=self.init_wc
        )
        self.model._initialize() # initialize our simulation model 


        # shift the start day of simulation by random amount, which happens in reality 
        if self.dayshift:
            dayshift=np.random.randint(1,self.dayshift+1) 
            self.model.run_model(dayshift, initialize_model=False)
        
        # we store the irrigation events
        self.irr_sched=[]

        return self.get_obs(self.model._init_cond) 
    
 
    def get_obs(self,_init_cond):
        """
        package variables from _init_cond into a numpy array
        and return as observation
        """

        # calculate relative depletion 
        if _init_cond.taw>0:
            dep = _init_cond.depletion/_init_cond.taw
        else:
            dep=0

        # calculate mean daily precipitation and ETo from last 7 days
        start = max(0,self.model._clock_struct.time_step_counter -7)
        end = self.model._clock_struct.time_step_counter
        forecast1 = self.model.weather_df.to_numpy()[start:end,2:4].mean(axis=0).flatten()

        # calculate mean max temperature from last 7 days
        temperature = self.model.weather_df.to_numpy()[start:end,0].mean(axis=0).flatten()

        # calculate sum of daily precipitation and ETo for whole season so far
        start2 = max(0,self.model._clock_struct.time_step_counter -_init_cond.dap)
        forecastsum = self.model.weather_df.to_numpy()[start2:end,2:4].sum(axis=0).flatten()

        #  yesterday precipitation and ETo and irr
        start2 = max(0,self.model._clock_struct.time_step_counter-1)
        forecast_lag1 = self.model.weather_df.to_numpy()[start2:end,2:4].flatten() 

        # yesterday max temperature 
        temperature_lag1 = self.model.weather_df.to_numpy()[start2:end,0].flatten() 

        # calculate mean daily precipitation and ETo for next N days
        start = self.model._clock_struct.time_step_counter
        end = start+self.forecast_lead_time
        forecast2 = self.model.weather_df.to_numpy()[start:end,2:4].mean(axis=0).flatten()
        
        # state 
        # month and day
        month = (self.model._clock_struct.time_span[self.model._clock_struct.time_step_counter]).month
        day = (self.model._clock_struct.time_span[self.model._clock_struct.time_step_counter]).day
        
        # concatenate all weather variables 
        if self.observation_set == 'default':
            forecast = np.concatenate([forecast1,forecastsum,forecast_lag1]).flatten()
        
        elif self.observation_set=='forecast':
            forecast = np.concatenate([forecast1,forecastsum,forecast_lag1,forecast2,]).flatten() 

        elif self.observation_set=='temperature': 
            forecast = np.concatenate([forecast1,forecastsum,forecast_lag1, temperature, temperature_lag1]).flatten()

        elif self.observation_set=='no eto': 
            forecast = np.concatenate([[forecast1[0]],[forecastsum[0]],[forecast_lag1[0]]]).flatten()


        # put growth stage in one-hot format 
        gs = np.clip(int(self.model._init_cond.growth_stage)-1,0,4)
        gs_1h = np.zeros(4)
        gs_1h[gs]=1

        # create observation array with additional features, that is applicable to all state definitions 
        obs=np.array([
            day,
            month,
            dep, # root-zone depletion
            _init_cond.dap,#days after planting
            _init_cond.irr_net_cum, # irrigation used so far
            _init_cond.canopy_cover, # canopy cover 
            _init_cond.biomass, # biomass 
            self.chosen_max_irr_season-_init_cond.irr_net_cum, # remaining irrigation availibility         
            ]
            +[f for f in forecast]
            +[g for g in gs_1h]
            , dtype=np.float32).reshape(-1)
        
        return obs 
        
    def step(self,action):
        """
        Take in agents action choice

        apply irrigation depth on following day

        simulate N days until next irrigation decision point

        calculate and return profit at end of season

        """ 

        new_smt=np.ones(4)*(action+1)*50 # shape 4, range 0-100, float 

        start_day = self.model._init_cond.dap # our starting day of simulation 

        for i in range(self.days_to_irr): # run N days, this is how often we make a decision 

            # calculate depletion 
            if self.model._init_cond.taw>0:
                dep = self.model._init_cond.depletion/self.model._init_cond.taw
            else:
                dep=0

            # obtain the growth stage 
            gs = int(self.model._init_cond.growth_stage)-1
            if gs<0 or gs>3:
                depth=0 # no irrigation 
            else:
                if 1-dep < (new_smt[gs])/100: # if water lower than threshold, we irrigate 
                    depth = np.clip(self.model._init_cond.depletion,0,self.max_irr) # cap by the daily maximum amount 
                else:
                    depth=0

            # assign the amount we'd like to irrigate 
            self.model._param_struct.IrrMngt.depth = depth 
            self.irr_sched.append(self.model._param_struct.IrrMngt.depth) # record the irrigation schedule 

            self.model.run_model(initialize_model=False) # simulate for one day 

            # termination conditions
            # if this is the end of the season, we don't run the simulation anymore  
            if self.model._clock_struct.model_is_finished is True:
                break

            now_day = self.model._init_cond.dap
            if (now_day >0) and (now_day<start_day):
                # end of season 
                break
 
 
        done = self.model._clock_struct.model_is_finished # done if the season is finished 
        
        reward = 0 # reward is by default 0, and only non-zero when the season ends and we have the yield 
 
        next_obs = self.get_obs(self.model._init_cond) # get the next observation 
 
        if done:
        
            self.tsteps+=1 # increment the number of trajectories completed 

            crop_yield = self.model._outputs.final_stats['Yield potential (tonne/ha)'].mean() # get the crop yield 
            water_use = self.model._outputs.final_stats['Seasonal irrigation (mm)'].mean() # get the water use 
            profit = self.CROP_PRICE * crop_yield - self.IRRIGATION_COST * water_use - self.FIXED_COST # calculate the profit 
            end_reward = self.reward_scale[0]*self.CROP_PRICE*crop_yield \
                - self.reward_scale[1]*water_use*self.IRRIGATION_COST \
                - self.FIXED_COST # scale the reward by the weights, by default the weights are 1,1 
            self.reward = end_reward 
            self.yields.append(crop_yield) # record the yield 

            # keep track of best rewards in each season 
            # there can be multiple rewards for a season because of possibily repeated simulations 
            # only comparisons for the same year are meaningful due to weather variability 
            if self.chosen < self.split: 
                self.best_profit[self.chosen-1] = max(self.best_profit[self.chosen-1],profit) 
                self.best_yield[self.chosen-1] = max(self.best_yield[self.chosen-1],crop_yield)
                self.best_water[self.chosen-1] = min(self.best_water[self.chosen-1],water_use) 
                
                self.train_curve.append(np.mean(self.best_profit)) 
                self.yield_curve.append(np.mean(self.best_yield)) 
                self.water_curve.append(np.mean(self.best_water)) 

            reward = end_reward/1000 # scale reward down by 1000 for stability 

 
        return next_obs, reward, done, dict() # return the next observation, reward, done, and an empty info 
 
 
   