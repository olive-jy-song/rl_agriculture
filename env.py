'''
originally derived from 
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
 
        self.gendf = config["gendf"]
        self.days_to_irr=config["days_to_irr"]
        self.year1=config["year1"]
        self.year2=config["year2"] 
        self.dayshift = config["dayshift"]
        self.max_irr=config["max_irr"]
        self.init_wc = config["init_wc"]
        self.CROP_PRICE=config["crop_price"]
        self.IRRIGATION_COST=config["irrigation_cost"] 
        self.FIXED_COST = config["fixed_cost"]
        self.planting_month = int(config['planting_date'].split('/')[0])
        self.planting_day = int(config['planting_date'].split('/')[1])
        self.max_irr_season=config['max_irr_season']
        self.name=config["name"]
        # self.best=config["best"]*1
        self.simcalyear=config["simcalyear"]
        self.observation_set=config["observation_set"] 
        self.forecast_lead_time=config["forecast_lead_time"]
        self.split = self.year2 

        # randomly drawn weather year 
        self.chosen = np.random.choice([i for i in range(self.year1,self.year2+1)])

        # crop and soil choice 
        self.crop = Crop(config['crop'], planting_date=config['planting_date']) 
        self.soil = Soil(config['soil']) 

        self.best_profit = np.ones(self.year2-self.year1+1) * (-500) 
        self.best_yield = np.ones(self.year2-self.year1+1) * (-1) 
        self.best_water = np.ones(self.year2-self.year1+1) * (1000) 

        self.tsteps=0 

        # obsservation and action sets 
        if self.observation_set == 'default':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32) 
        elif self.observation_set == 'forecast': 
            raise NotImplementedError('forecast observation set not implemented yet') 
        
        self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)  

        self.train_curve = [] 
        self.yield_curve = []    
        self.water_curve = [] 
        self.yields = [] 

                
    def states(self):
        return dict(type='float', shape=(self.observation_space.shape[0],))
 
    def actions(self):
        return dict(type='float', num_values=self.action_space.shape[0])
        
    def reset(self):
        """
        re-initialize model and return first observation
        """ 
        
        # choose a random training year to simulate each time we reset 
        sim_year=int(np.random.choice(np.arange(self.year1,self.year2+1)))
        self.wdf = self.gendf[self.gendf.simyear==sim_year].drop('simyear',axis=1)
        self.chosen=sim_year*1 

        # irrigation cap
        self.chosen_max_irr_season =self.max_irr_season*1.
            
        # create and initialize model
        month = self.planting_month
        day=self.planting_day
        
        self.model = AquaCropModel(f'{self.simcalyear}/{month}/{day}',f'{self.simcalyear}/12/31',
                                self.wdf,self.soil,self.crop,
                                irrigation_management=IrrMngtClass(irrigation_method=5,MaxIrrSeason=self.chosen_max_irr_season),
                                initial_water_content=self.init_wc
                                )
        self.model._initialize()


        # shift the start day of simulation by specified amound
        # default 1 
        if self.dayshift:
            dayshift=np.random.randint(1,self.dayshift+1) 
            self.model.run_model(dayshift, initialize_model=False)
        
        # store irrigation events
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

        # calculate sum of daily precipitation and ETo for whole season so far
        start2 = max(0,self.model._clock_struct.time_step_counter -_init_cond.dap)
        forecastsum = self.model.weather_df.to_numpy()[start2:end,2:4].sum(axis=0).flatten()

        #  yesterday precipitation and ETo and irr
        start2 = max(0,self.model._clock_struct.time_step_counter-1)
        forecast_lag1 = self.model.weather_df.to_numpy()[start2:end,2:4].flatten()

        # calculate mean daily precipitation and ETo for next N days
        start = self.model._clock_struct.time_step_counter
        end = start+self.forecast_lead_time
        forecast2 = self.model.weather_df.to_numpy()[start:end,2:4].mean(axis=0).flatten()
        
        # state 

        # month and day
        month = (self.model._clock_struct.time_span[self.model._clock_struct.time_step_counter]).month
        day = (self.model._clock_struct.time_span[self.model._clock_struct.time_step_counter]).day
        
        # concatenate all weather variables

        if self.observation_set in ['default']:
            forecast = np.concatenate([forecast1,forecastsum,forecast_lag1]).flatten()
        
        elif self.observation_set=='forecast':
            forecast = np.concatenate([forecast1,forecastsum,forecast_lag1,forecast2,]).flatten()

        # put growth stage in one-hot format

        gs = np.clip(int(self.model._init_cond.growth_stage)-1,0,4)
        gs_1h = np.zeros(4)
        gs_1h[gs]=1

        # create observation array

        if self.observation_set in ['default','forecast']:
            obs=np.array([
                        day,
                        month,
                        dep, # root-zone depletion
                        _init_cond.dap,#days after planting
                        _init_cond.irr_net_cum, # irrigation used so far
                        _init_cond.canopy_cover, # canopy cover 
                        _init_cond.biomass, # biomass 
                        self.chosen_max_irr_season-_init_cond.irr_net_cum,
                        # _init_cond.GrowthStage,
                        
                        ]
                        +[f for f in forecast]
                        # +[ir for ir in ir_sched]
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

        new_smt=np.ones(4)*(action+1)*50 # shape 4, range 0-50, float 

        start_day = self.model._init_cond.dap 

        for i in range(self.days_to_irr): # run N days 

            if self.model._init_cond.taw>0:
                dep = self.model._init_cond.depletion/self.model._init_cond.taw
            else:
                dep=0

            gs = int(self.model._init_cond.growth_stage)-1
            if gs<0 or gs>3:
                depth=0 # no irrigation 
            else:
                if 1-dep < (new_smt[gs])/100: # if water lower than threshold 
                    depth = np.clip(self.model._init_cond.depletion,0,self.max_irr)
                else:
                    depth=0

            self.model._param_struct.IrrMngt.depth = depth
            self.irr_sched.append(self.model._param_struct.IrrMngt.depth)

            # self.model.step()
            self.model.run_model(initialize_model=False) # simulate for one day 

            # termination conditions
            # if this is the end of the year, don't run anymore  
            if self.model._clock_struct.model_is_finished is True:
                break

            now_day = self.model._init_cond.dap
            if (now_day >0) and (now_day<start_day):
                # end of season 
                break
 
 
        done = self.model._clock_struct.model_is_finished
        
        reward = 0
 
        next_obs = self.get_obs(self.model._init_cond)
 
        if done:
        
            self.tsteps+=1

            crop_yield = self.model._outputs.final_stats['Yield potential (tonne/ha)'].mean() 
            water_use = self.model._outputs.final_stats['Seasonal irrigation (mm)'].mean() 
            # calculate profit reward 
            end_reward = (
                self.CROP_PRICE * crop_yield # crop earning 
                - self.IRRIGATION_COST * water_use # water cost 
                - self.FIXED_COST # fixed cost 
            ) 
            self.reward = end_reward 
            self.yields.append(crop_yield) 

            # keep track of best rewards in each season 
            # there can be multiple rewards for a season because of possibily repeated simulations 
            # only comparisons for the same year are meaningful due to weather variability 
            if self.chosen < self.split: 
                self.best_profit[self.chosen-1] = max(self.best_profit[self.chosen-1],end_reward) 
                self.best_yield[self.chosen-1] = max(self.best_yield[self.chosen-1],crop_yield)
                self.best_water[self.chosen-1] = min(self.best_water[self.chosen-1],water_use) 
                
                self.train_curve.append(np.mean(self.best_profit)) 
                self.yield_curve.append(np.mean(self.best_yield)) 
                self.water_curve.append(np.mean(self.best_water)) 

            reward = end_reward/1000 # scale reward down by 1000 

 
        return next_obs, reward, done, dict() 
 
 
   