'''
'''

# from aquacrop.classes import *
# from aquacrop.core import *
from aquacrop import InitialWaterContent as InitWCClass, Crop as CropClass, Soil as SoilClass, AquaCropModel 
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
        self.include_rain=config["include_rain"]
        self.max_irr=config["max_irr"]
        self.init_wc = config["init_wc"]
        self.CROP_PRICE=config["crop_price"]
        self.IRRIGATION_COST=config["irrigation_cost"] 
        self.FIXED_COST = config["fixed_cost"]
        self.planting_month = int(config['planting_date'].split('/')[0])
        self.planting_day = int(config['planting_date'].split('/')[1])
        self.max_irr_season=config['max_irr_season']
        self.name=config["name"]
        self.best=config["best"]*1
        self.simcalyear=config["simcalyear"]
        self.CO2conc=config["CO2conc"]
        self.observation_set=config["observation_set"]
        self.action_set=config["action_set"]
        self.forecast_lead_time=config["forecast_lead_time"]

        # randomly drawn weather year 
        self.chosen = np.random.choice([i for i in range(self.year1,self.year2+1)])

        # crop and soil choice

        crop = config['crop']        
        if isinstance(crop,str):
            self.crop = CropClass(crop, planting_date=config['planting_date'])
        else:
            assert isinstance(crop,CropClass), "crop needs to be 'str' or 'CropClass'"
            self.crop=crop

        soil = config['soil']
        if isinstance(soil,str):
            self.soil = SoilClass(soil)
        else:
            assert isinstance(soil,SoilClass), "soil needs to be 'str' or 'SoilClass'"
            self.soil=soil
     
     
        self.tsteps=0 

        # obsservation and action sets

        if self.observation_set in ['default',]:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        
        if self.action_set=='smt4':
            self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)

        elif self.action_set=='depth':
            self.action_space = spaces.Box(low=-5., high=self.max_irr+5., shape=(1,), dtype=np.float32)
    
        elif self.action_set=='depth_discreet':
            self.action_depths=[0,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25]
            self.action_space = spaces.Discrete(len(self.action_depths))    

        elif self.action_set=='binary':
            self.action_depths=[0,self.max_irr]
            self.action_space = spaces.Discrete(len(self.action_depths))    

                
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

        if isinstance(self.max_irr_season, list):
            if isinstance(self.max_irr_season[0], list):
                self.chosen_max_irr_season = float(np.random.choice(self.max_irr_season[0]))
            else:
                self.chosen_max_irr_season = float(np.random.randint(self.max_irr_season[0],self.max_irr_season[1]))
        else:
            self.chosen_max_irr_season =self.max_irr_season*1.
            

        # create and initialize model

        month = self.planting_month
        day=self.planting_day

        self.model = AquaCropModel(f'{self.simcalyear}/{month}/{day}',f'{self.simcalyear}/12/31',
                                self.wdf,self.soil,self.crop,
                                irrigation_management=IrrMngtClass(irrigation_method=5,MaxIrrSeason=self.chosen_max_irr_season),\
                                # co2_concentration=self.CO2conc,\
                                initial_water_content=self.init_wc
                                )
        self.model._initialize()

        # remove rainfall from weather if requested

        if not self.include_rain:
            self.model.weather_df[:,2]=0

        # shift the start day of simulation by specified amound
        # default 1

        if self.dayshift:
            dayshift=np.random.randint(1,self.dayshift+1)
            # self.model.step(dayshift)
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
        # if _init_cond.taw>0:
        #     dep = _init_cond.Depletion/_init_cond.taw
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


        else:
            assert 1==2, 'no obs set'
        
        return obs
        
    def step(self,action):
        """
        Take in agents action choice

        apply irrigation depth on following day

        simulate N days until next irrigation decision point

        calculate and return profit at end of season

        """

        # if choosing discrete depths

        if self.action_set in ['depth_discreet']:

            depth = self.action_depths[int(action)]

            self.model._param_struct.IrrMngt.depth = depth

        # if making banry yes/no irrigation decisions

        elif self.action_set in ['binary']:

            if action == 1:
                depth = self.max_irr #apply max irr
            else:
                depth=0
            
            self.model._param_struct.IrrMngt.depth = depth

        # if spefiying depth from continuous range

        elif self.action_set in ['depth']:

            depth=np.clip(action[0],0,self.max_irr)
            self.model._param_struct.IrrMngt.depth = depth

        # if deciding on soil-moisture targets

        elif self.action_set=='smt4':

            new_smt=np.ones(4)*(action+1)*50 # shape 4, range 0-50, float 


        start_day = self.model._init_cond.dap 

        for i in range(self.days_to_irr): # run N days 

            # apply depth next day, and no more events till next decision
            
            if self.action_set in ['depth_discreet','binary','depth']:
                self.irr_sched.append(self.model._param_struct.IrrMngt.depth)
                # self.model.step()
                self.model.run_model(initialize_model=False) 
                self.model._param_struct.IrrMngt.depth = 0
            
            # if specifying soil-moisture target, 
            # irrigate if root zone soil moisture content
            # drops below threshold

            elif self.action_set=='smt4':

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

            # calculate profit reward 
            end_reward = (self.CROP_PRICE*self.model._outputs.final_stats['Yield potential (tonne/ha)'].mean() # crop earning 
                        - self.IRRIGATION_COST*self.model._outputs.final_stats['Seasonal irrigation (mm)'].mean() # water cost 
                        - self.FIXED_COST ) 
            self.reward = end_reward 

            # keep track of best rewards in each season 
            # there can be multiple rewards for a season because of possibily repeated simulations 
            # only comparisons for the same year are meaningful due to weather variability 
            self.best[self.chosen-1] = max(self.best[self.chosen-1],end_reward) 

            reward = end_reward/1000 # scale reward down by 1000 
 
        return next_obs, reward, done, dict() 
 
 
   