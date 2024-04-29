'''
This file contains all the base configurations for the environments we use in the project, 
including the weather data, crop type, soil type, and other important parameters for a certain region. 
'''

from eto import calc_eto_faopm 
from aquacrop import InitialWaterContent 
import numpy as np 


# Default configuration for the Nebraska Maize  

nebraska_gendf = calc_eto_faopm(
    'data/nebraska.dat',
    year=1995,
    latitude=40.4,
    altitude=1072
) 

nebraska_maize_config = dict(
    name='nebraska_maize',
    gendf=nebraska_gendf, # generated and processed weather dataframe
    year1=1, # years are deafault train years, from 1 - 70 
    year2=700, 
    crop='Maize', # crop type 
    planting_date='05/01',
    soil='SiltClayLoam', # soil type 
    dayshift=1, # maximum number of days to simulate at start of season (ramdomly drawn)
    days_to_irr=7, # decision period, or the frequency of steps 
    max_irr=25, # maximum irrigation depth per event
    max_irr_season=10000, # maximum irrigation appl for season
    init_wc=InitialWaterContent(wc_type='Pct',value=[70]), # initial water content
    crop_price=180., # $/TONNE
    irrigation_cost = 1.,# $/HA-MM
    fixed_cost = 1728, # the fix cost of irrigation per year 
    best=np.ones(1000)*-1000, # a record of the best rewards among simulated per year 
    observation_set='default', # the choise of state definition 
    forecast_lead_time=7, # if we want to use forecast, how many days ahead 
    simcalyear=1995, 
    reward_scale=(1,1), # scaling of yield, water 
) 

# the following configurations are variations of days to irrigate 
nebraska_1day_config = nebraska_maize_config.copy() 
nebraska_1day_config['days_to_irr'] = 1 

nebraska_3day_config = nebraska_maize_config.copy() 
nebraska_3day_config['days_to_irr'] = 3 

nebraska_5day_config = nebraska_maize_config.copy() 
nebraska_5day_config['days_to_irr'] = 5 

nebraska_14day_config = nebraska_maize_config.copy() 
nebraska_14day_config['days_to_irr'] = 14 

# the following configurations are variations of water availability 
nebraska_scarcewater_config = nebraska_maize_config.copy() 
nebraska_scarcewater_config['max_irr'] = 5 
nebraska_scarcewater_config['max_irr_season'] = 2000 

nebraska_abunwater_config = nebraska_maize_config.copy() 
nebraska_abunwater_config['max_irr'] = 100 
nebraska_abunwater_config['max_irr_season'] = 40000 

# with more control over action space 
nebraska_control_config = nebraska_maize_config.copy() 
nebraska_control_config['max_irr'] = 100 
nebraska_control_config['max_irr_season'] = 1000 

nebraska_morecontrol_config = nebraska_maize_config.copy() 
nebraska_morecontrol_config['max_irr'] = 100 
nebraska_morecontrol_config['max_irr_season'] = 500 

nebraska_control400_config = nebraska_maize_config.copy() 
nebraska_control400_config['max_irr'] = 100 
nebraska_control400_config['max_irr_season'] = 400 

nebraska_control750_config = nebraska_maize_config.copy() 
nebraska_control750_config['max_irr'] = 100
nebraska_control750_config['max_irr_season'] = 750 

nebraska_best = nebraska_morecontrol_config.copy() 
# the following configurations are for reward scaling of yield & water 
nebraska_scale1_config = nebraska_best.copy() 
nebraska_scale1_config['reward_scale'] = (1, 0.1) 

nebraska_scale2_config = nebraska_best.copy() 
nebraska_scale2_config['reward_scale'] = (1, 0) # no reward cost considered 

nebraska_scale3_config = nebraska_best.copy() 
nebraska_scale3_config['reward_scale'] = (0.7, 1) 

# the following configurations are used for using weather forecast as states 
nebraska_forecast_config = nebraska_best.copy() 
nebraska_forecast_config['observation_set'] = 'forecast' 
nebraska_forecast_config['forecast_lead_time'] = 7 

nebraska_forecast2_config = nebraska_best.copy() 
nebraska_forecast2_config['observation_set'] = 'forecast' 
nebraska_forecast2_config['forecast_lead_time'] = 7 
nebraska_forecast2_config['max_irr_season'] = 1000 

# use max temperature as state 
nebraska_maxtemp_config = nebraska_best.copy() 
nebraska_maxtemp_config['observation_set'] = 'temperature' 
nebraska_maxtemp_config['max_irr_season'] = 750 

# use no eto as state 
nebraska_noeto_config = nebraska_best.copy()
nebraska_noeto_config['observation_set'] = 'noeto' 





