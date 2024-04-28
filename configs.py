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








