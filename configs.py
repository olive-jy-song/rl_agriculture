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
    year2=70, 
    crop='Maize', # crop type 
    planting_date='05/01',
    soil='SiltClayLoam', # soil type 
    dayshift=1, # maximum number of days to simulate at start of season (ramdomly drawn)
    include_rain=True, # we'd like to include precipitation in the simulation 
    days_to_irr=7, # decision period, or the frequency of steps 
    max_irr=25, # maximum irrigation depth per event
    max_irr_season=10_000, # maximum irrigation appl for season
    init_wc=InitialWaterContent(wc_type='Pct',value=[70]), # initial water content
    crop_price=180., # $/TONNE
    irrigation_cost = 1.,# $/HA-MM
    fixed_cost = 1728, # the fix cost of irrigation per year 
    best=np.ones(1000)*-1000, # a record of the best rewards among simulated per year 
    observation_set='default', # the choise of state definition 
    action_set='smt4',
    forecast_lead_time=7, # if we want 
    CO2conc=363.8, # CO2 concentration in ppm 
    simcalyear=1995, 
) 



