import pandas as pd 
from aquacrop import IrrigationManagement 
from aquacrop import InitialWaterContent, Crop, Soil, AquaCropModel 
import numpy as np 
from utils import configs 

config = configs['nebraska_maize_base']

df = config['gendf'] 
dates = df[df['simyear']==1]['Date'] 

best_mean_yields = 0 

for irr_amount in range(10, 100, 10):  
    print(f'Running for {irr_amount} mm') 
     
    durations = [] 
    yields = [] 

    for year in range(700, 1000):
        simyear = year + 1 
        weather_df = config['gendf'][config['gendf']['simyear'] == simyear].drop('simyear',axis=1) 

        schedule_df = pd.DataFrame({'Date':dates, 'Depth':irr_amount})

        irrigate_schedule = IrrigationManagement(irrigation_method=3,schedule=schedule_df) 
        crop = Crop(config['crop'], planting_date=config['planting_date']) 
        soil = Soil(config['soil']) 
        init_wc = InitialWaterContent(wc_type='Pct',value=[70]) 

        simcalyear = config['simcalyear'] 
        month = int(config['planting_date'].split('/')[0]) 
        day = int(config['planting_date'].split('/')[1])
            
        model = AquaCropModel(
            f'{simcalyear}/{month}/{day}',
            f'{simcalyear}/12/31',
            weather_df,
            soil=soil,
            crop=crop, 
            initial_water_content=init_wc,
            irrigation_management=irrigate_schedule 
        ) 

        model.run_model(till_termination=True) 
        print(model._outputs.final_stats) 
        durations.append(float(model._outputs.final_stats['Harvest Date (Step)'].iloc[0])) 
        yields.append(float(model._outputs.final_stats['Yield potential (tonne/ha)'].iloc[0])) 

    best_mean_yields = max(best_mean_yields, np.mean(yields)) 

print(f'Best mean yield for constant irrigation: {best_mean_yields}')  
