from aquacropeto import * 
import pandas as pd 

def calc_eto_faopm(file,
                    year,
                    latitude,
                    altitude=0,
                    generated=True,
                    order=["simyear","jday","minTemp","maxTemp","precip","rad"]):
    """ 
    Uses FAO-PM to calculate reference evapotranspiration for LARS generated and baseline input data. 
    """

    df = pd.read_csv(file,delim_whitespace=True,header=None)

    if generated:
        df.columns=order
        df["tdelta"]=pd.to_timedelta(df.jday,unit='D')
        df["date"]=pd.to_datetime(f'{year-1}/12/31')+df["tdelta"]


        net_sw = net_in_sol_rad(df.rad)
        ext_rad = et_rad(deg2rad(latitude),sol_dec(df.jday),sunset_hour_angle(deg2rad(latitude),sol_dec(df.jday)),inv_rel_dist_earth_sun(df.jday))
        cl_sky_rad = cs_rad(altitude,ext_rad)
        net_lw_rad = net_out_lw_rad(df.minTemp,df.maxTemp,df.rad,cl_sky_rad,avp_from_tmin(df.minTemp))
        net_radiation = net_rad(net_sw,net_lw_rad)

        av_temp=(df.minTemp+df.maxTemp)*0.5

        ws=2
        svp = mean_svp(df.minTemp,df.maxTemp)
        avp = avp_from_tmin(df.minTemp)
        delta = delta_svp(av_temp)
        psy=psy_const(atm_pressure(altitude))
        faopm = fao56_penman_monteith(net_radiation,av_temp+273,ws,svp,avp,delta,psy)

        df["eto"] = faopm
        df.eto=df.eto.clip(0.1)
        df=df[["simyear","minTemp","maxTemp","precip","eto",'date']]
        df.columns=["simyear","MinTemp","MaxTemp","Precipitation","ReferenceET","Date"]

    return df

