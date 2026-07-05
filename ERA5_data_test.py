import cdsapi
import xarray as xr

def retrive_era5(year, month, day, time, lat, lon):
    c = cdsapi.Client() #gets api credentials 

    result = c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind", #east west wind
                "10m_v_component_of_wind" #north south wind
            ],
            "year": year,
            "month": month,
            "day": day,
            "time": time,
            "data_format": "netcdf"
        }, "era5_data.nc"
    )


    ds = xr.open_dataset("era5_data.nc") 

    point = ds.sel(
        latitude=lat,
        longitude=lon,
        method="nearest"
    )
    


    u = point["u10"].values.item()
    v = point["v10"].values.item()
  
    return u,v