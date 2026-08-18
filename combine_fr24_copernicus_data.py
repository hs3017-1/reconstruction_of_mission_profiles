import pandas as pd # import Pandas for working with large amounts of data
import numpy as np
import os # import Os to reference files on disk
import cdsapi # import Copernicus dataset API library
import xarray as xr # imports xarray to handle the 4D structure that copernicus data has (lat, long, pres, time)

fr24csv = "B67_030226.csv"
df = pd.read_csv(fr24csv) # read the csv file into a Pandas dataframe object

df[["Latitude", "Longitude"]] = df["Position"].str.split(",", expand=True) # splits the position column (at the comma) and expands into new latitude and longitude columns seperately, to better match the format of the copernicus data
df = df.drop(columns=["Position"]) # removes the old position column as it is no longer needed, and reassigns the updated dataframe
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce") # convert the string value to a number
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce") # convert the string value to a number
df["Date/Time"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True) # convert the timestamp (which is in seconds) to a datetime object as used in Copernicus

# define ICAO Standard Atmosphere raw constants for barometric formula
p0 = 1013.25
T0 = 288.15
L = 0.0065
g = 9.80665
R = 287.05287
exponent = g / (L * R) # calculate exponent value
scaling = L / T0 # calculate scaling factor      
df["Altitude (m)"] = df["Altitude"] * 0.3048 # convert feet to metres by multiplying by 0.3048
df["Pressure (hPa)"] = p0 * (1 - scaling * df["Altitude (m)"]) ** exponent # calculates pressure in hectopascals by using the barometric formula
df["Pressure (hPa)"] = df["Pressure (hPa)"].clip(upper=1000) # clip any values above 1000 as Copernicus levels end at 1000 otherwise get NaNs

# extracts the individual days, months and years that the flight used in the datetime column
# .strftime converts the y,m,d to string
# .unique removes duplicates (if 5000 entries were all on the same day, just reduce it to one)
# list() converts the array to a list so in right format for cdsapi
years = list(df["Date/Time"].dt.strftime("%Y").unique()) 
months = list(df["Date/Time"].dt.strftime("%m").unique())
days = list(df["Date/Time"].dt.strftime("%d").unique())
print(f"Dates in FR24 dataset:{"\n"}years: {years}{"\n"}months: {months}{"\n"}days: {days}")

min_hour = df["Date/Time"].min().hour # finds the earliest datetime in df and extracts just the hour
max_hour = df["Date/Time"].max().hour # finds the latest datetime in df and extracts just the hour
# loops through each integer in hour range created and converts it to right format for copernicus
# the ranges (and the use of +1) make sure that all hours are included (as python ranges end one before stop)
hours = [f"{h:02d}:00" for h in range(min_hour, max_hour + 1)] 
print(f"Hours in FR24 dataset:  {hours}")

# creates a sort of rectangle around the flight path, so only areas the plane moved in are taken from copernicus
# converts the coordinates to python floats
# also rounds to 2 d.p to match the correct format
north = round(float(df["Latitude"].max() + 1.0), 2) # finds the furthest north point that the aircraft reaches
south = round(float(df["Latitude"].min() - 1.0), 2) # finds the furthest south point that the aircraft reaches
west = round(float(df["Longitude"].min() - 1.0), 2) # finds the furthest west point that the aircraft reaches
east = round(float(df["Longitude"].max() + 1.0), 2) # finds the furthest east point that the aircraft reaches
# the +/-1 makes the rectangle a bit bigger so my interpolation doesn't get messed up at the edges of the rectangles
area_bounds = [north, west, south, east] #organises the 4 coordinates into a list in that format for copernicus
print(f"Bounding rectangle: {area_bounds}") # Order required by Copernicus API
print(f"Latitudes covered:  {area_bounds[2]}° to {area_bounds[0]}°") # South to North
print(f"Longitudes covered: {area_bounds[1]}° to {area_bounds[3]}°") # West to East
#prints output to check where this rectangle lies

all_era5_levels = np.arange(100,1025,25) # ERA5 stores 27 different 'slices' of the atmosphere for different pressure levels depending on altitude
min_p = df["Pressure (hPa)"].min() # finds minimum pressure in df (highest altitude, cruising)
max_p = df["Pressure (hPa)"].max() # finds maximum pressure in df (lowest altitude, takeoff or landing)
needed_levels = [
    str(level) #converts numbers into strings for cdsapi download
    for level in all_era5_levels #loops through all levels above, only keeps the ones in flight range
    if (min_p - 50) <= level <= (max_p + 50) # creates vertical space above where the aircarft flies for interpolation later
]
#checks which ERA5 levels fall within the vertcial slice needed
# Print verification for Pressure Levels
print(f"min pressure: {min_p:.2f} hPa")
print(f"max pressure: {max_p:.2f} hPa")
print(f"ERA5 levels needed: {needed_levels}")

filename = "dynamic_copernicus_data.nc"
cdsapi_key = '15bdf592-e944-4b2e-a3b2-c67f2c53d972' 
fetch_data = False

if fetch_data == False:
        print(f"Skipping CDS API download. Using existing file '{filename}'.")

if fetch_data == True:
    try:
        print("Connecting to CDS API and requesting data...")
        os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api'
        os.environ['CDSAPI_KEY'] = cdsapi_key
        c = cdsapi.Client()
        print("CDS API client initialised")
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": [
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "relative_humidity",
                    "geopotential",
                ],
                "pressure_level": needed_levels,
                "year": years,
                "month": months,
                "day": days,
                "time": hours,
                "area": area_bounds,
            },
            filename,
        )
        print(f"Download complete: Successfully saved to '{filename}'.")
        
    except Exception as e:
        print(f"An error occurred while retrieving data from CDS API: {e}")

ds = xr.open_mfdataset("dynamic_copernicus_data.nc", chunks={}, engine="netcdf4") #opens a multi-file dataset
#chunks enables 'lazy loading' so only reads times, coordinates etc to save memory space
print("Copernicus dataset loaded")
print(f"Dataset dimensions: {dict(ds.sizes)}") # tells us how many different grid points, pressure levels and hours the dataset covers

flight_idx = xr.DataArray(df.index, dims="flight_point")
# extracts row labels, and turns every row in the table into a single point along a line
# allows xarray to look up the right weather values for every row at once later
print(f"created 'flight_point' index with {len(flight_idx)} rows")
# tells me how many flight points there are

target_time = xr.DataArray(df["Date/Time"], dims="flight_point", coords={"flight_point": flight_idx})
target_lat = xr.DataArray(df["Latitude"], dims="flight_point", coords={"flight_point": flight_idx})
target_lon = xr.DataArray(df["Longitude"], dims="flight_point", coords={"flight_point": flight_idx})
target_pressure = xr.DataArray(df["Pressure (hPa)"], dims="flight_point", coords={"flight_point": flight_idx})
# df["column_name"] passes each column from df into xr.dataarray()
# dims="flight_point" tells xarray that each of 4 coordinate lists is 1D and has same dimension name (flight_point)
# coords={"flight_point": flight_idx} ensures the row numbers stay bound to the coordinate values 
# i.e all row 100s time, lat, long & pres stay together as flight point 100
print(f"target_time: {target_time.size}")
print(f"target_lat: {target_lat.size}")
print(f"target_lon: {target_lon.size}")
print(f"target_pressure: {target_pressure.size}")
print(f"total flight_points ready for lookup: {len(target_time)}") #states how many flight points there are and that all 4 coordinates are ready for each flight point

''' Using .interp instead of .sel means that the code calculates a mathematical average across all dimensions at once to give 
the best estimate of values between grid points, rather than 'snapping' to the nearest node which can skew the data'''
extracted_ds = ds.interp( 
    valid_time=target_time, # calculates temporal weights (how far into each hour the specific time is and blends the weather values based on that exact proportion
    latitude=target_lat, # uses bilinear interpolation to find where the plane lies inside its 0.25 x 0.25 degree grid 
    longitude=target_lon,
    pressure_level=target_pressure, # standard pressure levels go up in 25s, so like time the weights are calculated and an estimation of the true altitude is made
    method="linear" # tells xarray to use multilinear interpolation (quadrilinear as I am using 4 variables)
) # calculates a weighted average across 16 surrounding grid points for each variable and determines exact weather conditions experienced by the aircraft at any specific point in time
print("4D multilinear interpolation setup complete")
# xarray and scipy calculated continuous weather values
print(f"processed flight coordinates: {dict(extracted_ds.sizes)}")
# all of the properties of the flight points have been correctly interpolated
#ds.close() # close the xr.ds object to clear RAM and avoid filesystem locking the file

weather_df = extracted_ds.compute().to_dataframe().reset_index(drop=True)
# computes and does the interpolation maths, and brings the actual weather numbers into RAM
# converts the xr.ds object into a pandas dataframe
# flight point would have been made the new index, but this is removed for a cleaner table that matches the original
final_df = df.join(weather_df) # merges the original FR24 dataframe with the new weather_df containing the atmospheric variables from Copernicus
print(f"merged weather data with flight dataframe")
print(f"final dataframe shape: {final_df.shape}")
print(f"new columns: {[col for col in weather_df.columns if col in final_df.columns]}")

final_df["Temperature (C)"] = final_df["t"] - 273.15 # adds a new column converting temperature from kelvin to celcius
final_df["Wind Speed"] = np.sqrt(final_df["u"] ** 2 + final_df["v"] ** 2) # calculates magnitude of the wind speed using the u and v components
final_df["Wind Direction"] = (np.degrees(np.arctan2(-final_df["u"], -final_df["v"]))) % 360
# the components are negative as vectors usually point in the direction they are going towards, however wind direction is the direction in which the air mass came from, so the vectors are reversed
# the % 360 automatically makes negative angles into positive ones, as arctan returns a value from -pi to pi, whereas meteorologists use a standard 0-360 degree compass
ground_speed_ms = final_df["Speed"] * 0.514444 # convert from knots (as given by flightradar) to m/s, to match ERA5 wind vectors
track_rad = np.radians(final_df["Direction"]) # convert to radians to be able to use trigonometric functions in python
v_aircraft_u = ground_speed_ms * np.sin(track_rad)
v_aircraft_v = ground_speed_ms * np.cos(track_rad)
# split the single ground speed into its 2 components, sin and cos are swapped around as aviation compass directions start from north rather than east like in mathematics
v_air_u = v_aircraft_u - final_df["u"] # eastward velocity relative to the air mass (ground u-wind u)
v_air_v = v_aircraft_v - final_df["v"] # northward velocity relative to the air mass (ground v-wind v)
final_df["True Air Speed"] = np.sqrt(v_air_u**2 + v_air_v**2) # true airspeed vector=(ground speed vector - wind vector) (rearranged vector traingle rules)
# use pythagorean theorem to combine 2 vector components back into single scalar speed value (magnitude)

gamma = 1.4 # heat capacity ratio
R = 287.058 # specific gas constant
speed_of_sound = np.sqrt(gamma * R * final_df["t"]) # calculates the speed of sound using the newton-laplace equation. t=absolute temperature (kelvin)
final_df["Mach Number"] = final_df["True Air Speed"] / speed_of_sound # mach number = true air speed/speed of sound
final_df["Geopotential"] = final_df["z"]

columns_to_drop = [
    "number",  # The column that stays at 0
    "latitude",  # Duplicate latitude from xarray (if named with suffix)
    "longitude",  # Duplicate longitude from xarray
    "pressure_level",  # Duplicate pressure level
    "valid_time",  # Duplicate time
    "z"
]

# Only drop columns if they actually exist in final_df to avoid errors
final_df = final_df.drop(
    columns=[col for col in columns_to_drop if col in final_df.columns]
)

# 2. Rename weather variables to clear, descriptive titles
final_df = final_df.rename(
    columns={
        "t": "Air Temperature (K)",  # Air Temperature (Kelvin)
        "u": "U-wind component (m/s)",  # U-wind component (m/s)
        "v": " V-wind component (m/s)",  # V-wind component (m/s)
        "r": "Relative Humidity (%)",  # Relative Humidity (%)
    }
)

csv_columns = ['Timestamp', 'UTC', 'Callsign', 'Altitude', 'Speed', 'Direction',
       'Latitude', 'Longitude', 'Date/Time', 'Altitude (m)', 'Pressure (hPa)',
       'Air Temperature (K)', 'U-wind component (m/s)',
       ' V-wind component (m/s)', 'Relative Humidity (%)', 'Temperature (C)',
       'Wind Speed', 'Wind Direction', 'True Air Speed', 'Mach Number',
       'Geopotential']
output_csv = fr24csv.replace(".csv","_output.csv")
final_df.to_csv(output_csv, index=False, columns=csv_columns)
print(f"Saved to {output_csv}")
# saves selected columns from final_df to csv
