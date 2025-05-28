This project performs crop-yield predictions based on machine learning techniques, where meteorological and EO data are used as predictor, while crop yield statistics as target variables. In particular, the project is carried out through the following tasks.

### Task 1: Crop-yield prediction across European countries using meteorological and pesticide data from 2000 to 2021.

The dataset is collected from the following sources:

- Annual Yield (yield.csv) and Pesticides (pesticides.csv) are collected from Food and Agriculture Organization (FAO): https://www.fao.org/faostat/en/#data

- Annual Rainfall (rainfall.csv) and Avg. Temperature (avg_temp_1950-2014.csv and avg_temp_2015-2100.csv) are collected from World Data Bank: https://data.worldbank.org/

- The data is preprocessed and merged to achieve the final_data.csv file.

### Task 2: Crop-yield prediction in Luxembourg using weather + pesticide data and the Normalized Difference Vegetation Index (NDVI) time series

Dataset used for this task:

- NDVI data: The NDVI index is calculated using the reflectance values of bands B4 and B8, collected from the Sentinel-2 satellites (A and B) via Google Earth Engine (GEE).
Specifically, the Luxembourg area is divided into 16 smaller grids to facilitate data collection.
Monthly NDVI data for the years 2017-2024 is collected by taking the median value across all grids.
Finally, the annual NDVI parameters (e.g., mean, max, sum) are determined using the monthly data.
Here, a resolution of 10m and a maximum allowable cloud coverage of 20% are applied.

- Weather data: Using the Luxembourg city's coordinates (49.8153°N and 6.1296°E), the daily weather data (e.g., precipitation, avg. temp, humidity, cloud amount, etc.) from the NASA POWER for the last 25 years (2000-2024) are collected.
- The annual weather data is then aggregated from this dataset.

- Pesticides and Yield statistics from 2000 to 2022 are collected from FAO: https://www.fao.org/faostat/en/#data.
