import ee
ee.Initialize(project="methaneplume")

# permian basin aoi
AOI = ee.Geometry.Rectangle([
    -105.219880, 29.462935,
    -100.036107, 34.021515
])

start_date = ee.Date("2024-06-17")
end_date   = ee.Date("2024-06-30")


# datasets
ch4_ic = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4") \
    .select("CH4_column_volume_mixing_ratio_dry_air") \
    .filterBounds(AOI)

era5_ic = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
    .filterBounds(AOI) \
    .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])

ogim = ee.FeatureCollection("EDF/OGIM/current").filterBounds(AOI)


# convert OGIM to raster mask (fixx)
ogim_mask = ee.Image(0).byte().paint(
    featureCollection=ogim,
    color=1
).rename("ogim")


# methane enhancement (focal_median on image)
def add_enhancement(img):
    background = img.focal_median(
        radius=50000,
        units="meters"
    )
    enh = img.subtract(background)
    return enh.rename("ch4_enh")


# build per day image (ee safe)
def build_day(d):

    d = ee.Date(d)

    # methane daily composite
    ch4 = ch4_ic.filterDate(d, d.advance(1, "day")).mean()

    # if empty day fallback
    ch4 = ee.Image(0).where(ch4.mask().Not(), ch4)

    ch4 = add_enhancement(ch4)

    # wind daily composite
    wind = era5_ic.filterDate(d, d.advance(1, "day")).mean()

    u = wind.select("u_component_of_wind_10m").rename("u")
    v = wind.select("v_component_of_wind_10m").rename("v")

    # stack everything correctly
    img = ee.Image.cat([ch4, u, v, ogim_mask])

    return img.set({
        "system:time_start": d.millis(),
        "date": d.format("YYYY-MM-dd")
    })


# date sequence (fix ltr)
n_days = end_date.difference(start_date, "day")

dates = ee.List.sequence(0, n_days.subtract(1)).map(
    lambda i: start_date.advance(i, "day")
)

dataset = ee.ImageCollection(dates.map(build_day))

print("dataset size:", dataset.size().getInfo())
print(dataset.first().bandNames().getInfo())
