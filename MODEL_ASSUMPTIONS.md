# Sensor-model assumptions

## Sun sensors

The simulation uses one coarse cosine-response Sun sensor on each of the six
CubeSat faces. This matches common CubeSat coarse Sun-sensor architectures.
NASA's 2024 Small Spacecraft Technology State of the Art report lists the
Space Micro CSS-01/CSS-02 as cosine Sun sensors with a 120-degree full-cone
field of view and approximately 5-degree 3-sigma accuracy.

For a cosine sensor, around 45-degree incidence, a 5-degree 3-sigma angular
uncertainty corresponds approximately to a normalized one-sigma signal
uncertainty of 0.02. Therefore the baseline model uses:

- normalized random noise standard deviation: 0.02

The per-face gain and offset differences are retained as residual calibration
mismatch assumptions for sensitivity analysis. They are not vendor datasheet
specifications and should be described as such in the report.

A second useful reference is Tensor Tech's CSS100 coarse Sun sensor, which is
photodiode-based, has a +/-60-degree field of view, and explicitly recommends
multiple sensors on different CubeSat faces to work with a magnetometer for
rough attitude determination.

## Magnetometer

The magnetometer model is based on a PNI RM3100-class sensor. Published
nanosatellite work using the RM3100 reports representative worst-axis values:

- noise: 17 nT
- bias: 440 nT
- scale-factor error: 4885 ppm

NOAA states that Earth's surface magnetic-field magnitude is roughly
25,000-65,000 nT. The simulation uses 50,000 nT (50 microtesla) as a convenient
representative normalization value.

This gives normalized baseline values of approximately:

- random noise standard deviation: 17 / 50,000 = 0.00034
- worst-axis bias: 440 / 50,000 = 0.0088
- scale-factor error: 4885 ppm = 0.004885

## Magnetic disturbance study

Spacecraft-generated magnetic interference can be much larger than the sensor's
intrinsic noise. Published CubeSat measurements report magnetorquer-generated
magnetic noise exceeding 7,500 nT peak-to-peak. The representative fault case
therefore uses a 7.5 microtesla disturbance magnitude, while the sweep varies
the imposed disturbance systematically.

## Sources to cite in the report

- NASA, Small Spacecraft Technology State of the Art 2024, GNC chapter,
  Sun-sensor comparison table.
- PNI RM3100 technical specifications / nanosatellite sensor-architecture
  literature reporting RM3100 noise, bias, and scale-factor error.
- NOAA NCEI geomagnetism FAQ for representative Earth magnetic-field strength.
- Spacecraft-noise literature reporting CubeSat onboard magnetic interference.
