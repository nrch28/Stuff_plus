# Stuff+ Model

## Overview 

This is the stuff+ model I integrated into the Trackman app. It attempts to quantify the effectiveness of a pitch by predicting CSW% (called strike whiff percentage) using raw pitch characteristics.

Credit to Thomas Nestico and his articles on his stuff+ model (https://medium.com/@thomasjamesnestico/modelling-tjstuff-v3-0-10b48294c7fb) as they were very helpful resources throughout this process.

## Feature Selection and Engeneering 

I used the following raw features: release_speed, release_spin_rate, release_extension, pfx_x (horizontal break), pfx_z (vertical break), release_pos_x (horizontal release position), release_pos_z (release height), vx0 (velocity in x direction at 50 feet from home plate), vy0 (velocity in y direction at 50 feet from home plate), vz0 (velocity in z direction at 50 feet from home plate), ax (acceleration in x direction at 50 feet from home plate, ay (acceleration in y direction at 50 feet from home plate).

I also used  the following engineered features: spin_axis_sin (sin of spin axis angle), spin_axis_cos (cos of spin axis angle), velo_diff (difference between velo and primary fastball avg velo), ivb_diff (difference between induced vertical break (ivb) and primary fastball avg ivb), hb_diff (difference between horizontal break (hb) and primary fastball avg hb).


  ## Model Trainng and Evauluation

I used LGMRegressor for this model. To test the model, I ran correlations between 2023 and 2024 stats as shown in the table below (similar to what Nestico did in his article). While I'm sure there is room for improvement, this was solid enough to integrate into the app.
  

|                    | FIP_2024 | K_BB_2024 | SIERA_2024 | avg_stuff_plus_2024 |
|--------------------|----------|-----------|------------|---------------------|
| **FIP_2023**       | 0.359    | 0.285     | 0.430      | 0.229               |
| **K_BB_2023**      | 0.288    | 0.455     | 0.323      | 0.171               |
| **SIERA_2023**     | 0.403    | 0.290     | 0.517      | 0.252               |
| **avg_stuff_plus_2023** | 0.216    | 0.150     | 0.274      | 0.673               |







  
