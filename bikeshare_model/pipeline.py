import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer, WeathersitImputer, Mapper, CustomOutlierHandler, WeekdayOneHotEncoder


bikeshare_rental_pipe=Pipeline([
    
    ("WeekdayImputer", WeekdayImputer()),
    ("WeathersitImputer", WeathersitImputer()), 
    ("Map_yr", Mapper(config.model_config_.yr_var, config.model_config_.yr_mapping)),
    ("Map_mnth", Mapper(config.model_config_.mnth_var, config.model_config_.yr_mapping)),
    ("Map_season", Mapper(config.model_config_.season_var, config.model_config_.season_mapping)),
    ("Map_weathersit", Mapper(config.model_config_.weathersit_var, config.model_config_.weather_mapping)),
    ("Map_Holiday", Mapper(config.model_config_.holiday_var, config.model_config_.holiday_mapping)),
    ("Map_Workingday", Mapper(config.model_config_.workingday_var, config.model_config_.workingday_mapping)),
    ("Map_Hr", Mapper(config.model_config_.hr_var, config.model_config_.hour_mapping)),
    ('Handle Outlier', CustomOutlierHandler()),
    ('Weekday encoder', WeekdayOneHotEncoder()),
    # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestClassifier(n_estimators=config.model_config_.n_estimators, 
                                         max_depth=config.model_config_.max_depth, 
                                         max_features=config.model_config_.max_features,
                                         random_state=config.model_config_.random_state))
          
     ])
