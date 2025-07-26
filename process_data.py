import pandas as pd
import numpy as np
import json
import glob

# A number of fields are stored as json, but can contain characters that will cause json parsing to fail (for example, a creator name with a quotation mark or comma). We don't need the values of any of those fields, so replace the offending characters so that we can parse the json to get what we DO need.
KS_data = (pd.read_csv("Kickstarter_2025-06-12/Kickstarter.csv").
        replace(',,','', regex=True).
        replace('""', '"', regex=True).
        replace(r'([^{:,])"([^:,}])', r'\1\2', regex=True).
        replace(r'(",)([^"])', r'\2', regex=True).
        replace(r'\\', r' ', regex=True).
        replace([r"\\t|\\n|\\r", "\t|\n|\r"], value=[" "," "], regex=True))

# Loop through all the separate files and combine data (I'm not sure why the data provider stored the data in so many separate CSVs...)
for f in glob.glob("Kickstarter_2025-06-12" + "/Kickstarter*.csv"):
  data = (pd.read_csv(f).
        replace(',,','', regex=True).
        replace('""', '"', regex=True).
        replace(r'([^{:,])"([^:,}])', r'\1\2', regex=True).
        replace(r'(",)([^"])', r'\2', regex=True).
        replace(r'\\', r' ', regex=True).
        replace([r"\\t|\\n|\\r", "\t|\n|\r"], value=[" "," "], regex=True))
  KS_data = pd.concat([KS_data, data], axis=0) 

KS_1 = KS_data.copy()
# Remove campaigns that are currently live, since they have no outcome
KS_1 = KS_1[KS_1["state"].isin(["successful", "failed", "canceled"])]
KS_1.dropna(subset = ['deadline', 'created_at', 'launched_at', 'creator', 'location', 'urls', 'usd_pledged'], inplace=True)
KS_1.drop_duplicates(subset=['id'],inplace=True)

# More data sanitization to prevent the json parsing from failing
KS_1 = KS_1.replace([r"\\t|\\n|\\r", "\t|\n|\r"], value=[" "," "], regex=True).reset_index(drop=True)

# Convert columns to usable formats
KS_1['deadline'] = pd.to_datetime(KS_1['deadline'], unit='s')
# Some campaigns used other currencies, and we need to convert their funding goal to USD for comparability
KS_1['usd_goal'] = KS_1['goal'] * KS_1['static_usd_rate']
KS_1['country'] = KS_1['country_displayable_name'].str.replace("the ", "", regex=True)
KS_1['created_at'] = pd.to_datetime(KS_1['created_at'], unit='s')
KS_1['launched_at'] = pd.to_datetime(KS_1['launched_at'], unit='s')
KS_1[['parent_category', 'sub_category']] = pd.json_normalize(KS_1['category'].apply(json.loads))[['parent_name', 'name']]
# We only care about creator_id and backing_action_count here, but need to parse everything to get them
KS_1[['creator_id', 'creator_name', 'is_registered', 'is_email_verified', 'is_superbacker', 'backing_action_count']] = pd.json_normalize(KS_1['creator'].apply(json.loads))[['id', 'name', 'is_registered', 'is_email_verified', 'is_superbacker', 'backing_action_count']]

KS_1.dropna(subset=['parent_category', 'sub_category'], inplace=True)

KS_1.sort_values('launched_at', inplace=True)
KS_1.reset_index(inplace=True, drop=True)
KS_1['state_num'] = np.where(KS_1['state']=="successful",1,0)
KS_1['state_binary'] = np.where(KS_1['state']=="successful","Successful","Failed")
# Number of projects the creator has launched before (cumulative count) - this is why we sorted by launched_at
KS_1['creator_prev_projects'] = KS_1.groupby('creator_id').cumcount()

KS_1['has_video'] = pd.notnull(KS_1['video'])

# Parse launch date into month, day, hour, and get campaign length based on deadline date and launch date
KS_1 = pd.concat((KS_1, pd.DataFrame({'launch_year': KS_1['launched_at'].dt.year,
              'launch_month': KS_1['launched_at'].dt.month,
              'launch_day': KS_1['launched_at'].dt.day_name(),
              'launch_hour': KS_1['launched_at'].dt.hour,
              'days_active': (KS_1['deadline'] - KS_1['launched_at']).dt.days})), axis=1)
KS_1['launched_at'] = KS_1['launched_at'].dt.date
KS_1['blurb_length'] = KS_1['blurb'].str.len()
KS_1['hours_from_noon'] = abs(12-KS_1['launch_hour'])
KS_1.fillna({'blurb_length':0}, inplace=True)

# The full dataset has over 100,000 rows, so to make it more manageable I chose to focus on the most current subset (past year and a half) and remove a few outlier funding goals.
KS_data = KS_1[(KS_1['launch_year'] >= 2024) & (KS_1['usd_goal'] < 100000)].drop('launch_year', axis=1)

KS = KS_data[['state_num', 'state_binary', 'launched_at', 'usd_goal', 'parent_category', 'sub_category', 'backing_action_count', 'has_video', 'creator_prev_projects', 'launch_day', 'hours_from_noon', 'launch_month', 'country', 'prelaunch_activated', 'blurb_length',  'staff_pick', 'days_active']]

KS.to_csv("Full_Kickstarter_data.csv")