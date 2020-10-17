import pandas as pd
from collections import Counter
import numpy as np

# interested_col = ['Type',
#        'Part_of_a_policing_operation', 'Policing_operation', 'Latitude',
#        'Longitude', 'Gender', 'Age_range', 'Self_defined_ethnicity',
#        'Officer_defined_ethnicity', 'Legislation', 'Object_of_search',
#        'Outcome', 'Outcome_linked_to_object_of_search',
#        'Removal_of_more_than_just_outer_clothing',
#        ]

interested_col = ['Type',
       'Part_of_a_policing_operation', 'Policing_operation',
       'Gender', 'Age_range', 'Self_defined_ethnicity',
       'Officer_defined_ethnicity', 'Legislation', 'Object_of_search',
       'Outcome', 'Outcome_linked_to_object_of_search',
       'Removal_of_more_than_just_outer_clothing',
       ]


df = pd.read_csv('./crime_Stop_And_Search.csv')
df['Formatted_Date'] = pd.to_datetime(df.Date, format = '%Y-%m-%d %H:%M:%S')

df = df.sort_values(by='Formatted_Date')
df['Small_Date'] = df['Formatted_Date'].dt.strftime('%m-%d-%Y')
df['Small_Date'] = pd.to_datetime(df['Small_Date'], format = '%m-%d-%Y')
df['Small_Timestamp'] = df['Small_Date'].values.astype(np.int64) // 10**9
group_data = df.groupby(['Small_Date', 'Small_Timestamp'])

data = []
for val, val2 in group_data:
    data.append((val, val2))

sorted_data = sorted(data, key=lambda x:x[0][1])

all_data = []
for val, val2 in sorted_data:
    row_data = {}
    row_data['Date'] = val[0]
    for col_name in val2.columns:
        if col_name in interested_col:
            unique_val = Counter(val2[col_name])
            for key, val in unique_val.items():
                if type(key) is type('1'):
                    row_data[col_name + '_' + key] = val
    row_data['Total(Crime)'] = val2.shape[0]
    all_data.append(row_data)

all_data = pd.DataFrame(all_data)
df2 = all_data.pop('Total(Crime)') # remove column x and store it in df2
all_data['Total(Crime)']=df2 # add b series as a 'new' column.
all_data = all_data.fillna(0)

all_data.to_csv('./processed_data.csv', index=False)
