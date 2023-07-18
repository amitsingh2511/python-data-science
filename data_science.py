import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but IsolationForest was fitted with feature names")

warnings.filterwarnings("ignore", message="Unknown extension is not supported and will be removed")


data = pd.read_excel('data.xlsx')

# Assuming 'time' is a column representing the timestamp of each record, set it as the index
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)

# Convert any non-numeric values to NaN (Not a Number)
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing (NaN) values
data.dropna(inplace=True)

# Select the variables for anomaly detection
variables_to_detect = ['Cyclone_Inlet_Gas_Temp', 'Cyclone_Gas_Outlet_Temp',
                       'Cyclone_Outlet_Gas_draft', 'Cyclone_cone_draft',
                       'Cyclone_Material_Temp']

# Initialize the Isolation Forest model
model = IsolationForest(contamination=0.01) 

# Fit the model to the selected variables
model.fit(data[variables_to_detect])


data['Anomaly'] = model.predict(data[variables_to_detect])
print("data['Anomaly']===",data['Anomaly'])

num_anomalies = len(data[data['Anomaly'] == -1])
print(f"Number of anomalies: {num_anomalies}")

# Get the row numbers where anomalies occur
anomaly_rows = data.index[data['Anomaly'] == -1]
print("Row numbers where anomalies occur:")
print(anomaly_rows)


plt.figure(figsize=(12, 6))

# Plot the variables along with the anomalies
for variable in variables_to_detect:
    plt.plot(data.index, data[variable], label=variable)
    plt.scatter(data[data['Anomaly'] == -1].index, data[data['Anomaly'] == -1][variable],
                color='red', label='Anomaly')

plt.xlabel('time')
plt.ylabel('Values')
plt.legend()
plt.title('Cyclone Preheater Anomalies')
plt.tight_layout()
plt.show()
