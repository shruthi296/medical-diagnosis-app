import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('lung_cancer.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Smoking', 'YellowFingers', 'Anxiety', 'PeerPressure', 'ChronicDisease', 'Fatigue', 'Allergy', 'Wheezing', 'AlcoholConsuming', 'Coughing', 'ShortnessOfBreath', 'SwallowingDifficulty', 'ChestPain']
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split data
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open('lung_cancer_model.pkl', 'wb') as f:
    pickle.dump(model, f)