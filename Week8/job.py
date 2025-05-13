import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def filter_location(location):
    if ", " in location:
        return location[-2:]
    else:
        return location

data = pd.read_excel("job_dataset.ods", dtype=str)
data = data.dropna()
data["location"] = data["location"].apply(filter_location)
#print(data["career_level"].value_counts())
target_col = "career_level"
# Drop columns
x = data.drop(labels=target_col, axis=1)
y = data[target_col]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(ngram_range=(1,2), stop_words="english"), "description"),
    ("function", OrdinalEncoder(), ["function"]),
    ("industry", TfidfVectorizer(stop_words="english"), "industry")
])

output = preprocessor.fit_transform(x_train)
print(output.shape)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=2024, n_jobs=-1, class_weight="balanced", n_estimators=100))
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))