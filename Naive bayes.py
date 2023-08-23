import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rainy", "Rainy", "Rainy", "Overcast", "Sunny", "Sunny", "Rainy"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal"],
    "Windy": ["False", "True", "False", "False", "False", "True", "True", "False", "False", "False"],
    "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes"]
}


df = pd.DataFrame(data)

df = pd.get_dummies(df, columns=["Outlook", "Temperature", "Humidity", "Windy"])

X = df.drop("Play", axis=1)
y = df["Play"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


accuracy = (y_pred == y_test).sum() / len(y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")
