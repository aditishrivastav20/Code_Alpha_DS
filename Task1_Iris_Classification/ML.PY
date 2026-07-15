import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 
# Dataset Loading
df = pd.read_csv("Task1_Iris_Classification/iris copy.csv")
print(df.head())
# Features 
X= df.drop(["Species", "Id"], axis=1)

# Target
y= df["Species"]
print("Features:")
print(X.head())
print("\nTarget:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
# Create Model
model = KNeighborsClassifier(n_neighbors=3)
# Train Model
model.fit(X_train, y_train)
print("Model trained successfully.")
# Prediction
y_pred = model.predict(X_test)
print("Predictions:", y_pred)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100, "%")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm=confusion_matrix(y_test, y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()