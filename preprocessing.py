# 2. PREPROCESSING TECHNIQUES
#a) Attribute Selection
import pandas as pd

df = pd.DataFrame({
    "id": [1,2,3],
    "name": ["Ravi","Anjali","Vikram"],
    "marks": [95, 88, 76],
    "grade": ["A+","A","B"]
})
print("original dataset:\n",df)
# Select important attributes
selected = df[["name", "marks"]]
print("\n",selected)
print()

#b) Handling Missing Values (Mean, Median, Mode)
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name": ["Ravi","Anjali","Vikram"],
    "marks": [95, np.nan, 76]
})

df["marks_mean"] = df["marks"].fillna(df["marks"].mean())
df["marks_median"] = df["marks"].fillna(df["marks"].median())
df["marks_mode"] = df["marks"].fillna(df["marks"].mode()[0])

print(df)
print()

#C) Discretization
import pandas as pd

df = pd.DataFrame({
    "marks": [95, 88, 76, 90]
})

# 3 bins → 3 labels
df["bin"] = pd.cut(df["marks"], bins=3, labels=["Low", "Medium", "High"])
print(df)
print()

#D) Outlier Elimination (IQR Method)
import pandas as pd

df = pd.DataFrame({
    "marks": [95, 88, 76, 800]
})

q1 = df["marks"].quantile(0.25)
q3 = df["marks"].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

filtered = df[(df["marks"] >= lower) & (df["marks"] <= upper)]
print(filtered)
print()
