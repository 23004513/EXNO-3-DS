# EX NO:3-Feature Encoding and Transformation

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:
STEP 1:

Read the given Data.

STEP 2:

Clean the Data Set using Data Cleaning Process.

STEP 3:

Apply Feature Encoding for the feature in the data set.

STEP 4:

Apply Feature Transformation for the feature in the data set.

STEP 5:

Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
  ### 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  ### 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:
### Developed by : N.NAVYA SREE
### Reg No : 212223040138

```python

import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/0ca37861-aa74-4b34-a434-ac7b92fa501c)



```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/bf0a1060-133f-4056-8f4f-424dde02c6b1)


```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/c76c2555-5242-46ac-83b5-b5c6e2a3e030)



```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/340dd1b7-c035-4c7b-b0dc-68e01945120c)


```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

![image](https://github.com/user-attachments/assets/99cc80c7-2908-40ec-a76d-675adcce5766)

```py
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/5bf7e719-b503-4081-a7f8-6f05740596b1)




```py
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/9ea27ff8-6202-4592-9159-c06d65a382ee)


```py
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/f3a44a4e-da18-4d7e-94a6-3c7f2e909165)

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/b740a73d-8866-4e79-b052-938a84e51e26)


```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/91cd65f9-f274-465a-96f0-8fd9b6741cae)


```py
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/00d82cc8-42a2-4114-a5cc-a0a878dfee8d)



```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/11b77012-9973-4315-a924-cd4d6c09e961)



```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/67182f91-62f8-4e18-af49-d5dee85a97d7)


```py
df.skew()
```
![image](https://github.com/user-attachments/assets/5cd31b83-d96d-4936-8d78-cb42e164da5c)



```py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/bb62bd1d-6e79-4c54-9f49-47946f8ede5b)



```py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/64e6879a-a6d4-485f-9202-ce2e45e9b9de)




```py
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/b9ff615d-90cd-491b-b0b8-7f9edff28117)



```py
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/cdf08503-a0d2-463e-8df6-dfc355e7ffc3)



```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/57041cd7-b6cb-4dde-ab7a-2a6391598147)


```py
df.skew()
```
![image](https://github.com/user-attachments/assets/24429d16-81ce-41e6-8fa1-c81e0ddac327)



```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/3650b07b-62c4-4f1d-9ade-dc23d20e17dc)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/c9498572-aa58-45d5-be29-c9c0e713e98e)


```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c9ef2b1e-6ba5-45c2-acfe-206bcbf8470f)



```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/29bbcd01-12c4-4e24-954f-be005bcd7a0f)




```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/8bbbd31b-ae2c-4140-9cfb-3dd35d6d3fed)




```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/f5211916-d097-4870-abf3-0e13ff30595c)


```py
dt=pd.read_csv("titanic_dataset.csv")
dt
```

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

![image](https://github.com/user-attachments/assets/f3cf99ee-e177-41b4-9ebf-1007884a80b3)

```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/2d20e56f-4004-4025-a974-578572271546)




## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       



    
