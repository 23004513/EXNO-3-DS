## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
# DEVELOPED BY:N.NAVYA SREE
# REG.NO:212223040138
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/c64a4941-d109-4fa9-b259-594d51b34d7f)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/f85a0951-c2bb-4fc9-bd7c-4439b45ee30a)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/fcd059b5-a509-4859-8e58-da07e2356f28)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/48fa5058-b5c6-484d-abc4-c0d95bb0f0d2)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![image](https://github.com/user-attachments/assets/471de1cd-d218-4afe-a3c6-deee74e1c514)
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/81ae329e-0355-4b31-9478-c2d915d0a306)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/b285dc6a-8b85-490c-94e6-70aa9103dbfd)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/c026822f-0de4-4678-a002-ae9a60150088)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/6e2d76e3-f1cf-46e9-ac52-c1e5f6b49301)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/c577586d-585d-4351-9385-f849c813a08c)
```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/28eafa10-327c-4ae6-be1c-fa3088b1d6a0)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/325d4102-3a0c-4ef9-b906-6002d6945cba)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/498ee340-96fd-41b4-b4a1-51700aea3bf2)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/e17fd074-ea76-4e69-892f-aa408e1ce05b)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/8ee67817-acca-48f5-b696-c0ec87b805ae)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/8c8e21d1-06ba-4ae4-8131-59a693e372f3)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/c5c4b854-e675-48f2-abbe-f22bd0a056c4)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/1bbcf851-0ac0-4372-826c-79aeb4c18e34)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/58c03fdd-1cd8-4f5e-aebd-3993011e7fa0)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/abc9e26c-90bb-475a-8012-e9566ce15f5d)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/4dd048c9-ddd1-479a-8573-fb334e01e418)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/16b47c6f-e935-4024-9f11-fa6762b98f83)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/038425bb-c6d2-4ef8-a635-6f12279b062e)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/4ac69d79-14f5-48ae-82f8-79e270ceb2b9)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c0a3f4cc-0735-4434-a169-8c3e5b1264f3)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/17aafe4f-08f8-4e8a-bfb7-21d90d1cce91)
```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![image](https://github.com/user-attachments/assets/a8321446-250a-4c6e-91b8-e8750209c7f0)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/2fe79f6a-dddd-40ed-922e-d10f2ea7b2ef)

# RESULT:
 Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

     

       
