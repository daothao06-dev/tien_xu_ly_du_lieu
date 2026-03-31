import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = {
    'Hours':[1,2,3,4,5,6,7,8,],
    'Score':[2,4,5,6,7,8,8.5,9]
}
df=pd.DataFrame(data)
print(df)
X=df[['Hours']]
Y=df[['Score']]
model = LinearRegression()
model.fit(X,Y)
new_hours =pd.DataFrame([[6]],columns=['Hours'])
predicted_score = model.predict(new_hours)
print('Predicted score:',predicted_score)
new_data =pd.DataFrame([[4],[6],[9]],columns=['Hours'])
predictions = model.predict(new_data)
print(predictions)
plt.scatter(X,Y)
plt.plot(X, model.predict(X))
plt.xlabel('Hours studied')
plt.ylabel('Score')
plt.title('Hours vs Score')
plt.show()
from sklearn.metrics import r2_score
y_pred = model.predict(X)
Score = r2_score(Y,y_pred)
print('R2 Score:',Score)

