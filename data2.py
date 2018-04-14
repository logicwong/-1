from data_process import *
from sklearn import linear_model

df = pd.read_csv('Building_Permits.csv', encoding='utf-8')
# 选取标称属性列
df_nominal = df.select_dtypes(include=['object','int64'])
# 选取数值属性列
df_numeric = df.select_dtypes(include='float')

nominal_dict = {}
for col in df_nominal:
    nominal_dict[col] = status(df[col], len(df[col]), True)
stats_numeric = df_numeric.apply(status, args=(len(df),False,))
visualize(df_numeric, 3, 4)


#将缺失值剔除
df_dropna = df.dropna()
print(df_dropna.shape)

#通过属性的相关关系来填补缺失值
cols = ['Structural Notification', 'Voluntary Soft-Story Retrofit', 'Fire Only Permit', 'TIDF Compliance']
df_fillna = df[cols].fillna('N')

old_dict = {}
for col in cols:
    old_dict[col] = status(df[col], len(df[col]), True)

new_dict = {}
for col in cols:
    new_dict[col] = status(df_fillna[col], len(df_fillna[col]), True)






