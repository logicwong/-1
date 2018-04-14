from data_process import *
from sklearn import linear_model

df = pd.read_csv('NFL Play by Play 2009-2017 (v4).csv', encoding='utf-8', dtype={'GameID':str, 'down':str,
                                                       'GoalToGo':str, 'FirstDown':str})
# 选取标称属性列
df_nominal = df.select_dtypes(include=['object','int64'])
# 选取数值属性列
df_numeric = df.select_dtypes(include='float')

nominal_dict = {}
for col in df_nominal:
    nominal_dict[col] = status(df[col], len(df[col]), True)
stats_numeric = df_numeric.apply(status, args=(len(df),False,))
visualize(df_numeric, 4, 9)

#可填充属性
cols = ['No_Score_Prob', 'Opp_Field_Goal_Prob', 'Opp_Safety_Prob', 'Opp_Touchdown_Prob', 'Field_Goal_Prob',
        'Safety_Prob', 'Touchdown_Prob', 'ExpPts', 'EPA', 'airEPA', 'yacEPA', 'Home_WP_pre', 'Away_WP_pre',
        'Home_WP_post', 'Away_WP_post', 'Win_Prob', 'WPA', 'airWPA', 'yacWPA']

# 删除缺失值所在的行
df_dropna = df_numeric.dropna(subset=cols)
compare(df_numeric, df_dropna)

# 利用众数填充缺失值
df_fillna = df_numeric.fillna(df_numeric.mode().loc[0])
compare(df_numeric, df_fillna)

# 通过数据对象之间的相似性来填补缺失值并进行直方图比较
df_filled_inter = df_numeric.copy()
# 对每一列数据，分别进行处理
for col in df_filled_inter:
    df_filled_inter[col].interpolate(inplace=True)
compare(df_numeric, df_filled_inter)

# 通过属性的相关关系来填补缺失值
df_sim = df_numeric.copy()
corr_mat = np.abs(df_sim.corr())
for col in df_sim:
    corr_mat.loc[col, col] = 0.0
    sim_attr = corr_mat[col].idxmax()
    hebing = df_numeric[[col,sim_attr]].dropna()
    clf = linear_model.LinearRegression()
    clf = clf.fit(np.array(hebing[sim_attr]).reshape(-1,1), np.array(hebing[col]).reshape(-1,1))
    for index, row in df_sim.iterrows():
       if np.isnan(df_sim.loc[index, col]):
           if not np.isnan(df_sim.loc[index, sim_attr]):
               df_sim.loc[index, col] = clf.predict(df_sim.loc[index, sim_attr])
compare(df_numeric, df_sim)






