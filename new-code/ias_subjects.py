import pandas as pd
import collections

df = pd.read_csv("./new-datasets/ias-education.csv")

new_df = df[['Subject', 'Category_of_Subject']].copy()#create df containing just these two columns

subject = df['Subject'].tolist()
count = collections.Counter(subject)#counting the occurence of each subject in a new column
df_occ = pd.DataFrame.from_dict(count, orient='index').reset_index()#making it into a dataframe
df_occ.columns = ["Subject", "Count"]#renaming columns
df_occ = df_occ[df_occ.Subject != 'N.A.']#dropping N.A.

def find_in_new_df(ele):
    series = new_df[new_df['Subject'] == ele]['Category_of_Subject']#for each subject finding its corresponding category values
    series = series.reset_index(drop= True)#resetting index
    return series[0]#returning the first element of the series which would be what is needed

for index, row in df_occ.iterrows():
    x = find_in_new_df(row['Subject'])
    df_occ.loc[index, 'Category_of_Subject'] = x#inserting the category value in a new column
    
df_occ1 = df_occ.dropna()
df_occ1.to_csv("./new-datasets/IAS subjects.csv")



    