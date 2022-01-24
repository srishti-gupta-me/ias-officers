#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from itertools import product


# In[5]:


experience_df = pd.read_csv("./new-datasets/ias-experience.csv")
education_df = pd.read_csv("./new-datasets/ias-education.csv")


# In[7]:


subjects_df = pd.read_csv("./new-datasets/IAS subjects.csv")


# In[4]:


# copy the subject columns to a list
subjects_items = education_df["Subject"].copy()


for subject, category in zip(subjects_df["Subject"], subjects_df["Category_of_Subject"]):
    subjects_items.replace({subject: category}, inplace=True)


# In[5]:


education_df["Category_of_Subject"] = subjects_items


# In[56]:


education_df = education_df.loc[education_df["Category_of_Subject"] != "N.A."]
experience_df = experience_df.loc[experience_df["Category_of_Experience"] != "N.A."]


# In[57]:


education_df.dropna(inplace=True)
experience_df.dropna(inplace=True)


# In[8]:


names = education_df["Name"].unique()


# In[9]:


from tqdm import tqdm


# In[58]:


subject_to_exp_map = []
for name in tqdm(names):
    temp_edu = education_df.loc[education_df["Name"] == name]
    temp_exp = experience_df.loc[experience_df["Name"] == name]

    subject_list = temp_edu["Category_of_Subject"].unique().tolist()
    exp_list = temp_exp["Category_of_Experience"].apply(str).map(str.strip).unique().tolist()

    products = list(product(subject_list, exp_list))
    subject_to_exp_map.extend(products)


# In[59]:


temp = pd.Series(subject_to_exp_map)


# In[60]:


count_df = pd.DataFrame(temp.value_counts())


# In[61]:


subject = []
experience = []

for item in count_df.index:
    subject.append(item[0])
    experience.append(item[1])


# In[62]:


count_df["Subject"] = subject
count_df["Experience"] = experience


# In[63]:


count_df.to_clipboard(index=False)
count_df.to_csv("./new-datasets/unigram_relations.csv", index = False)

# In[ ]:




