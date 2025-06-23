import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def feature_engineer(df):
    df=df.copy()
    df['Income_per_Minute']=df['AnnualIncome']/(df['TimeSpentOnWebsite']+1)
    categorical=['ProductCategory', 'LoyaltyProgram']
    encoder=OneHotEncoder(sparse_output=False, drop='first')
    encoded= encoder.fit_transform(df[categorical])
    encoded_df=pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical))
    df=pd.concat([df.drop(columns=categorical), encoded_df], axis=1)
    return df
   
    