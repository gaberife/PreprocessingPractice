'''
https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease
Thyroid Disease Data Set
sick.names
sick.data
#Does Medical Ailment predict Age???
'''

import pandas as pd
import numpy as np

def readSickData(numRows = None):
    colNames = ["age","sex","on thyroxine","query on thyroxine", 
                "on antithyroid medication","sick","pregnant",
                "thyroid surgery","I131 treatment","query hypothyroid",
                "query hyperthyroid","lithium",
                "goitre","tumor","hypopituitary","psych", 
                "TSH measured","TSH","T3 measured","T3", 
                "TT4 measured","TT4","T4U measured","T4U", 
                "FTI measured","FTI","TBG measured","TBG", 
                "referral source"]
    df = pd.read_csv("data/sick.data", index_col=False, na_values="?", delimiter = ",", header=None, names=colNames, engine='python', nrows=numRows)
    return df

def preprocessing(df):
    df['sex'] = df['pregnant'].map(lambda v: 'F' if v == 't' else None)
    df['sex'] = df['sex'].map(lambda v: 0 if v == 'F' else 1) 
    
    nonNumericAttributes = ['sex','on thyroxine', 'query on thyroxine',
       'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'T3 measured', 'TT4 measured', 'T4U measured', 'FTI measured',
       'TBG measured']
    df[nonNumericAttributes] = df[nonNumericAttributes].apply(lambda df: TrueOrFalse(df, nonNumericAttributes), axis=1) 
    
    missing = ['TSH','T3','TT4','T4U','FTI']
    df[missing] = df[missing].apply(lambda df: Missing(df, missing), axis=1) 

    '''
    TO PREPROCESS: 'referral source' = 'SVHC', 'other', 'SVI', 'STMW', 'SVHD'
    '''
    
    del df['TBG']
    del df['referral source']
    return df

def TrueOrFalse(df, series):
    df[series] = df[series].map(lambda row: 0 if row == 'f' else 1)
    return df

def Missing(df, series):
    df[series] = df[series].map(lambda row: 0 if pd.isna(row) else row)
    return df

def checkTotal(df):
    check = ['on thyroxine', 'query on thyroxine',
       'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'T3 measured', 'TT4 measured', 'T4U measured', 'FTI measured',]
    i = 0
    while i < len(check):
        print("#", check[i], ":", df.loc[df.loc[:,check[i]] == 1,:].shape[0])
        i += 1
        
def test():
    df = readSickData()
    df.to_csv("data/Raw Data.csv", index=False)
    df = preprocessing(df)
    df.to_csv("data/Preprocessed Data.csv", index=False)
    checkTotal(df)
    print(df.loc[:10,:])
    return df

def main():
    pd.set_option('display.max_columns', 30)
    print("Testing Main...")
    test()

if __name__ == "__main__":
    main()