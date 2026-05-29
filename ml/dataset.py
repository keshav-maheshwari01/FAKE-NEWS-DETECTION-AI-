import pandas  as pd 
from sklearn.model_selection import train_test_split


def load_news_dataset():
    fake = pd.read_csv("datasets/raw/Fake.csv")
    true = pd.read_csv("datasets/raw/True.csv")
    fake["label"]=0
    true["label"]=1



    fake["title"] = fake["title"].fillna("")
    fake["text"] = fake["text"].fillna("")
    true["title"] = true["title"].fillna("")
    true["text"] = true["text"].fillna("")
   

    df = pd.concat([fake,true],ignore_index=True)
 

    #print(df.isnull().sum())
    #means the dataset is complete 
    #we only need news title and its content 
    df["news"] = df["title"] +" "+ df["text"] 


    
    df = df[["news","label"]]
    df  = df.sample(frac = 1 , random_state = 45 ).reset_index(drop=True)
    df.to_csv("datasets/processed/cleaned.csv",index=False)

    return df 




def splitting_dataset(df):
    train_news,test_news,train_labels,test_labels= train_test_split(
        df["news"].tolist(),
        df["label"].tolist(),
        train_size = 0.8,
        random_state = 45,
        stratify = df["label"]   )
    return train_news,test_news,train_labels,test_labels

