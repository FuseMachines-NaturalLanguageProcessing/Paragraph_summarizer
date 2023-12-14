import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv("bbc-news-summary.csv")[["Articles", "Summaries"]]
    
    data["Articles"] = data["Articles"].apply(lambda x: x.replace("..", '\n\n'))
    
    data['Articles'] = data['Articles'].str.encode('ascii', 'ignore').str.decode('ascii')
    data['Summaries'] = data['Summaries'].str.encode('ascii', 'ignore').str.decode('ascii')
    
    data = data.dropna()
    
    return train_test_split(data, test_size=0.2)
    