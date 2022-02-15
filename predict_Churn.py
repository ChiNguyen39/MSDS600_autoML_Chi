import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(path = '/Users/kie/Documents/Regis University/MS/01. MSDS 600/05. Week 5/new_churn_data.csv'):
    """
    Loads churn data into a DataFrame
    """
    df = pd.read_csv(path)
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    model = load_model('GBC')
    predictions = predict_model(model, data=df)
    predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    
    return predictions['Churn_prediction']


if __name__ == "__main__":
    df = load_data('/Users/kie/Documents/Regis University/MS/01. MSDS 600/05. Week 5/new_churn_data.csv')
    predictions = make_predictions(df)
    predicted_df = pd.concat([df,predictions], axis=1)
    print('predictions:')
    print(predicted_df)
