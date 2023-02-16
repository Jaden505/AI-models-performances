import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataFormat:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        # Load dataset
        self.df_bank = pd.read_csv('bank.csv')
        self.df_bank = self.df_bank.drop('duration', axis=1)
        self.df_bank.head()

    def standard_scaler(self):
        # Copying original dataframe
        self.df_bank_ready = self.df_bank.copy()
        scaler = StandardScaler()
        num_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
        self.df_bank_ready[num_cols] = scaler.fit_transform(self.df_bank_ready[num_cols])
        self.df_bank_ready.head()

    def one_hot_encoder(self):
        encoder = OneHotEncoder(sparse=False)
        cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

        # Encode Categorical Data
        df_encoded = pd.DataFrame(encoder.fit_transform(self.df_bank_ready[cat_cols]))
        df_encoded.columns = encoder.get_feature_names_out(cat_cols)

        # Replace Categotical Data with Encoded Data
        self.df_bank_ready = self.df_bank_ready.drop(cat_cols, axis=1)
        self.df_bank_ready = pd.concat([df_encoded, self.df_bank_ready], axis=1)

        # Encode target value
        self.df_bank_ready['deposit'] = self.df_bank_ready['deposit'].apply(lambda x: 1 if x == 'yes' else 0)

        print('Shape of dataframe:', self.df_bank_ready.shape)
        self.df_bank_ready.head()

    def split_data(self):
        # Select Features and targets
        feature = self.df_bank_ready.drop('deposit', axis=1)
        target = self.df_bank_ready['deposit']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(feature, target,
                                                            shuffle=True,
                                                            test_size=0.2,
                                                            random_state=1)

        # Show the Training and Testing Data
        print('Shape of training feature:', self.X_train.shape)
        print('Shape of testing feature:', self.X_test.shape)
        print('Shape of training label:', self.y_train.shape)
        print('Shape of training label:', self.y_test.shape)

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
