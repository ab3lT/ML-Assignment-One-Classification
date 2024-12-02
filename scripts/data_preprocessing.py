import pandas as pd
from sklearn.ensemble import IsolationForest

class DataProcessing:
    
    def __init__(self, data:pd.DataFrame) -> None:
        self.data=data
        
    def outlier_removal(self, df, cols):
        iso = IsolationForest(contamination=0.01, random_state=42)
        valid_indices = pd.Series([True] * len(df))
        for col in cols:
            df['predict'] = iso.fit_predict(df[[col]])
            valid_indices &= df['predict'] == 1
        df_filtered = df[valid_indices].drop(columns=['predict'])
        return df_filtered
    def convert_education_with_number(self):
        """
        Convert the 'person_education' column to a consistent numeric format.
    
        Returns:
            pd.DataFrame: DataFrame with 'person_education' column converted to numeric categories.
        """
        # Convert the entire 'person_education' column to string
        self.data['person_education'] = self.data['person_education'].astype(str)
        
        education_mapping = {
            'High School': 0,
            'Associate': 1,
            'Bachelor': 2,
            'Master': 3,
            'Doctorate': 4
        }
        
        # Map values using the dictionary
        self.data['person_education'] = self.data['person_education'].map(education_mapping)
        
        # Fill NaN values with a default value (e.g., -1) before converting to int
        self.data['person_education'] = self.data['person_education'].fillna(-1).astype(int)
        
        return self.data

    def convert_gender_with_number(self):
        """
        Convert the 'person_gender' column to a consistent numeric format.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the 'person_gender' column.
        
        Returns:
            pd.DataFrame: DataFrame with 'person_gender' column converted to numeric categories.
        """
        # Convert the entire 'StateHoliday' column to string
        self.data['person_gender'] = self.data['person_gender'].astype(str)
      
        person_mapping = {
            'male': 0,
            'female': 1
        }
        
        # Convert 'StateHoliday' column using the mapping
        self.data['person_gender'] = self.data['person_gender'].map(person_mapping).astype(int)
        
        return self.data

    
    def convert_home_ownership_with_number(self):
        """
        Convert the 'person_home_ownership' column to a consistent numeric format.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the 'person_home_ownership' column.
        
        Returns:
            pd.DataFrame: DataFrame with 'person_home_ownership' column converted to numeric categories.
        """
        # Convert the entire 'StateHoliday' column to string
        self.data['person_home_ownership'] = self.data['person_home_ownership'].astype(str)
      
        person_home_ownership_mapping = {
            'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3
        }
        
        # Convert 'StateHoliday' column using the mapping
        self.data['person_home_ownership'] = self.data['person_home_ownership'].map(person_home_ownership_mapping).astype(int)
        
        return self.data
    
    def convert_loan_intent_mapping_with_number(self):
        """
        Convert the 'loan_intent_mapping' column to a consistent numeric format.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the 'loan_intent_mapping' column.
        
        Returns:
            pd.DataFrame: DataFrame with 'loan_intent_mapping' column converted to numeric categories.
        """
        # Convert the entire 'StateHoliday' column to string
        self.data['loan_intent_mapping'] = self.data['loan_intent_mapping'].astype(str)
      
        person_home_ownership_mapping = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
        
        # Convert 'StateHoliday' column using the mapping
        self.data['loan_intent_mapping'] = self.data['loan_intent_mapping'].map(person_home_ownership_mapping).astype(int)
        
        return self.data
    def convert_previous_loan_defaults_mapping_with_number(self):
        """
        Convert the 'previous_loan_defaults_mapping ' column to a consistent numeric format.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the 'previous_loan_defaults_mapping ' column.
        
        Returns:
            pd.DataFrame: DataFrame with 'previous_loan_defaults_mapping' column converted to numeric categories.
        """
        # Convert the entire 'previous_loan_defaults_mapping' column to string
        self.data['previous_loan_defaults_mapping'] = self.data['previous_loan_defaults_mapping'].astype(str)
      
        previous_loan_defaults_mapping = {'No': 0, 'Yes': 1}
        # Convert 'StateHoliday' column using the mapping
        self.data['previous_loan_defaults_mapping'] = self.data['previous_loan_defaults_mapping'].map(previous_loan_defaults_mapping).astype(int)
        
        return self.data