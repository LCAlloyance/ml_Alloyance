import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
import os

class LCAPredictor:
    def __init__(self, model_dir='models/'):
        """
        Initialize the LCA Predictor with saved models
        
        Args:
            model_dir: Directory containing saved model files
        """
        self.model_dir = model_dir
        self.models = {}
        self.label_encoders = {}
        
        # Complete column structure from your actual CSV data
        self.categorical_cols = [
            'Process Stage', 'Technology', 'Time Period', 'Location', 
            'Functional Unit', 'Raw Material Type', 'Energy Input Type',
            'Transport Mode', 'Fuel Type', 'End-of-Life Treatment'
        ]
        
        self.numerical_cols = [
            'Raw Material Quantity (kg or unit)', 'Energy Input Quantity (MJ)',
            'Transport Distance (km)', 'Emissions to Air CO2 (kg)',
            'Emissions to Air SOx (kg)', 'Emissions to Air NOx (kg)',
            'Emissions to Air Particulate Matter (kg)', 'Emissions to Water BOD (kg)',
            'Emissions to Water Heavy Metals (kg)', 'Greenhouse Gas Emissions (kg CO2-eq)'
        ]
        
        self.engineered_cols = [
            'Energy_per_Material', 'Total_Air_Emissions', 'Total_Water_Emissions',
            'Circularity_Score', 'Transport_Intensity', 'GHG_per_Material',
            'Time_Period_Numeric'
        ]
        
        # CRITICAL: This must match your training data column order exactly!
        self.expected_column_order = [
            'Process Stage', 'Technology', 'Time Period', 'Location', 
            'Functional Unit', 'Raw Material Type', 
            'Raw Material Quantity (kg or unit)', 'Energy Input Type', 
            'Energy Input Quantity (MJ)', 'Transport Mode', 
            'Transport Distance (km)', 'Fuel Type', 
            'Emissions to Air CO2 (kg)', 'Emissions to Air SOx (kg)', 
            'Emissions to Air NOx (kg)', 'Emissions to Air Particulate Matter (kg)', 
            'Emissions to Water BOD (kg)', 'Emissions to Water Heavy Metals (kg)', 
            'Greenhouse Gas Emissions (kg CO2-eq)', 'End-of-Life Treatment',
            'Energy_per_Material', 'Total_Air_Emissions', 'Total_Water_Emissions',
            'Circularity_Score', 'Transport_Intensity', 'GHG_per_Material',
            'Time_Period_Numeric'
        ]
        
        # Default values for missing columns - updated based on your data ranges
        self.default_values = {
            'Emissions to Air CO2 (kg)': 0.5,
            'Emissions to Air SOx (kg)': 0.025,
            'Emissions to Air NOx (kg)': 0.015,
            'Emissions to Air Particulate Matter (kg)': 0.01,
            'Emissions to Water BOD (kg)': 0.02,
            'Emissions to Water Heavy Metals (kg)': 0.01,
            'Greenhouse Gas Emissions (kg CO2-eq)': 0.6,
            'Transport Distance (km)': 500.0,
            'Raw Material Quantity (kg or unit)': 2.0,
            'Energy Input Quantity (MJ)': 40.0
        }
        
        self.load_models()
        self.load_encoders()
    
    def load_models(self):
        """Load the saved XGBoost models"""
        try:
            if os.path.exists(os.path.join(self.model_dir, 'model_recycled_content.json')):
                model_recycled = XGBRegressor()
                model_recycled.load_model(os.path.join(self.model_dir, 'model_recycled_content.json'))
                self.models['recycled_content'] = model_recycled
            
            if os.path.exists(os.path.join(self.model_dir, 'model_reuse_potential.json')):
                model_reuse = XGBRegressor()
                model_reuse.load_model(os.path.join(self.model_dir, 'model_reuse_potential.json'))
                self.models['reuse_potential'] = model_reuse
            
            if os.path.exists(os.path.join(self.model_dir, 'model_recovery_rate.json')):
                model_recovery = XGBRegressor()
                model_recovery.load_model(os.path.join(self.model_dir, 'model_recovery_rate.json'))
                self.models['recovery_rate'] = model_recovery
            
            if len(self.models) == 0:
                print("Warning: No models found. Creating dummy models for testing.")
                self._create_dummy_models()
            else:
                print(f"Models loaded successfully! Available models: {list(self.models.keys())}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Creating dummy models for testing...")
            self._create_dummy_models()
    
    def _create_dummy_models(self):
        """Create dummy models for testing when no trained models exist"""
        print("Creating dummy XGBoost models for testing purposes...")
        
        # Create dummy training data with correct feature count and exact column order
        n_features = len(self.expected_column_order)
        X_dummy = np.random.randn(100, n_features)
        X_dummy_df = pd.DataFrame(X_dummy, columns=self.expected_column_order)
        
        # Create realistic dummy targets based on your data ranges
        y_recycled = np.random.uniform(10, 90, 100)  # 10-90% range
        y_reuse = np.random.uniform(0, 85, 100)      # 0-85% range  
        y_recovery = np.random.uniform(20, 80, 100)   # 20-80% range
        
        # Train dummy models with correct feature names
        self.models['recycled_content'] = XGBRegressor(random_state=42, n_estimators=10)
        self.models['reuse_potential'] = XGBRegressor(random_state=42, n_estimators=10)
        self.models['recovery_rate'] = XGBRegressor(random_state=42, n_estimators=10)
        
        self.models['recycled_content'].fit(X_dummy_df, y_recycled)
        self.models['reuse_potential'].fit(X_dummy_df, y_reuse)
        self.models['recovery_rate'].fit(X_dummy_df, y_recovery)
        
        print("Dummy models created successfully!")
    
    def load_encoders(self):
        """Load or create label encoders for categorical variables"""
        try:
            with open(os.path.join(self.model_dir, 'label_encoders.pkl'), 'rb') as f:
                self.label_encoders = pickle.load(f)
            print("Label encoders loaded successfully!")
        except:
            print("Creating new label encoders with mappings from your data...")
            self._create_encoders_from_data()
    
    def _create_encoders_from_data(self):
        """Create label encoders based on your actual data values"""
        # Mappings based on your CSV data
        data_mappings = {
            'Process Stage': ['Raw Material Extraction', 'Manufacturing', 'Use', 'Transport', 'End-of-Life'],
            'Technology': ['Conventional', 'Advanced', 'Emerging'],
            'Time Period': ['2010-2014', '2015-2019', '2020-2025'],
            'Location': ['North America', 'Europe', 'South America', 'Asia'],
            'Functional Unit': ['1 kg Aluminium Sheet', '1 kg Copper Wire', '1 m2 Aluminium Panel'],
            'Raw Material Type': ['Copper Ore', 'Copper Scrap', 'Aluminium Scrap', 'Aluminium Ore'],
            'Energy Input Type': ['Electricity', 'Coal', 'Natural Gas'],
            'Transport Mode': ['Rail', 'Ship', 'Truck'],
            'Fuel Type': ['Diesel', 'Electric', 'Heavy Fuel Oil'],
            'End-of-Life Treatment': ['Recycling', 'Landfill', 'Incineration', 'Reuse']
        }
        
        for col, values in data_mappings.items():
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(values)
    
    def save_encoders(self):
        """Save label encoders for future use"""
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, 'label_encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print("Label encoders saved!")
    
    def estimate_emissions(self, input_data):
        """
        Estimate emission values based on input parameters
        Uses emission factors derived from your data patterns
        """
        # Get basic parameters
        material_qty = input_data.get('Raw Material Quantity (kg or unit)', 2.0)
        energy_input = input_data.get('Energy Input Quantity (MJ)', 40.0)
        material_type = input_data.get('Raw Material Type', 'Aluminium Scrap')
        energy_type = input_data.get('Energy Input Type', 'Electricity')
        process_stage = input_data.get('Process Stage', 'Manufacturing')
        transport_dist = input_data.get('Transport Distance (km)', 500.0)
        
        # Base emission factors (simplified from your data patterns)
        base_factors = {
            'Electricity': {'co2_factor': 0.02, 'sox_factor': 0.001, 'nox_factor': 0.0006},
            'Coal': {'co2_factor': 0.06, 'sox_factor': 0.003, 'nox_factor': 0.002},
            'Natural Gas': {'co2_factor': 0.035, 'sox_factor': 0.0005, 'nox_factor': 0.001}
        }
        
        # Process stage multipliers
        stage_multipliers = {
            'Raw Material Extraction': 2.0,
            'Manufacturing': 1.0,
            'Use': 0.3,
            'Transport': 0.5,
            'End-of-Life': 0.4
        }
        
        # Get factors
        factors = base_factors.get(energy_type, base_factors['Electricity'])
        stage_mult = stage_multipliers.get(process_stage, 1.0)
        
        # Calculate emissions based on your data scale
        emissions = {}
        base_co2 = energy_input * factors['co2_factor'] * stage_mult
        emissions['Emissions to Air CO2 (kg)'] = max(0.01, base_co2)
        emissions['Emissions to Air SOx (kg)'] = max(0.001, base_co2 * 0.05)
        emissions['Emissions to Air NOx (kg)'] = max(0.001, base_co2 * 0.03)
        emissions['Emissions to Air Particulate Matter (kg)'] = max(0.001, base_co2 * 0.02)
        
        # Water emissions
        emissions['Emissions to Water BOD (kg)'] = max(0.005, material_qty * 0.01)
        emissions['Emissions to Water Heavy Metals (kg)'] = max(0.001, material_qty * 0.005)
        
        # GHG total
        emissions['Greenhouse Gas Emissions (kg CO2-eq)'] = emissions['Emissions to Air CO2 (kg)'] * 1.2
        
        return emissions
    
    def autofill_missing_data(self, input_data):
        """
        Automatically fill missing data with intelligent defaults
        """
        filled_data = input_data.copy()
        
        # Fill missing numerical values with defaults
        for col, default_val in self.default_values.items():
            if col not in filled_data or pd.isna(filled_data.get(col)):
                filled_data[col] = default_val
        
        # Estimate emissions if not provided
        estimated_emissions = self.estimate_emissions(filled_data)
        for emission_col, value in estimated_emissions.items():
            if emission_col not in filled_data or pd.isna(filled_data.get(emission_col)):
                filled_data[emission_col] = value
        
        # Fill missing categorical values with your data defaults
        categorical_defaults = {
            'Process Stage': 'Manufacturing',
            'Technology': 'Advanced',
            'Time Period': '2020-2025',
            'Location': 'Europe',
            'Functional Unit': '1 kg Aluminium Sheet',
            'Raw Material Type': 'Aluminium Scrap',
            'Energy Input Type': 'Electricity',
            'Transport Mode': 'Rail',
            'Fuel Type': 'Electric',
            'End-of-Life Treatment': 'Recycling'
        }
        
        for col, default_val in categorical_defaults.items():
            if col not in filled_data or pd.isna(filled_data.get(col)):
                filled_data[col] = default_val
        
        return filled_data
    
    def create_engineered_features(self, df):
        """
        Create all engineered features exactly as in training
        """
        # Energy per material (avoid division by zero)
        df['Energy_per_Material'] = df['Energy Input Quantity (MJ)'] / np.maximum(df['Raw Material Quantity (kg or unit)'], 0.1)
        
        # Total air emissions
        air_emission_cols = [
            'Emissions to Air CO2 (kg)', 'Emissions to Air SOx (kg)',
            'Emissions to Air NOx (kg)', 'Emissions to Air Particulate Matter (kg)'
        ]
        df['Total_Air_Emissions'] = df[air_emission_cols].sum(axis=1)
        
        # Total water emissions
        water_emission_cols = [
            'Emissions to Water BOD (kg)', 'Emissions to Water Heavy Metals (kg)'
        ]
        df['Total_Water_Emissions'] = df[water_emission_cols].sum(axis=1)
        
        # Circularity Score (placeholder during prediction)
        df['Circularity_Score'] = 50.0  # Neutral baseline
        
        # Transport intensity
        df['Transport_Intensity'] = df['Transport Distance (km)'] / np.maximum(df['Raw Material Quantity (kg or unit)'], 0.1)
        
        # GHG per material
        df['GHG_per_Material'] = df['Greenhouse Gas Emissions (kg CO2-eq)'] / np.maximum(df['Raw Material Quantity (kg or unit)'], 0.1)
        
        # Time period numeric - extract year from period strings
        def extract_year(period_str):
            try:
                if '-' in str(period_str):
                    # Handle ranges like "2020-2025"
                    return int(str(period_str).split('-')[0])
                else:
                    # Handle single years
                    return int(str(period_str))
            except:
                return 2020  # Default year
        
        df['Time_Period_Numeric'] = df['Time Period'].apply(extract_year)
        
        return df
    
    def preprocess_input(self, input_data):
        """
        Complete preprocessing pipeline with autofill and feature engineering
        """
        # Step 1: Autofill missing data
        filled_data = self.autofill_missing_data(input_data)
        
        # Step 2: Convert to DataFrame
        df = pd.DataFrame([filled_data])
        
        # Step 3: Create engineered features
        df = self.create_engineered_features(df)
        
        # Step 4: Handle categorical variables
        for col in self.categorical_cols:
            if col in df.columns:
                try:
                    # Try to transform using existing encoder
                    df[col] = self.label_encoders[col].transform(df[col])
                except (KeyError, ValueError) as e:
                    # If category is new, assign most common value (usually 0)
                    print(f"Unknown category in {col}: {df[col].iloc[0]} - using default encoding 0")
                    df[col] = 0
                
                # Convert to categorical for XGBoost
                df[col] = df[col].astype('category')
        
        # Step 5: Ensure all required columns exist and use EXACT column order from training
        for col in self.expected_column_order:
            if col not in df.columns:
                if col in self.default_values:
                    df[col] = self.default_values[col]
                else:
                    df[col] = 0.0  # Default for missing columns
        
        # Step 6: CRITICAL - Reorder columns to match exact training order
        df = df[self.expected_column_order]
        
        # Debug info
        print(f"Expected columns: {len(self.expected_column_order)}")
        print(f"Actual columns: {len(df.columns)}")
        print(f"Column order match: {list(df.columns) == self.expected_column_order}")
        
        return df
    
    def predict(self, input_data):
        """
        Make predictions with intelligent autofill for missing data
        """
        try:
            # Preprocess input with autofill
            X_processed = self.preprocess_input(input_data)
            
            print(f"Preprocessed data shape: {X_processed.shape}")
            print(f"Feature names: {list(X_processed.columns)}")
            
            # Make predictions
            predictions = {}
            predictions['recycled_content'] = float(self.models['recycled_content'].predict(X_processed)[0])
            predictions['reuse_potential'] = float(self.models['reuse_potential'].predict(X_processed)[0])
            predictions['recovery_rate'] = float(self.models['recovery_rate'].predict(X_processed)[0])
            
            # Ensure predictions are within valid range (0-100%)
            for key in predictions:
                predictions[key] = max(0, min(100, predictions[key]))
            
            return predictions
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            print("Input data keys:", list(input_data.keys()))
            import traceback
            traceback.print_exc()
            raise
    
    def predict_with_details(self, input_data):
        """
        Make predictions and return both predictions and filled input data
        """
        filled_data = self.autofill_missing_data(input_data)
        predictions = self.predict(input_data)
        
        return {
            'predictions': predictions,
            'filled_input': filled_data,
            'missing_filled': [k for k in filled_data.keys() if k not in input_data.keys()]
        }
    
    def batch_predict(self, input_data_list):
        """Make predictions for multiple inputs with autofill"""
        results = []
        for input_data in input_data_list:
            try:
                result = self.predict(input_data)
                results.append(result)
            except Exception as e:
                print(f"Error in batch prediction: {e}")
                results.append(None)
        return results


# Convenience function for easy integration
def predict_lca_circularity(user_input, model_dir='models/'):
    """
    Convenience function with intelligent autofill for missing data
    """
    predictor = LCAPredictor(model_dir)
    return predictor.predict(user_input)


def predict_with_autofill_details(user_input, model_dir='models/'):
    """
    Convenience function that shows what data was auto-filled
    """
    predictor = LCAPredictor(model_dir)
    return predictor.predict_with_details(user_input)


# Example usage with minimal input based on your data
if __name__ == "__main__":
    # Example with minimal input - system will autofill missing values
    minimal_input = {
        'Raw Material Quantity (kg or unit)': 2.5,
        'Energy Input Quantity (MJ)': 50.0,
        'Transport Distance (km)': 600,
        'Process Stage': 'Manufacturing',
        'Technology': 'Advanced',
        'Raw Material Type': 'Aluminium Scrap',
        'Location': 'Europe'
    }
    
    print("Testing with minimal input based on your data structure...")
    print("Provided:", list(minimal_input.keys()))
    
    try:
        # Get predictions with details about what was filled
        result = predict_with_autofill_details(minimal_input)
        
        print("\nPredictions:")
        for key, value in result['predictions'].items():
            print(f"  {key}: {value:.2f}%")
        
        print(f"\nAuto-filled {len(result['missing_filled'])} missing values:")
        for col in result['missing_filled'][:8]:  # Show first 8
            print(f"  {col}: {result['filled_input'][col]}")
        
        if len(result['missing_filled']) > 8:
            print(f"  ... and {len(result['missing_filled']) - 8} more")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()