import numpy as np
import json
from typing import Dict, Any, Optional, Tuple
import random
from dataclasses import dataclass

@dataclass
class PredictionModel:
    """Represents a prediction model with coefficients and factors"""
    base_value: float
    coefficients: Dict[str, float]
    material_factors: Dict[str, float]
    location_factors: Dict[str, float]
    noise_factor: float = 0.1

class PredictionEngine:
    """AI-based prediction engine for sustainability metrics"""
    
    def __init__(self):
        self.models = self._initialize_models()
        self.random_seed = 42  # For reproducible results
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
    
    def generate_predictions(self, enhanced_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate AI predictions for sustainability metrics
        
        Args:
            enhanced_data: Enhanced input data from autofill
            
        Returns:
            Dictionary containing predicted sustainability metrics
        """
        
        print("🤖 Generating AI-powered sustainability predictions...")
        
        # Extract relevant inputs
        user_inputs = enhanced_data.get('user_inputs', {})
        product_name = enhanced_data.get('product_name', '')
        material_type = self._extract_material_type(product_name)
        
        # Generate all predictions
        predictions = {}
        
        # Core LCA metrics
        predictions['gwp_kg_co2_eq'] = self._predict_gwp(user_inputs, material_type)
        predictions['water_consumption_m3'] = self._predict_water_consumption(user_inputs, material_type)
        predictions['energy_per_material_mj'] = self._predict_energy_intensity(user_inputs, material_type)
        
        # Air and water emissions
        predictions['total_air_emissions_kg'] = self._predict_air_emissions(user_inputs, material_type, predictions['gwp_kg_co2_eq'])
        predictions['total_water_emissions_kg'] = self._predict_water_emissions(user_inputs, material_type, predictions['water_consumption_m3'])
        
        # Circularity metrics
        predictions['material_circularity_indicator'] = self._predict_mci(user_inputs, material_type)
        predictions['end_of_life_recycling_rate_percent'] = self._predict_recycling_rate(user_inputs, material_type)
        predictions['circularity_score'] = self._calculate_circularity_score(predictions, user_inputs)
        
        # Improvement potential calculations
        predictions['potential_gwp_reduction_renewable_percent'] = self._calculate_renewable_potential(user_inputs, predictions['gwp_kg_co2_eq'])
        predictions['potential_mci_improvement_recycling_percent'] = self._calculate_recycling_improvement_potential(predictions['material_circularity_indicator'])
        
        # Round all predictions to reasonable precision
        predictions = {k: round(v, 2) if isinstance(v, float) else v for k, v in predictions.items()}
        
        print(f"   ✓ Generated {len(predictions)} sustainability predictions")
        
        return predictions
    
    def _predict_gwp(self, inputs: Dict, material_type: str) -> float:
        """Predict Global Warming Potential (kg CO2-eq)"""
        
        model = self.models['gwp']
        
        # Base calculation
        base = model.base_value * model.material_factors.get(material_type, 1.0)
        
        # Energy source impact
        energy_factor = {
            'Renewable (Hydro)': 0.1,
            'Renewable (Solar)': 0.2,
            'Renewable (Wind)': 0.15,
            'Nuclear': 0.3,
            'Natural Gas': 0.6,
            'Mixed Grid': 1.0,
            'Mixed Grid (EU)': 0.8,
            'Coal-dominated Grid': 1.8,
            'Electricity': 1.0
        }.get(inputs.get('energy_source', 'Mixed Grid'), 1.0)
        
        # Transport impact
        transport_distance = float(inputs.get('transport_distance_km', 500))
        transport_factor = {
            'Rail': 0.03,
            'Road': 0.12,
            'Ship': 0.015,
            'Air': 0.5
        }.get(inputs.get('transport_mode', 'Road'), 0.12)
        
        transport_impact = transport_distance * transport_factor
        
        # Recycled content benefit
        recycled_percent = float(inputs.get('recycled_content_percent', 0)) / 100
        recycling_benefit = recycled_percent * 0.7  # Up to 70% reduction
        
        # Location factor
        location_factor = model.location_factors.get(inputs.get('location', 'Global Average'), 1.0)
        
        # Final calculation with some realistic noise
        gwp = base * energy_factor * location_factor + transport_impact
        gwp = gwp * (1 - recycling_benefit)
        gwp = gwp * (1 + np.random.normal(0, model.noise_factor))
        
        return max(0, gwp)
    
    def _predict_water_consumption(self, inputs: Dict, material_type: str) -> float:
        """Predict water consumption (m³)"""
        
        model = self.models['water']
        base = model.base_value * model.material_factors.get(material_type, 1.0)
        
        # Processing method impact
        processing_factor = {
            'Conventional': 1.0,
            'Electric': 0.8,
            'Recycling': 0.3,
            'Smelting': 1.5
        }.get(inputs.get('processing_method', 'Conventional'), 1.0)
        
        # Location water stress factor
        location_factor = model.location_factors.get(inputs.get('location', 'Global Average'), 1.0)
        
        # Recycled content impact
        recycled_percent = float(inputs.get('recycled_content_percent', 0)) / 100
        water_reduction = recycled_percent * 0.6
        
        water = base * processing_factor * location_factor * (1 - water_reduction)
        water = water * (1 + np.random.normal(0, model.noise_factor))
        
        return max(0, water)
    
    def _predict_energy_intensity(self, inputs: Dict, material_type: str) -> float:
        """Predict energy intensity per material (MJ per unit)"""
        
        model = self.models['energy']
        base = model.base_value * model.material_factors.get(material_type, 1.0)
        
        # Processing method impact
        processing_factor = {
            'Conventional': 1.0,
            'Electric': 1.2,
            'Recycling': 0.4,
            'Smelting': 1.8
        }.get(inputs.get('processing_method', 'Conventional'), 1.0)
        
        # Energy source efficiency
        energy_efficiency = {
            'Renewable (Hydro)': 1.1,
            'Renewable (Solar)': 1.0,
            'Nuclear': 1.05,
            'Natural Gas': 0.9,
            'Mixed Grid': 1.0,
            'Coal-dominated Grid': 0.85
        }.get(inputs.get('energy_source', 'Mixed Grid'), 1.0)
        
        energy = base * processing_factor / energy_efficiency
        energy = energy * (1 + np.random.normal(0, model.noise_factor))
        
        return max(0, energy)
    
    def _predict_air_emissions(self, inputs: Dict, material_type: str, gwp: float) -> float:
        """Predict total air emissions (kg) based on GWP and other factors"""
        
        # Air emissions correlate with GWP but include other pollutants
        base_ratio = 0.035  # Base ratio of air emissions to GWP
        
        material_multipliers = {
            'steel': 1.2,
            'aluminum': 1.5,
            'copper': 1.1,
            'plastic': 0.8,
            'generic': 1.0
        }
        
        multiplier = material_multipliers.get(material_type, 1.0)
        air_emissions = gwp * base_ratio * multiplier
        air_emissions = air_emissions * (1 + np.random.normal(0, 0.15))
        
        return max(0, air_emissions)
    
    def _predict_water_emissions(self, inputs: Dict, material_type: str, water_consumption: float) -> float:
        """Predict water emissions (kg) based on water consumption"""
        
        # Water emissions are typically 10-20% of water consumption in kg equivalent
        base_ratio = 0.15
        
        processing_impact = {
            'Conventional': 1.0,
            'Electric': 0.7,
            'Recycling': 0.5,
            'Smelting': 1.3
        }.get(inputs.get('processing_method', 'Conventional'), 1.0)
        
        water_emissions = water_consumption * 1000 * base_ratio * processing_impact  # Convert m³ to kg
        water_emissions = water_emissions * (1 + np.random.normal(0, 0.2))
        
        return max(0, water_emissions / 1000)  # Convert back for reasonable scale
    
    def _predict_mci(self, inputs: Dict, material_type: str) -> float:
        """Predict Material Circularity Indicator (0-1 scale)"""
        
        model = self.models['mci']
        base = model.base_value * model.material_factors.get(material_type, 1.0)
        
        # Recycled content is the primary driver of MCI
        recycled_percent = float(inputs.get('recycled_content_percent', 0)) / 100
        
        # Processing method impact on circularity
        processing_bonus = {
            'Recycling': 0.3,
            'Electric': 0.1,
            'Conventional': 0.0,
            'Smelting': -0.05
        }.get(inputs.get('processing_method', 'Conventional'), 0.0)
        
        mci = base + (recycled_percent * 0.6) + processing_bonus
        mci = mci * (1 + np.random.normal(0, model.noise_factor))
        
        return max(0, min(1.0, mci))  # Constrain to 0-1 range
    
    def _predict_recycling_rate(self, inputs: Dict, material_type: str) -> float:
        """Predict end-of-life recycling rate (%)"""
        
        # Base recycling rates by material
        base_rates = {
            'steel': 85,
            'aluminum': 75,
            'copper': 80,
            'plastic': 25,
            'glass': 30,
            'generic': 50
        }
        
        base_rate = base_rates.get(material_type, 50)
        
        # Location impact on recycling infrastructure
        location_bonus = {
            'Europe': 15,
            'North America': 10,
            'Scandinavia': 20,
            'Global Average': 0,
            'Asia': -5,
            'China': 5
        }.get(inputs.get('location', 'Global Average'), 0)
        
        recycling_rate = base_rate + location_bonus
        recycling_rate = recycling_rate * (1 + np.random.normal(0, 0.1))
        
        return max(0, min(100, recycling_rate))
    
    def _calculate_circularity_score(self, predictions: Dict, inputs: Dict) -> float:
        """Calculate overall circularity score (0-100)"""
        
        mci = predictions['material_circularity_indicator']
        recycling_rate = predictions['end_of_life_recycling_rate_percent']
        recycled_content = float(inputs.get('recycled_content_percent', 0))
        
        # Weighted combination of circularity indicators
        score = (mci * 40) + (recycling_rate * 0.3) + (recycled_content * 0.3)
        score = min(100, score)
        
        return score
    
    def _calculate_renewable_potential(self, inputs: Dict, current_gwp: float) -> Optional[float]:
        """Calculate potential GWP reduction from switching to renewable energy"""
        
        current_energy = inputs.get('energy_source', 'Mixed Grid')
        
        # Reduction potential based on current energy source
        reduction_potential = {
            'Coal-dominated Grid': 60,
            'Mixed Grid': 30,
            'Mixed Grid (EU)': 20,
            'Natural Gas': 40,
            'Electricity': 30,
            'Nuclear': 10,
            'Renewable (Hydro)': 0,
            'Renewable (Solar)': 0,
            'Renewable (Wind)': 0
        }
        
        return reduction_potential.get(current_energy, 25)
    
    def _calculate_recycling_improvement_potential(self, current_mci: float) -> Optional[float]:
        """Calculate potential MCI improvement from increased recycling"""
        
        # Higher current MCI means lower improvement potential
        if current_mci > 0.8:
            return 5
        elif current_mci > 0.6:
            return 15
        elif current_mci > 0.4:
            return 25
        else:
            return 35
    
    def _extract_material_type(self, product_name: str) -> str:
        """Extract material type from product name (same as autofill.py)"""
        product_lower = product_name.lower()
        
        material_keywords = {
            'steel': ['steel', 'iron'],
            'aluminum': ['aluminum', 'aluminium'],
            'copper': ['copper'],
            'plastic': ['plastic', 'polymer', 'pvc', 'pet'],
            'glass': ['glass'],
            'concrete': ['concrete', 'cement'],
            'wood': ['wood', 'timber']
        }
        
        for material, keywords in material_keywords.items():
            if any(keyword in product_lower for keyword in keywords):
                return material
        
        return 'generic'
    
    def _initialize_models(self) -> Dict[str, PredictionModel]:
        """Initialize prediction models with realistic parameters"""
        
        return {
            'gwp': PredictionModel(
                base_value=800.0,
                coefficients={'energy': 0.6, 'transport': 0.2, 'recycling': -0.4},
                material_factors={
                    'steel': 2.2,
                    'aluminum': 10.5,
                    'copper': 1.5,
                    'plastic': 2.5,
                    'glass': 0.8,
                    'generic': 1.0
                },
                location_factors={
                    'Europe': 0.85,
                    'North America': 1.0,
                    'Asia': 1.4,
                    'China': 1.6,
                    'Global Average': 1.0,
                    'Scandinavia': 0.6,
                    'South America': 1.1
                },
                noise_factor=0.15
            ),
            
            'water': PredictionModel(
                base_value=8.0,
                coefficients={'processing': 0.5, 'location': 0.3},
                material_factors={
                    'steel': 1.2,
                    'aluminum': 2.8,
                    'copper': 1.8,
                    'plastic': 0.6,
                    'glass': 0.4,
                    'generic': 1.0
                },
                location_factors={
                    'Europe': 1.1,
                    'North America': 1.0,
                    'Asia': 1.3,
                    'Global Average': 1.0,
                    'South America': 0.8,
                    'Africa': 1.5
                },
                noise_factor=0.2
            ),
            
            'energy': PredictionModel(
                base_value=15.0,
                coefficients={'processing': 0.7, 'efficiency': -0.3},
                material_factors={
                    'steel': 2.0,
                    'aluminum': 6.5,
                    'copper': 1.8,
                    'plastic': 3.2,
                    'glass': 1.5,
                    'generic': 1.0
                },
                location_factors={
                    'Europe': 0.9,
                    'North America': 1.0,
                    'Asia': 1.1,
                    'Global Average': 1.0
                },
                noise_factor=0.18
            ),
            
            'mci': PredictionModel(
                base_value=0.2,
                coefficients={'recycled_content': 0.6, 'processing': 0.2},
                material_factors={
                    'steel': 1.3,
                    'aluminum': 1.2,
                    'copper': 1.25,
                    'plastic': 0.6,
                    'glass': 0.8,
                    'generic': 1.0
                },
                location_factors={
                    'Europe': 1.2,
                    'North America': 1.1,
                    'Global Average': 1.0,
                    'Asia': 0.9
                },
                noise_factor=0.1
            )
        }

# Test function
def test_predictions():
    """Test the prediction engine"""
    
    test_data = {
        "product_name": "Steel Beam",
        "process_route": "Iron Ore → Blast Furnace → Steel Production",
        "user_inputs": {
            "energy_source": "Mixed Grid",
            "transport_mode": "Rail",
            "transport_distance_km": 500.0,
            "recycled_content_percent": 30.0,
            "location": "Europe",
            "functional_unit": "1 kg Steel Beam",
            "raw_material_type": "Steel",
            "processing_method": "Conventional"
        }
    }
    
    engine = PredictionEngine()
    predictions = engine.generate_predictions(test_data)
    
    print("Test Input:")
    print(json.dumps(test_data, indent=2))
    print("\nPredictions:")
    print(json.dumps(predictions, indent=2))

if __name__ == "__main__":
    test_predictions()