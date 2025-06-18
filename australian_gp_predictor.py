import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')


class MonacoGPPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_trained = False

    def create_sample_data(self):
        """Create comprehensive sample historical data for Monaco GP"""
        np.random.seed(42)

        # Sample drivers from recent F1 seasons
        drivers = ['Hamilton', 'Verstappen', 'Leclerc', 'Russell', 'Sainz', 'Norris',
                   'Piastri', 'Alonso', 'Stroll', 'Ocon', 'Gasly', 'Albon', 'Bottas',
                   'Zhou', 'Magnussen', 'Hulkenberg', 'Tsunoda', 'Ricciardo', 'Perez', 'Vettel']

        # Teams with their relative competitiveness
        team_strength = {
            'Mercedes': 0.85, 'Red Bull': 0.95, 'Ferrari': 0.80, 'McLaren': 0.75,
            'Alpine': 0.60, 'Aston Martin': 0.70, 'Williams': 0.45, 'Alfa Romeo': 0.50,
            'Haas': 0.40, 'AlphaTauri': 0.55
        }

        driver_teams = {
            'Hamilton': 'Mercedes', 'Russell': 'Mercedes',
            'Verstappen': 'Red Bull', 'Perez': 'Red Bull',
            'Leclerc': 'Ferrari', 'Sainz': 'Ferrari',
            'Norris': 'McLaren', 'Piastri': 'McLaren',
            'Alonso': 'Aston Martin', 'Stroll': 'Aston Martin',
            'Ocon': 'Alpine', 'Gasly': 'Alpine',
            'Albon': 'Williams', 'Sargeant': 'Williams',
            'Bottas': 'Alfa Romeo', 'Zhou': 'Alfa Romeo',
            'Magnussen': 'Haas', 'Hulkenberg': 'Haas',
            'Tsunoda': 'AlphaTauri', 'Ricciardo': 'AlphaTauri',
            'Vettel': 'Aston Martin'
        }

        data = []
        years = list(range(2015, 2024))

        for year in years:
            # Simulate 20 drivers per race
            race_drivers = np.random.choice(drivers, size=20, replace=False)

            for i, driver in enumerate(race_drivers):
                team = driver_teams.get(driver, np.random.choice(list(team_strength.keys())))

                # Monaco-specific features
                monaco_experience = np.random.randint(0, 15)  # Years racing at Monaco
                street_circuit_wins = np.random.randint(0, 8)  # Wins on street circuits

                # Qualifying performance (crucial for Monaco)
                base_quali = team_strength[team] + np.random.normal(0, 0.1)
                quali_pos = max(1, min(20, int(np.random.exponential(3) * (1 - base_quali) + 1)))

                # Practice times (normalized)
                fp1_time = np.random.normal(90, 2)  # Base time around 1:30
                fp2_time = np.random.normal(89, 1.5)
                fp3_time = np.random.normal(88.5, 1)

                # Weather and track conditions
                weather_clear = np.random.choice([0, 1], p=[0.3, 0.7])
                track_temp = np.random.normal(35, 8)

                # Driver form (recent performance)
                recent_points = np.random.poisson(team_strength[team] * 15)
                recent_podiums = np.random.poisson(team_strength[team] * 3)

                # Monaco historical performance
                monaco_prev_finish = np.random.randint(1, 21)
                monaco_crashes = np.random.poisson(0.5)  # Monaco is crash-prone

                # Car setup and reliability
                car_reliability = team_strength[team] + np.random.normal(0, 0.05)
                downforce_level = np.random.uniform(0.7, 1.0)  # Monaco needs high downforce

                # Strategy factors
                tire_strategy = np.random.choice(['conservative', 'aggressive', 'medium'])
                pit_stop_avg = np.random.normal(3.2, 0.3)  # Pit stop times

                # Determine winner (simplified logic based on multiple factors)
                win_probability = (
                                          team_strength[team] * 0.4 +
                                          (21 - quali_pos) / 20 * 0.3 +  # Qualifying crucial at Monaco
                                          (monaco_experience / 15) * 0.1 +
                                          (street_circuit_wins / 8) * 0.1 +
                                          car_reliability * 0.1
                                  ) + np.random.normal(0, 0.1)

                won_race = 1 if win_probability > 0.75 else 0

                # Ensure only one winner per year
                if year not in [d['year'] for d in data if d['won_race'] == 1]:
                    if i == 0:  # First driver has chance to win
                        won_race = 1 if win_probability > 0.6 else 0
                else:
                    won_race = 0

                data.append({
                    'year': year,
                    'driver': driver,
                    'team': team,
                    'qualifying_position': quali_pos,
                    'fp1_time': fp1_time,
                    'fp2_time': fp2_time,
                    'fp3_time': fp3_time,
                    'monaco_experience': monaco_experience,
                    'street_circuit_wins': street_circuit_wins,
                    'recent_points': recent_points,
                    'recent_podiums': recent_podiums,
                    'monaco_prev_finish': monaco_prev_finish,
                    'monaco_crashes': monaco_crashes,
                    'car_reliability': car_reliability,
                    'downforce_level': downforce_level,
                    'weather_clear': weather_clear,
                    'track_temperature': track_temp,
                    'tire_strategy': tire_strategy,
                    'pit_stop_avg': pit_stop_avg,
                    'won_race': won_race
                })

        # Ensure each year has exactly one winner
        df = pd.DataFrame(data)
        for year in years:
            year_data = df[df['year'] == year]
            if year_data['won_race'].sum() == 0:
                # Pick the driver with best qualifying
                best_quali_idx = year_data['qualifying_position'].idxmin()
                df.loc[best_quali_idx, 'won_race'] = 1
            elif year_data['won_race'].sum() > 1:
                # Keep only the first winner
                winners = year_data[year_data['won_race'] == 1].index
                df.loc[winners[1:], 'won_race'] = 0

        return df

    def prepare_features(self, df):
        """Prepare and engineer features for the model"""
        # Create derived features
        df['quali_advantage'] = 21 - df['qualifying_position']  # Higher is better
        df['practice_consistency'] = df[['fp1_time', 'fp2_time', 'fp3_time']].std(axis=1)
        df['best_practice_time'] = df[['fp1_time', 'fp2_time', 'fp3_time']].min(axis=1)
        df['form_score'] = df['recent_points'] + df['recent_podiums'] * 5
        df['monaco_specialist'] = (df['monaco_experience'] > 5).astype(int)
        df['street_circuit_expert'] = (df['street_circuit_wins'] > 2).astype(int)

        # Encode categorical variables
        df['tire_strategy_encoded'] = df['tire_strategy'].map({
            'conservative': 0, 'medium': 1, 'aggressive': 2
        })

        # Team encoding (using label encoder)
        df['team_encoded'] = self.label_encoder.fit_transform(df['team'])

        # Driver encoding
        driver_le = LabelEncoder()
        df['driver_encoded'] = driver_le.fit_transform(df['driver'])

        return df

    def select_features(self, df):
        """Select the most relevant features for Monaco GP prediction"""
        feature_cols = [
            'qualifying_position', 'quali_advantage', 'monaco_experience',
            'street_circuit_wins', 'recent_points', 'recent_podiums',
            'monaco_prev_finish', 'monaco_crashes', 'car_reliability',
            'downforce_level', 'weather_clear', 'track_temperature',
            'tire_strategy_encoded', 'pit_stop_avg', 'practice_consistency',
            'best_practice_time', 'form_score', 'monaco_specialist',
            'street_circuit_expert', 'team_encoded'
        ]

        return df[feature_cols]

    def train_models(self, X, y):
        """Train multiple models and find the best one"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models to try
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }

        best_score = 0
        best_model_name = None

        print("Training and evaluating models...")
        print("-" * 50)

        for name, model in models.items():
            if name == 'LogisticRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name}: {accuracy:.4f}")

            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name
                if name == 'LogisticRegression':
                    self.models[name] = (model, True)  # Needs scaling
                else:
                    self.models[name] = (model, False)  # No scaling needed

        print(f"\nBest model: {best_model_name} (Accuracy: {best_score:.4f})")

        # Store the best model
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name][0]
        self.needs_scaling = self.models[best_model_name][1]

        # Feature importance for tree-based models
        if best_model_name in ['RandomForest', 'GradientBoosting']:
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nTop 10 Most Important Features for {best_model_name}:")
            print(feature_importance.head(10))

        return best_score

    def fit(self, df=None):
        """Train the complete model"""
        if df is None:
            print("Creating sample data...")
            df = self.create_sample_data()

        print(f"Dataset shape: {df.shape}")
        print(f"Winners in dataset: {df['won_race'].sum()}")

        # Prepare features
        df = self.prepare_features(df)
        X = self.select_features(df)
        y = df['won_race']

        self.feature_columns = X.columns.tolist()

        # Train models
        best_accuracy = self.train_models(X, y)
        self.is_trained = True

        return best_accuracy

    def predict_winner(self, race_data):
        """Predict the winner for a given race scenario"""
        if not self.is_trained:
            raise ValueError("Model must be trained first. Call fit() method.")

        # Convert to DataFrame if it's a dictionary
        if isinstance(race_data, dict):
            race_data = pd.DataFrame([race_data])
        elif isinstance(race_data, list):
            race_data = pd.DataFrame(race_data)

        # Prepare features
        race_data = self.prepare_features(race_data)
        X = race_data[self.feature_columns]

        # Make predictions
        if self.needs_scaling:
            X_scaled = self.scaler.transform(X)
            probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = self.best_model.predict_proba(X)[:, 1]

        # Create results DataFrame
        results = race_data[['driver', 'team']].copy()
        results['win_probability'] = probabilities
        results = results.sort_values('win_probability', ascending=False)

        return results

    def simulate_race(self):
        """Simulate a Monaco GP race with current drivers"""
        print("Simulating Monaco GP 2024...")

        # Current F1 grid (simplified)
        current_drivers = [
            {'driver': 'Verstappen', 'team': 'Red Bull', 'qualifying_position': 1},
            {'driver': 'Leclerc', 'team': 'Ferrari', 'qualifying_position': 2},
            {'driver': 'Hamilton', 'team': 'Mercedes', 'qualifying_position': 3},
            {'driver': 'Norris', 'team': 'McLaren', 'qualifying_position': 4},
            {'driver': 'Russell', 'team': 'Mercedes', 'qualifying_position': 5},
            {'driver': 'Sainz', 'team': 'Ferrari', 'qualifying_position': 6},
            {'driver': 'Perez', 'team': 'Red Bull', 'qualifying_position': 7},
            {'driver': 'Piastri', 'team': 'McLaren', 'qualifying_position': 8},
            {'driver': 'Alonso', 'team': 'Aston Martin', 'qualifying_position': 9},
            {'driver': 'Stroll', 'team': 'Aston Martin', 'qualifying_position': 10}
        ]

        # Add realistic features for each driver
        for driver_data in current_drivers:
            driver_data.update({
                'year': 2024,
                'fp1_time': np.random.normal(88, 1),
                'fp2_time': np.random.normal(87.5, 0.8),
                'fp3_time': np.random.normal(87, 0.5),
                'monaco_experience': np.random.randint(3, 12),
                'street_circuit_wins': np.random.randint(0, 5),
                'recent_points': np.random.randint(50, 200),
                'recent_podiums': np.random.randint(1, 8),
                'monaco_prev_finish': np.random.randint(1, 15),
                'monaco_crashes': np.random.randint(0, 3),
                'car_reliability': np.random.uniform(0.7, 0.95),
                'downforce_level': np.random.uniform(0.8, 1.0),
                'weather_clear': 1,
                'track_temperature': 32,
                'tire_strategy': np.random.choice(['conservative', 'medium', 'aggressive']),
                'pit_stop_avg': np.random.normal(3.0, 0.2),
                'won_race': 0  # We're predicting this
            })

        # Predict winner
        predictions = self.predict_winner(current_drivers)

        print("\nMonaco GP 2024 - Winner Predictions:")
        print("=" * 50)
        for i, (_, row) in enumerate(predictions.head(10).iterrows()):
            print(f"{i + 1:2d}. {row['driver']:12s} ({row['team']:12s}) - {row['win_probability']:.1%}")

        return predictions


# Example usage and demonstration
if __name__ == "__main__":
    # Create and train the model
    predictor = MonacoGPPredictor()

    print("F1 Monaco GP Winner Prediction Model")
    print("=" * 40)

    # Train the model
    accuracy = predictor.fit()

    print(f"\nModel trained successfully!")
    print(f"Cross-validation accuracy: {accuracy:.1%}")

    # Simulate a race
    predictions = predictor.simulate_race()

    # Show feature importance if available
    if hasattr(predictor, 'best_model') and hasattr(predictor.best_model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        feature_importance = pd.DataFrame({
            'feature': predictor.feature_columns,
            'importance': predictor.best_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)

        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('Top 10 Most Important Features for Monaco GP Winner Prediction')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()
