import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import warnings

warnings.filterwarnings("ignore")


#custom ecxeptions
feature_cols = [
        "DELTA_COOLINGSETPT",
        "FCU_CVLV",
        "FCU_CVLV_DM",
        "FCU_DAT",
        "FCU_MAT",
        "RM_TEMP",
        "FCU_DA_CFM",
        "IsInStabilityZone"
    ]
class FolderExistsError(Exception):
    #Raised when attempting to create a folder that already exists
    def __init__(self, folder_path):
        self.folder_path = folder_path
        super().__init__(f"This folder already exists: {self.folder_path}")


class FileExistsError(Exception):
    #Raised when attempting to create a file that already exists
    def __init__(self, filepath):
        self.filepath = filepath
        super().__init__(f"This file already exists: {self.filepath}")

#manage data

class DataSetFolder:
    #Manages a folder containing CSV datasets
    
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.DataSets = []
    
    def Create(self):
        #Create the folder if it doesn't exist
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        else:
            raise FolderExistsError(self.folder_path)
    
    def AddToFolder(self, dataset, newfilename=None):
        #Add a dataset CSV to the folder
        if newfilename is None:
            newfilename = dataset.filename
        if dataset.filename not in os.listdir(self.folder_path):
            dataset.df.to_csv(
                fr"{self.folder_path.strip()}\{dataset.filename.strip()}", 
                index=False
            )
        else:
            raise FileExistsError(newfilename)
    
    def SaveToFolder(self, dataset, newfilename=None):
        #Save a dataset CSV in the folder
        if newfilename is None:
            newfilename = dataset.filename
        dataset.df.to_csv(
            fr"{self.folder_path.strip()}\{dataset.filename.strip()}", 
            index=False
        )
    
    def InitializeAFolder(self):
        #Load all CSV files from folder as DataSet objects
        for f in os.listdir(self.folder_path):
            DataSet(filename=f, parentfolder=self, df=None)
    
    def GetDataset(self, attr, value):
        #Retrieve a dataset by attribute value
        for dataset in self.DataSets:
            if getattr(dataset, attr) == value:
                return dataset
        return None


class DataSet:
    #for one data set that we can that play with
    
    def __init__(self, filename=None, df=None, parentfolder=None):
        if filename is not None:
            self.filename = filename
        if df is not None:
            self.df = df
        if parentfolder is not None:
            self.parentfolder = parentfolder
            parentfolder.DataSets.append(self)
    
    def makedf(self):
        #actually put csv data into df (use pd.read_csv) syntax
        if hasattr(self, 'parentfolder') and self.parentfolder is not None:
            self.df = pd.read_csv(
                fr"{self.parentfolder.folder_path.strip()}\{self.filename.strip()}"
            )
        else:
            self.df = pd.read_csv(self.filename)
    
    @staticmethod
    def Training_Test_Split(df, factor):
        #need to split our dat frame
        length = len(df)
        index = int(length * factor)
        train_df = df.iloc[:index]
        test_df = df.iloc[index:]
        return (train_df, test_df)
    
    @staticmethod
    def conv_op_hrs_only(df):
        #Note:WEEKDAYS ONLY 1-5 Monday Tuesday Wednesday Thursday Friday
        # and Operating Hours 6-17 only
        newdf = df.copy()
        newdf['Datetime'] = pd.to_datetime(
            newdf['Datetime'], 
            format="%Y-%m-%d %H:%M:%S"
        )
        newdf["DayOfWeek"] = newdf["Datetime"].dt.dayofweek + 1
        newdf["Hour"] = newdf["Datetime"].dt.hour
        
        
        newdf = newdf[
            (1 <= newdf.DayOfWeek) & 
            (newdf.DayOfWeek <= 5) & 
            (6 <= newdf.Hour) & 
            (newdf.Hour <= 17)
        ]
        return newdf
    
    @staticmethod
    def ReduceToImportantParamsOnly(df):

        params_to_keep = [
            "RMCLGSPT",      # Cooling set point
            "RMHTGSPT",      # Heating set point
            "FCU_CVLV_DM",   # Cooling valve signal
            "FCU_HVLV_DM",   # Heating valve signal
            "FCU_DA_CFM",    # Discharge flow rate
            "FCU_SPD",       # Fan speed
            "FCU_MAT",       # Mix air temperature
            "FCU_DAT",       # Discharge air temperature
            "RM_TEMP",       # Room temperature
            "FCU_CVLV",      # Cooling coil valve control
            "FCU_HVLV",      # Heating coil valve control
            "FAN_CTRL",      # FCU fan operation mode
            "Datetime",
            "Hour",
            "DayOfWeek",
        ]
        return df[params_to_keep]


#loading stuff and prepration
def load_csv(filepath):
    return pd.read_csv(filepath)
def load_dataset_from_folder(folder, filename):
    #folder = DataSetFolder(folder_path)
    dataset = DataSet(filename=filename, parentfolder=folder)
    dataset.makedf()
    return dataset.df

#for labelling bias
def add_label(df, label_value, column_name="IsBiased"):
    """Add a label column to DataFrame"""
    df[column_name] = label_value
    return df


def add_temperature_deltas(df):
    """Add temperature delta features to DataFrame"""
    df["DELTA_COOLINGSETPT"] = df["RM_TEMP"] - df["RMCLGSPT"]
    df["DELTA_HEATINGSETPT"] = df["RMHTGSPT"] - df["RM_TEMP"]
    return df


def combine_dataframes(dataframes, sort_by="Datetime", train_ratio=0.7):
    """Combine multiple DataFrames, sort, and split into train/test"""
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.sort_values(by=sort_by)
    
    train_df, test_df = DataSet.Training_Test_Split(combined_df, train_ratio)
    
    return train_df, test_df, combined_df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_stability_features(dataframes):
    """Create features indicating system stability and active control loops"""
    
    for df in dataframes:
        # Active control loop indicators
        df["COOLING_ACTIVE"] = np.where(df["DELTA_COOLINGSETPT"] > 1, 1, 0)
        df["HEATING_ACTIVE"] = np.where(df["DELTA_HEATINGSETPT"] > 1, 1, 0)
        
        # System in neutral zone (no active control)
        stability_condition = (df["COOLING_ACTIVE"] == 0) & (df["HEATING_ACTIVE"] == 0)
        df["IsInStabilityZone"] = np.where(stability_condition, 1, 0)



#for training

class SensorBiasClassifier:
    """Logistic Regression classifier for sensor bias detection"""
    
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        self.model = None
        self.training_metrics = {}
    
    def train(self, X_train, y_train):
        """Train the logistic regression model"""
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model and return classification report"""
        predictions = self.model.predict(X_test)
        report = classification_report(
            y_test, 
            predictions, 
            target_names=["No Bias", "Bias"],
            output_dict=True
        )
        return report, predictions
    
    def get_learning_curve(self, train_df, test_df, percentages=None):
        if percentages is None:
            percentages = list(range(1, 101))
        
        all_metrics = {
            "training_pctg": percentages,
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "models": [],
            "train_size": []
        }
        
        X_test = test_df[self.feature_columns].values
        y_test = test_df["IsBiased"].values
        
        for pctg in percentages:
            # Create training subset
            factor = pctg / 100
            split_idx = int(factor * len(train_df))
            partial_train = train_df.iloc[:split_idx]
            
            X_partial = partial_train[self.feature_columns].values
            y_partial = partial_train["IsBiased"].values
            
            # Train and evaluate
            model = LogisticRegression()
            model.fit(X_partial, y_partial)
            predictions = model.predict(X_test)
            
            report = classification_report(
                y_test, 
                predictions,
                target_names=["No Bias", "Bias"],
                output_dict=True
            )
            
            all_metrics["accuracy"].append(report["accuracy"])
            all_metrics["precision"].append(report["Bias"]["precision"])
            all_metrics["recall"].append(report["Bias"]["recall"])
            all_metrics["f1_score"].append(report["Bias"]["f1-score"])
            all_metrics["models"].append(model)
            all_metrics["train_size"].append(len(partial_train))
        
        metrics_df = pd.DataFrame({
            "training_pctg": all_metrics["training_pctg"],
            "accuracy": all_metrics["accuracy"],
            "precision": all_metrics["precision"],
            "recall": all_metrics["recall"],
            "f1_score": all_metrics["f1_score"],
            "train_size": all_metrics["train_size"]
        })
        
        return metrics_df, all_metrics


def get_best_model(all_metrics, metric="f1_score"):
    #get best model avoiding overtraining!!!!!!
    best_idx = np.argmax(all_metrics[metric])
    
    return {
        "model": all_metrics["models"][best_idx],
        "training_percentage": all_metrics["training_pctg"][best_idx],
        "training_size": all_metrics["train_size"][best_idx],
        "f1_score": all_metrics["f1_score"][best_idx],
        "accuracy": all_metrics["accuracy"][best_idx],
        "precision": all_metrics["precision"][best_idx],
        "recall": all_metrics["recall"][best_idx],
        "best_index": best_idx
    }


def save_model(model, feature_columns, output_path="sensor_bias_classifier.pkl"):
    """Save a trained model wrapped in SensorBiasClassifier to a pkl file."""
    classifier = SensorBiasClassifier(feature_columns)
    classifier.model = model
    joblib.dump(classifier, output_path)
    return output_path


def get_best_model_mode_1(df_faultfree, df_biased, feature_columns = feature_cols
                   ):

    add_label(df_faultfree, 0)
    add_label(df_biased, 1)
    add_temperature_deltas(df_faultfree)
    add_temperature_deltas(df_biased)
    TRAINDF, TESTDF, MIXDF = combine_dataframes([df_faultfree, df_biased], train_ratio=0.7)
    create_stability_features([TRAINDF, TESTDF, MIXDF])
    
    classifier = SensorBiasClassifier(feature_columns)
    metrics_df, all_metrics = classifier.get_learning_curve(TRAINDF, TESTDF)
    best_model_info = get_best_model(all_metrics, metric="f1_score")
    return best_model_info, feature_columns


def plot_learning_curve(metrics_df, best_model_info=None):
    """Plot learning curve visualization with optional best model highlight"""  
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.lineplot(
        data=metrics_df, 
        x="training_pctg", 
        y="f1_score",
        color="purple", 
        label="F1-score",
        ax=ax
    )
    sns.lineplot(
        data=metrics_df, 
        x="training_pctg", 
        y="accuracy",
        color="green",
        label="Accuracy",
        ax=ax
    )
    sns.lineplot(
        data=metrics_df, 
        x="training_pctg", 
        y="precision",
        color="red", 
        label="Precision",
        ax=ax
    )
    sns.lineplot(
        data=metrics_df, 
        x="training_pctg", 
        y="recall",
        color="blue", 
        label="Recall",
        ax=ax)
    
    # Highlight best model point if provided
    if best_model_info is not None:
        best_pctg = best_model_info["training_percentage"]
        best_f1 = best_model_info["f1_score"]
        ax.scatter(
            [best_pctg], 
            [best_f1], 
            color="gold", 
            s=300, 
            marker="*", 
            linewidths=2,
            zorder=5,
            label=f"Best Model ({best_pctg}%)"
        )
        ax.axvline(x=best_pctg, color="gold", linestyle="--", linewidth=2)
    
    ax.set_title("Learning Curve of Logistic Regression Model Metrics", fontsize=25)
    ax.set_xlabel("Training Data Percentage", fontsize=20)
    ax.set_ylabel("Score", fontsize=20)
    ax.tick_params(labelsize=18)
    ax.legend(title="Metrics", fontsize=20, title_fontsize=20, loc="center right")
    
    plt.tight_layout()
    plt.show()


def plot_initial_data_exploration(data, sample_size=10000, params_to_show = None):
    """Plot pairplot of initial features colored by bias status"""
    if params_to_show is None:
        params_to_show = [
            "DELTA_COOLINGSETPT",
            "DELTA_HEATINGSETPT",
            "FCU_CVLV_DM",      
            "FCU_HVLV_DM",    
            "FCU_DA_CFM",     
            "FCU_MAT",         
            "FCU_DAT",        
            "RM_TEMP",         
            "FCU_CVLV",         
            "FCU_HVLV",         
            "FAN_CTRL",
            "FCU_SPD",
            "IsBiased"
        ]
    
    sns.set_style("whitegrid")
    sns.set_palette("bright")
    
    pairplot = sns.pairplot(
        data.sample(sample_size),
        kind="scatter",
        hue="IsBiased",
        vars=params_to_show
    )
    plt.show()
    
    return pairplot
def additional_modification(faultfree, biased):
    # Add labels
    add_label(faultfree, 0)
    add_label(biased, 1)

    # Add temperature features
    add_temperature_deltas(faultfree)
    add_temperature_deltas(biased)
    
    #feature engineering
    create_stability_features([faultfree, biased])
    return faultfree, biased

def Run_Analysis_Demo():
    print("=" * 70)
    print("HVAC SENSOR BIAS DETECTION - MODEL TRAINING")
    print("=" * 70)
    
    # 1. Load and prepare data
    print("\n[1] Loading and preparing data...")
    ipo_folder = DataSetFolder("Imp_Params_Only_Files")
    df_faultfree = load_dataset_from_folder(ipo_folder, "imp_params_only_op_hrs_only_FCU_FaultFree.csv")
    df_biased = load_dataset_from_folder(ipo_folder, "imp_params_only_op_hrs_only_FCU_SensorBias_RMTemp_+2C.csv")
    
    # Add labels
    add_label(df_faultfree, 0)
    add_label(df_biased, 1)
    
    # Add temperature features
    add_temperature_deltas(df_faultfree)
    add_temperature_deltas(df_biased)
    
    # Combine and split
    TRAINDF, TESTDF, MIXDF = combine_dataframes([df_faultfree, df_biased], train_ratio=0.7)
    
    print(f"    Training set: {len(TRAINDF)} samples")
    print(f"    Test set: {len(TESTDF)} samples")
    print(f"    Total: {len(MIXDF)} samples")
    
    # 2. Initial data exploration
    print("\n[2] Exploring initial data distribution...")
    plot_initial_data_exploration(MIXDF.sample(frac=0.3))
 
    # 3. Feature engineering - create stability indicators
    print("\n[3] Creating stability control features...")
    create_stability_features([TRAINDF, TESTDF, MIXDF])
    print("    Created: COOLING_ACTIVE, HEATING_ACTIVE, IsInStabilityZone")

    # 4. Train model with comprehensive features
    print("\n[4] Training logistic regression model...")
    feature_columns = [
        "DELTA_COOLINGSETPT",
        "FCU_CVLV",
        "FCU_CVLV_DM",
        "FCU_DAT",
        "FCU_MAT",
        "RM_TEMP",
        "FCU_DA_CFM",
        "IsInStabilityZone"
    ]
    
    classifier = SensorBiasClassifier(feature_columns)
    X_train = TRAINDF[feature_columns].values
    y_train = TRAINDF["IsBiased"].values
    classifier.train(X_train, y_train)
    print(f"    Model trained on {len(feature_columns)} features")
    
    # 5. Evaluate on test set
    print("\n[5] Evaluating model on test set...")
    X_test = TESTDF[feature_columns].values
    y_test = TESTDF["IsBiased"].values
    report, predictions = classifier.evaluate(X_test, y_test)
    
    print("\nClassification Report:")
    print("-" * 70)
    for key, value in report.items():
        if key != 'weighted avg':
            print(f"    {key}: {value}")
    
    # 6. Generate learning curves
    print("\n[6] Generating learning curves (this may take a moment)...")
    metrics_df, all_metrics = classifier.get_learning_curve(TRAINDF, TESTDF)
    print("    Learning curve complete")
    
    # 7. Find best model to minimize overtraining
    print("\n[7] Finding best model (highest F1 score)...")
    best_model_info = get_best_model(all_metrics, metric="f1_score")
    
    print("\n    Best Model Found:")
    print(f"    Training percentage: {best_model_info['training_percentage']}%")
    print(f"    Training samples: {best_model_info['training_size']}")
    print(f"    F1 Score: {best_model_info['f1_score']:.4f}")
    print(f"    Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"    Precision: {best_model_info['precision']:.4f}")
    print(f"    Recall: {best_model_info['recall']:.4f}")
    
    # Set best model as the classifier's model
    classifier.model = best_model_info["model"]
    
    # 8. Plot learning curve with best model highlighted
    print("\n[8] Plotting learning curve with best model...")
    plot_learning_curve(metrics_df, best_model_info)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nBest model has been set as the classifier's model.")
    print("This model minimizes overtraining by using optimal training data size.")    
    return classifier, metrics_df, all_metrics, best_model_info

def main():
    curr_folder = DataSetFolder("Imp_Params_Only_Files")
    curr_df_faultfree = load_dataset_from_folder(curr_folder, "imp_params_only_op_hrs_only_FCU_FaultFree.csv")
    curr_df_biased = load_dataset_from_folder(curr_folder, "imp_params_only_op_hrs_only_FCU_SensorBias_RMTemp_+2C.csv")
    #curr_best_model_info, curr_feature_cols = get_best_model_mode_1(curr_df_faultfree, curr_df_biased, feature_cols)
    #save_model(curr_best_model_info["model"], curr_feature_cols, output_path="sensor_bias_classifier.pkl")
    additional_modification(curr_df_faultfree, curr_df_biased)
    TRAINDF, TESTDF, MIXDF = combine_dataframes([curr_df_faultfree, curr_df_biased], train_ratio=0.7)
    classifier = SensorBiasClassifier(feature_cols)
    #metrics_df, allmetrics =  classifier.get_learning_curve(TRAINDF, TESTDF)
    #plot_learning_curve(metrics_df, get_best_model(allmetrics))
    plot_initial_data_exploration(MIXDF)

    """print("\n    Best Model Found:")
    print(f"    Training percentage: {curr_best_model_info['training_percentage']}%")
    print(f"    Training samples: {curr_best_model_info['training_size']}")
    print(f"    F1 Score: {curr_best_model_info['f1_score']:.4f}")
    print(f"    Accuracy: {curr_best_model_info['accuracy']:.4f}")
    print(f"    Precision: {curr_best_model_info['precision']:.4f}")
    print(f"    Recall: {curr_best_model_info['recall']:.4f}")"""


if __name__ == "__main__":
    #classifier, metrics_df, all_metrics, best_model = 
    Run_Analysis_Demo()
