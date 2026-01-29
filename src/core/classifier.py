"""
Machine Learning classifier for understory separation.
Uses Random Forest trained on geometric features.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import pickle
from pathlib import Path


@dataclass
class ClassifierResult:
    """Result of ML classification."""
    is_tree: np.ndarray           # (N,) boolean mask (True = tree)
    probabilities: np.ndarray     # (N,) probability of being tree
    n_tree: int
    n_understory: int


@dataclass
class TrainingData:
    """Training data for classifier."""
    features: np.ndarray          # (N, n_features) feature matrix
    labels: np.ndarray            # (N,) labels (0=understory, 1=tree)
    feature_names: list           # Names of features


def prepare_features(
    verticality: np.ndarray,
    linearity: np.ndarray,
    sphericity: np.ndarray,
    dist_to_ground: np.ndarray,
    planarity: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, list]:
    """
    Prepare feature matrix for classification.
    
    Returns
    -------
    features : np.ndarray
        (N, n_features) feature matrix
    feature_names : list
        Names of features in order
    """
    feature_list = [verticality, linearity, sphericity, dist_to_ground]
    feature_names = ['verticality', 'linearity', 'sphericity', 'dist_to_ground']
    
    if planarity is not None:
        feature_list.append(planarity)
        feature_names.append('planarity')
    
    features = np.column_stack(feature_list)
    return features, feature_names


def train_classifier(
    training_data: TrainingData,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
    verbose: bool = False
):
    """
    Train Random Forest classifier on labeled data.
    
    Parameters
    ----------
    training_data : TrainingData
        Training data with features and labels.
    n_estimators : int
        Number of trees in forest.
    max_depth : int
        Maximum depth of trees.
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        Print progress.
    
    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Trained classifier.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    if verbose:
        print(f"Training Random Forest classifier...")
        print(f"  Samples: {len(training_data.labels):,}")
        print(f"  Features: {len(training_data.feature_names)}")
        print(f"  Class balance: tree={np.sum(training_data.labels):,}, understory={np.sum(~training_data.labels.astype(bool)):,}")
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Cross-validation score
    if verbose:
        scores = cross_val_score(clf, training_data.features, training_data.labels, cv=5)
        print(f"  Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
    
    # Train on full data
    clf.fit(training_data.features, training_data.labels)
    
    if verbose:
        print(f"  ✓ Training complete")
        print(f"  Feature importances:")
        for name, imp in sorted(zip(training_data.feature_names, clf.feature_importances_), key=lambda x: -x[1]):
            print(f"    {name}: {imp:.3f}")
    
    return clf


def classify_understory_ml(
    features: np.ndarray,
    classifier,
    probability_threshold: float = 0.5,
    verbose: bool = False
) -> ClassifierResult:
    """
    Classify points using trained ML model.
    
    Parameters
    ----------
    features : np.ndarray
        (N, n_features) feature matrix.
    classifier : sklearn classifier
        Trained classifier with predict_proba method.
    probability_threshold : float
        Threshold for classification (default 0.5).
    verbose : bool
        Print progress.
    
    Returns
    -------
    ClassifierResult
        Classification results.
    """
    if verbose:
        print(f"Classifying {len(features):,} points...")
    
    # Get probabilities
    probas = classifier.predict_proba(features)
    tree_proba = probas[:, 1]  # Probability of class 1 (tree)
    
    # Apply threshold
    is_tree = tree_proba >= probability_threshold
    
    n_tree = np.sum(is_tree)
    n_understory = len(is_tree) - n_tree
    
    if verbose:
        print(f"  Tree: {n_tree:,} ({100*n_tree/len(is_tree):.1f}%)")
        print(f"  Understory: {n_understory:,} ({100*n_understory/len(is_tree):.1f}%)")
    
    return ClassifierResult(
        is_tree=is_tree,
        probabilities=tree_proba,
        n_tree=n_tree,
        n_understory=n_understory
    )


def save_classifier(classifier, filepath: str, feature_names: list):
    """Save trained classifier to file."""
    data = {
        'classifier': classifier,
        'feature_names': feature_names
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Classifier saved to {filepath}")


def load_classifier(filepath: str):
    """Load trained classifier from file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['classifier'], data['feature_names']


def load_training_data_from_laz(
    filepath: str,
    label_field: str = 'training_label',
    verbose: bool = False
) -> TrainingData:
    """
    Load training data from LAZ file with labels.
    
    The LAZ file should have:
    - verticality, linearity, sphericity as extra dimensions
    - A label field (default: 'training_label') with 0=understory, 1=tree
    
    Parameters
    ----------
    filepath : str
        Path to LAZ file with labeled data.
    label_field : str
        Name of the label field in LAZ file.
    verbose : bool
        Print progress.
    
    Returns
    -------
    TrainingData
        Training data ready for classifier.
    """
    import laspy
    
    las = laspy.read(filepath)
    
    # Get coordinates for dist_to_ground
    xyz = np.column_stack([las.x, las.y, las.z])
    dist_to_ground = xyz[:, 2]  # Z is height above ground (normalized)
    
    # Get features
    verticality = np.array(las.verticality)
    linearity = np.array(las.linearity)
    sphericity = np.array(las.sphericity)
    
    # Get labels
    labels = np.array(getattr(las, label_field)).astype(int)
    
    # Prepare features
    features, feature_names = prepare_features(
        verticality, linearity, sphericity, dist_to_ground
    )
    
    if verbose:
        print(f"Loaded training data from {filepath}")
        print(f"  Points: {len(labels):,}")
        print(f"  Tree (1): {np.sum(labels == 1):,}")
        print(f"  Understory (0): {np.sum(labels == 0):,}")
    
    return TrainingData(
        features=features,
        labels=labels,
        feature_names=feature_names
    )


@dataclass
class SliceClassifierResult:
    """Result of slice-based ML classification."""
    is_tree: np.ndarray           # (N,) boolean mask (True = tree)
    probabilities: np.ndarray     # (N,) probability of being tree (1.0 for protected zone)
    n_tree: int
    n_understory: int
    n_protected: int              # Points in upper zone (100% protected)
    n_ml_classified: int          # Points classified by ML


def slice_classify_understory(
    features: np.ndarray,
    dist_to_ground: np.ndarray,
    classifier,
    slice_height: float = 3.5,
    probability_threshold: float = 0.5,
    verbose: bool = False
) -> SliceClassifierResult:
    """
    Classify understory using ML, but ONLY on the lower slice.
    
    Upper zone (>=slice_height) is 100% protected and never enters ML.
    Only the lower zone (<slice_height) is classified by ML.
    
    Parameters
    ----------
    features : np.ndarray
        (N, n_features) feature matrix for ALL points.
    dist_to_ground : np.ndarray
        Height above ground for each point.
    classifier : sklearn classifier
        Trained classifier with predict_proba method.
    slice_height : float
        Height threshold. Points >= this are protected.
    probability_threshold : float
        Threshold for ML classification (default 0.5).
    verbose : bool
        Print progress.
    
    Returns
    -------
    SliceClassifierResult
        Classification results with protected upper zone.
    """
    n_points = len(features)
    
    # Separate zones
    lower_mask = dist_to_ground < slice_height
    upper_mask = ~lower_mask
    
    n_lower = np.sum(lower_mask)
    n_upper = np.sum(upper_mask)
    
    if verbose:
        print(f"Slice-based ML classification (threshold: {slice_height}m)...")
        print(f"  Upper zone (protected): {n_upper:,} points")
        print(f"  Lower zone (ML): {n_lower:,} points")
    
    # Initialize: upper zone = all tree (protected)
    is_tree = np.ones(n_points, dtype=bool)
    probabilities = np.ones(n_points, dtype=np.float32)
    
    # Apply ML only to lower zone
    if n_lower > 0:
        lower_features = features[lower_mask]
        
        # Get probabilities from ML
        probas = classifier.predict_proba(lower_features)
        tree_proba = probas[:, 1]  # Probability of class 1 (tree)
        
        # Apply threshold
        is_tree_lower = tree_proba >= probability_threshold
        
        # Update masks for lower zone
        is_tree[lower_mask] = is_tree_lower
        probabilities[lower_mask] = tree_proba
        
        n_understory_lower = np.sum(~is_tree_lower)
        if verbose:
            print(f"  ML removed {n_understory_lower:,} understory points from lower zone")
    
    n_tree = np.sum(is_tree)
    n_understory = n_points - n_tree
    
    if verbose:
        print(f"  Final: Tree={n_tree:,}, Understory={n_understory:,}")
    
    return SliceClassifierResult(
        is_tree=is_tree,
        probabilities=probabilities,
        n_tree=n_tree,
        n_understory=n_understory,
        n_protected=n_upper,
        n_ml_classified=n_lower
    )

