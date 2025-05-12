import numpy as np
import pandas as pd
import pytest
from prediction.short_term_outcome_prediction.timeseries_decomposition import decompose_and_label_timeseries


def test_decompose_and_label_timeseries_basic():
    """Test basic functionality with default parameters"""
    # Create test data - shape: (num_samples, num_features, num_timesteps, 1)
    timeseries = np.zeros((2, 10, 1, 1))
    
    # Set patient IDs
    timeseries[0, 0, 0, 0] = 100  # First patient ID: 100
    timeseries[1, 0, 0, 0] = 200  # Second patient ID: 200
    
    # Create events dataframe - patient 100 has event at t=5, patient 200 has event at t=8
    y_df = pd.DataFrame({
        'case_admission_id': [100, 200],
        'relative_sample_date_hourly_cat': [5, 8]
    })
    
    # Run function with default parameters
    map, flat_labels = decompose_and_label_timeseries(timeseries, y_df)
    
    # Check lengths
    assert len(map) == 20  # 10 timesteps for each of 2 patients
    assert len(flat_labels) == 20
    
    # Check map values
    assert map[0] == (0, 0)  # First element is (patient_idx=0, timestep=0)
    assert map[10] == (1, 0)  # Element 11 is (patient_idx=1, timestep=0)
    
    # Check labels for patient 1 (event at t=5)
    assert flat_labels[0] == 1  # At t=0, event at t=5 is within interval (0, 6]
    assert flat_labels[5] == 0  # At t=5, no event within interval (5, 11]
    
    # Check labels for patient 2 (event at t=8)
    assert flat_labels[12] == 1  # At t=2, event at t=8 is within interval (2, 8]
    assert flat_labels[18] == 0  # At t=8, no event within interval (8, 14]

def test_decompose_and_label_timeseries_no_interval():
    """Test with target_interval=False (exact timestep prediction)"""
    timeseries = np.zeros((2, 10, 1, 1))
    timeseries[0, 0, 0, 0] = 100
    timeseries[1, 0, 0, 0] = 200
    
    y_df = pd.DataFrame({
        'case_admission_id': [100, 200],
        'relative_sample_date_hourly_cat': [5, 8]
    })
    
    map, flat_labels = decompose_and_label_timeseries(timeseries, y_df, target_interval=False)
    
    # For patient 1 with event at t=5, no timestep should predict it exactly with target_time_to_outcome=6
    patient1_labels = [flat_labels[i] for i in range(10)]
    assert sum(patient1_labels) == 0
    
    # For patient 2 with event at t=8, timestep 2 should predict it (2+6=8)
    assert flat_labels[12] == 1  # At t=2, event at exactly t=2+6=8
    assert flat_labels[13] == 0  # At t=3, no event at exactly t=3+6=9

def test_decompose_and_label_timeseries_restrict_to_first_event():
    """Test with restrict_to_first_event=True"""
    timeseries = np.zeros((2, 10, 1, 1))
    timeseries[0, 0, 0, 0] = 100
    timeseries[1, 0, 0, 0] = 200
    
    # Patient 100 has events at t=5 and t=7
    multiple_events_df = pd.DataFrame({
        'case_admission_id': [100, 100, 200],
        'relative_sample_date_hourly_cat': [5, 7, 8]
    })
    
    map, flat_labels = decompose_and_label_timeseries(
        timeseries, multiple_events_df, restrict_to_first_event=True
    )
    
    # For patient 1, only consider timesteps up to the first event (t=5)
    patient1_timesteps = [m[1] for m in map if m[0] == 0]
    assert max(patient1_timesteps) == 4  # Last timestep is 4 (we go up to but not including 5)
    assert len(patient1_timesteps) == 5  # 5 timesteps for patient 1 (0,1,2,3,4)
    
    # For patient 2, consider timesteps up to t=8
    patient2_timesteps = [m[1] for m in map if m[0] == 1]
    assert max(patient2_timesteps) == 7  # Last timestep is 7 (we go up to but not including 8)
    assert len(patient2_timesteps) == 8  # 8 timesteps for patient 2

def test_decompose_and_label_timeseries_no_events():
    """Test with no events"""
    timeseries = np.zeros((2, 10, 1, 1))
    timeseries[0, 0, 0, 0] = 100
    timeseries[1, 0, 0, 0] = 200
    
    # Create a DataFrame with no matching case_admission_ids
    no_events_df = pd.DataFrame({
        'case_admission_id': [300, 400],
        'relative_sample_date_hourly_cat': [5, 8]
    })
    
    map, flat_labels = decompose_and_label_timeseries(timeseries, no_events_df)
    
    # All labels should be 0 since no events match our patients
    assert sum(flat_labels) == 0
    assert len(map) == 20  # Still have all timesteps

def test_decompose_and_label_timeseries_multiple_events():
    """Test with multiple events per patient"""
    timeseries = np.zeros((2, 10, 10, 1))
    timeseries[0, 0, 0, 0] = 100
    timeseries[1, 0, 0, 0] = 200
    
    # Patient 100 has events at t=5 and t=7
    multiple_events_df = pd.DataFrame({
        'case_admission_id': [100, 100, 200],
        'relative_sample_date_hourly_cat': [5, 7, 8]
    })
    
    map, flat_labels = decompose_and_label_timeseries(timeseries, multiple_events_df)
    
    # Check specific labels for patient with multiple events
    assert flat_labels[0] == 1  # At t=0, event at t=5 is within interval (0, 6]
    assert flat_labels[2] == 1  # At t=2, event at t=7 is within interval (2, 8]
    assert flat_labels[5] == 1  # At t=5, event at t=7 is within interval (5, 11]
    assert flat_labels[7] == 0  # At t=7, no event within interval (7, 13]

import numpy as np
import pandas as pd
import pytest
from prediction.short_term_outcome_prediction.timeseries_decomposition import decompose_and_label_timeseries_time_to_event

def test_decompose_and_label_timeseries_time_to_event_basic():
    """Test basic functionality with default parameters"""
    # Create test data - shape: (num_samples, num_features, num_timesteps, 1)
    timeseries = np.zeros((2, 10, 1, 1))
    
    # Set patient IDs
    timeseries[0, 0, 0, 0] = 100  # First patient ID: 100
    timeseries[1, 0, 0, 0] = 200  # Second patient ID: 200
    
    # Create events dataframe - patient 100 has event at t=5, patient 200 has event at t=8
    y_df = pd.DataFrame({
        'case_admission_id': [100, 200],
        'relative_sample_date_hourly_cat': [5, 8]
    })
    
    # Run function with default parameters
    map, flat_labels = decompose_and_label_timeseries_time_to_event(timeseries, y_df)
    
    # Check lengths
    assert len(map) == 20  # 10 timesteps for each of 2 patients
    assert len(flat_labels) == 20
    
    # Check map values
    assert map[0] == (0, 0)  # First element is (patient_idx=0, timestep=0)
    assert map[10] == (1, 0)  # Element 11 is (patient_idx=1, timestep=0)
    
    # Check time to event labels (patient 1, event at t=5)
    assert flat_labels[0] == 5  # At t=0, time to next event is 5
    assert flat_labels[4] == 1  # At t=4, time to next event is 1
    assert flat_labels[5] == 10  # At t=5, no more events, returns overall_max_ts (10)
    
    # Check time to event labels (patient 2, event at t=8)
    assert flat_labels[10] == 8  # At t=0, time to next event is 8
    assert flat_labels[17] == 1  # At t=7, time to next event is 1
    assert flat_labels[18] == 10  # At t=8, no more events, returns overall_max_ts (10)

def test_decompose_and_label_timeseries_time_to_event_no_events():
    """Test with no events"""
    timeseries = np.zeros((2, 10, 1, 1))
    timeseries[0, 0, 0, 0] = 100
    timeseries[1, 0, 0, 0] = 200
    
    # Create a DataFrame with no matching case_admission_ids
    no_events_df = pd.DataFrame({
        'case_admission_id': [300, 400],
        'relative_sample_date_hourly_cat': [5, 8]
    })
    
    map, flat_labels = decompose_and_label_timeseries_time_to_event(timeseries, no_events_df)
    
    # All labels should be the maximum timestep (10)
    assert sum(flat_labels) == 10 * len(flat_labels)
    assert len(map) == 20  # Still have all timesteps
    
    # Check specific values
    assert flat_labels[0] == 10  # Patient 1, t=0, censored at max_ts
    assert flat_labels[10] == 10  # Patient 2, t=0, censored at max_ts

def test_decompose_and_label_timeseries_time_to_event_multiple_events():
    """Test with multiple events per patient"""
    timeseries = np.zeros((2, 10, 1, 1))
    timeseries[0, 0, 0, 0] = 100
    timeseries[1, 0, 0, 0] = 200
    
    # Patient 100 has events at t=3, t=5, and t=7
    # Patient 200 has event at t=8
    multiple_events_df = pd.DataFrame({
        'case_admission_id': [100, 100, 100, 200],
        'relative_sample_date_hourly_cat': [3, 5, 7, 8]
    })
    
    map, flat_labels = decompose_and_label_timeseries_time_to_event(timeseries, multiple_events_df)
    
    # Check specific labels for patient with multiple events
    assert flat_labels[0] == 3  # At t=0, next event at t=3
    assert flat_labels[2] == 1  # At t=2, next event at t=3
    assert flat_labels[3] == 2  # At t=3, next event at t=5
    assert flat_labels[5] == 2  # At t=5, next event at t=7
    assert flat_labels[7] == 10  # At t=7, no more events, returns max_ts (10)
    
    # Check labels for patient with one event
    assert flat_labels[10] == 8  # At t=0, next event at t=8
    assert flat_labels[17] == 1  # At t=7, next event at t=8

def test_decompose_and_label_timeseries_time_to_event_restrict_to_first_event():
    """Test with restrict_to_first_event=True"""
    timeseries = np.zeros((2, 10, 1, 1))
    timeseries[0, 0, 0, 0] = 100
    timeseries[1, 0, 0, 0] = 200
    
    # Patient 100 has events at t=3 and t=7
    # Patient 200 has event at t=8
    multiple_events_df = pd.DataFrame({
        'case_admission_id': [100, 100, 200],
        'relative_sample_date_hourly_cat': [3, 7, 8]
    })
    
    map, flat_labels = decompose_and_label_timeseries_time_to_event(timeseries, multiple_events_df, restrict_to_first_event=True)
    
    # For patient 1, we only consider timesteps up to first event (t=3)
    patient1_timesteps = [m[1] for m in map if m[0] == 0]
    assert max(patient1_timesteps) == 2  # Last timestep is 2 (before first event at t=3)
    assert len(patient1_timesteps) == 3  # 3 timesteps for patient 1
    
    # Check time to event labels for patient 1
    assert flat_labels[0] == 3  # At t=0, time to first event is 3
    assert flat_labels[2] == 1  # At t=2, time to first event is 1
    
    # For patient 2, consider timesteps up to first event (t=8)
    patient2_timesteps = [m[1] for m in map if m[0] == 1]
    assert max(patient2_timesteps) == 7  # Last timestep is 7 (before first event at t=8)
    assert len(patient2_timesteps) == 8  # 8 timesteps for patient 2
    
    # Check time to event labels for patient 2
    first_p2_idx = len(patient1_timesteps)  # Index where patient 2 starts
    assert flat_labels[first_p2_idx] == 8  # At t=0, time to event is 8
    assert flat_labels[first_p2_idx + 7] == 1  # At t=7, time to event is 1

def test_decompose_and_label_timeseries_time_to_event_edge_cases():
    """Test edge cases with events at beginning or end"""
    timeseries = np.zeros((2, 10, 1, 1))
    timeseries[0, 0, 0, 0] = 100
    timeseries[1, 0, 0, 0] = 200
    
    # Patient 100 has event at second timestep t=1
    # Patient 200 has event at last timestep t=9
    edge_events_df = pd.DataFrame({
        'case_admission_id': [100, 200],
        'relative_sample_date_hourly_cat': [1, 9]
    })
    
    map, flat_labels = decompose_and_label_timeseries_time_to_event(timeseries, edge_events_df)
    
    # For patient 1 with event at t=1
    assert flat_labels[0] == 1  # At t=0, time to event is 1
    
    # For patient 2 with event at t=9 (last timestep)
    assert flat_labels[10] == 9  # At t=0, time to event is 9
    assert flat_labels[18] == 1  # At t=8, time to event is 1
    
    # With restrict_to_first_event=True
    map, flat_labels = decompose_and_label_timeseries_time_to_event(timeseries, edge_events_df, restrict_to_first_event=True)
    
    # For patient 1 with event at t=1, only t=0 should be included
    patient1_timesteps = [m[1] for m in map if m[0] == 0]
    assert len(patient1_timesteps) == 1  # Only includes t=0
    assert flat_labels[0] == 1  # At t=0, time to event is 1