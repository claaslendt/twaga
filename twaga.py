import numpy as np
import pandas as pd
import polars as pl
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from tensorflow import keras
from tensorflow.keras.models import load_model
from tcn import TCN

def get_walk_seqs(acc):
    '''

    Identify start and end indices for walking sequences in the accelerometer data.
    
    Input:
    acc: pandas dataframe with accelerometer data and activity labels

    Output:
    walk_seqs: pandas dataframe with start and end indices of walking sequences

    '''

    # filter the accelerometer data for walking activity
    walk = acc[acc['activity'] == 'walking'].copy()

    # calculate the time difference between consecutive rows
    walk['t_diff'] = walk.index.diff(1)
    walk['t_diff'].fillna(1, inplace=True)

    start = walk[walk['t_diff'] > 1].index.tolist()
    start.insert(0, walk.index[0])

    end = []
    for i in range(1, len(start)):
        end.append(walk[walk.index < start[i]].index[-1])
    end.append(walk.index[-1])

    # create a pandas df with start and end
    walk_seqs = pd.DataFrame(data= {'start': start, 'end': end})

    return walk_seqs

def estimate_speed(acc, height, a_coeff=2.064, b_coeff=0.349, LBR=0.53, min_lag=0.5, max_lag=3):

    '''
    Estimate walking speed from accelerometer data using a modified version of the Baroudi et al. (2020) method.

    Parameters
    ----------
    acc : pd.DataFrame
        DataFrame containing accelerometer data with columns 'x', 'y', 'z', and 'activity'.
    height : float
        Height of the individual in meters.
    a_coeff : float
        Coefficient a for speed estimation equation.
    b_coeff : float
        Coefficient b for speed estimation equation.
    LBR : float
        Leg-to-body ratio for estimating leg length.
    min_lag : float
        Minimum lag time for peak detection in seconds.
    max_lag : float
        Maximum lag time for peak detection in seconds.

    Returns
    -------
    speed : np.array
        Estimated walking speed time series in m/s.
    '''

    speed = np.repeat(np.nan, len(acc))

    walk_seqs = get_walk_seqs(acc)

    for seq in walk_seqs.itertuples():
        start_sq = seq.start
        end_sq = seq.end
        sq = acc.loc[start_sq:end_sq].copy()

        n_wndw = len(sq) // 400
        
        for wndw in range(n_wndw):
            start_wndw = wndw * 400
            end_wndw = (wndw + 1) * 400
            sq_wndw = sq.iloc[start_wndw:end_wndw].copy()
            speed_val = get_speed(sq_wndw['y'], height, a_coeff, b_coeff, LBR, min_lag, max_lag)

            speed[start_sq + start_wndw:start_sq + end_wndw] = speed_val

    return speed

def get_speed(acc, height, a_coeff=2.064,b_coeff=0.349, LBR=0.53, min_lag=0.5, max_lag=3):
        '''
        Estimate the walking speed for a given window using autocorrelation.

        Parameters
        ----------
        acc : np.array
            The accelerometer data time series (only a single axis is used).
        est_leg_length : float
            The estimated leg length in meters.
        a_coeff : float
            The coefficient a from the equation for estimating the walking speed.
        b_coeff : float
            The coefficient b from the equation for estimating the walking speed.
        min_dist : float
            The minimum distance between peaks in seconds.
        max_dist : float
            The maximum distance between peaks in seconds.

        Returns
        -------
        speed : float
            The estimated walking speed in m/s.
        stride_time : float
            The time between 0 and the first peak in seconds.
        stride_freq : float
            The frequency of the steps in Hz.
        peaks : np.array
            The indices of the filtered peaks in the autocorrelation.
        autocorr : np.array
            The autocorrelation of the accelerometer data.
        '''
        # estimate leg length based on the height
        leg_length = height * LBR

        # if acc is a pandas series, convert it to a numpy array
        if isinstance(acc, pd.Series):
            acc = acc.to_numpy()

        acc = acc - np.mean(acc)
        b,a = signal.butter(4, 5, btype='lowpass', fs=100)
        acc = signal.filtfilt(b, a, acc)

        autocorr = np.correlate(acc, acc, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Keep second half

        # find the peaks in the autocorrelation
        peaks, _ = signal.find_peaks(autocorr, height=0.1)
        peaks = peaks[peaks > min_lag * 100] # remove/filter peaks below the minimum distance
        peaks = peaks[peaks < max_lag * 100] # remove/filter peaks above the maximum distance
        if len(peaks) == 0:
            return None
        peak_max = peaks[np.argsort(autocorr[peaks])[-1]]  # find the peak with the highest autocorrelation value

        # calculate the time between 0 and the first peak
        stride_time = peak_max / 100
        stride_freq = 1 / stride_time
        stride_freq_norm = stride_freq / np.sqrt(9.81 / leg_length)
        speed_norm = np.exp(np.log(a_coeff*stride_freq_norm)/(1-b_coeff)) # calculate the normalized walking speed
        speed = speed_norm * np.sqrt(9.81 * leg_length) # calculate the walking speed in m/s using the estimated leg length

        return np.round(speed, 3)

def reshape_acc(df, seq_len=400):
    '''Reshape the accelerometer time series data to (n_seq, 400, 3)'''

    # assert df is a pandas df or polars df
    assert isinstance(df, (pd.DataFrame, pl.DataFrame)), "Input must be a pandas or polars dataframe."

    # if acc is a pandas dataframe, convert to numpy array
    if isinstance(df, pd.DataFrame):
        acc = df[['x', 'y', 'z']].to_numpy()
    elif isinstance(df, pl.DataFrame):
        acc = df['x', 'y', 'z'].to_numpy()
    n_seq = int(np.floor(acc.shape[0] / seq_len))
    acc = acc[:n_seq * seq_len]
    acc = acc.reshape(n_seq, seq_len, 3)

    return acc

def classify_act(df, model='Lendt_2024', str_label=True, filt_cycling=True):
    '''
    Classify physical activity type from accelerometer data using a pre-trained deep learning model based on Lendt et al. (2024).

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        DataFrame containing accelerometer data with columns 'x', 'y', 'z'.
    model : str, optional
        The name of the trained model to use for activity classification. Default is 'Lendt_2024'.
    str_label : bool, optional
        Whether to return activity labels as strings. If False, returns integer labels. Default is True.
    filt_cycling : bool, optional
        Whether to apply post-processing filter to smooth cycling predictions. Default is True.

    Returns
    -------
    predictions : np.array
        Array of predicted activity labels for each time point in the input DataFrame.

    '''
    
    if model == 'Lendt_2024':
       
        # Keras 2 to Keras 3 migration: define the model architecture and load the model weights separately
        # Todo: Re-train and save the model with Keras 3 to avoid this workaround
        model = keras.models.Sequential([
            keras.Input(shape=[400, 3]),
            keras.layers.Conv1D(filters=64, 
                                kernel_size=32, 
                                strides=1, 
                                activation='relu', 
                                padding='same'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Conv1D(filters=128, 
                                kernel_size=16, 
                                strides=1, 
                                activation='relu', 
                                padding='same'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False)),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(5, activation='softmax')
        ])

        model.load_weights('models/CNN_BiLSTM_weights.h5')
    
    else:
        raise ValueError("Model not recognized. Only 'Lendt_2024' is supported at the moment.")


    acc = reshape_acc(df, seq_len=400)
    predictions = model.predict(acc / 8, verbose=0)
    predictions = np.argmax(predictions, axis=1)

    # Todo: implement HMM smoothing instead
    if filt_cycling:

        cycling = np.zeros(len(predictions))
        cycling = np.where(predictions == 4, 1, 0)

        cycling_filt = np.convolve(cycling, np.ones(10)/10, mode='same')
        cycling_filt = np.where(cycling_filt >= 0.5, 1, 0)
        cycling_filt = np.where(cycling_filt < 0.5, 0, 1)

        predictions = np.where(cycling_filt == 1, 4, predictions)

    if str_label:

        predictions_str = []

        for pred in predictions:
                
                if pred == 0:
                    predictions_str.append('walking')
                elif pred == 1:
                    predictions_str.append('running')
                elif pred == 2:
                    predictions_str.append('standing')
                elif pred == 3:
                    predictions_str.append('sitting')
                elif pred == 4:
                    predictions_str.append('cycling')

        # convert to np array
        predictions_str = np.array(predictions_str)
        predictions = predictions_str

    # create series of predictions
    predictions = np.repeat(predictions, 400, axis=0)
    if len(predictions) < len(df):
        predictions = np.concatenate((predictions, np.repeat('unknown', len(df) - len(predictions))))

    # create a new column in the dataframe with the predictions
    if isinstance(df, pd.DataFrame):
        df['activity'] = predictions
    elif isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.Series('activity', predictions))
    
    return predictions

def filt_probs(probs, method, *args):
    '''
    Filter predicted probabilities using specified method.

    Parameters
    ----------
    probs : np.array
        Array of predicted probabilities.
    method : str
        Filtering method: 'gaussian', 'mean', or 'false' (no filtering).
    *args : additional arguments for the filtering method.

    Returns
    -------
    probs_filtered : np.array
        Filtered probabilities.

    '''

    if method == 'gaussian':
        probs_filtered = gaussian_filter(probs, sigma=args[0])

    elif method == 'mean':
        probs_filtered = np.convolve(probs, np.ones(args[0])/args[0], mode='same')
    
    elif method == 'false':
        probs_filtered = probs

    return probs_filtered

def detect_gait_events(df, model='Lendt_2025', prob_thresh=0.4, peak_dist=50, echo=True):
    '''
    Detect gait events (initial contact and final contact) from accelerometer data.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing accelerometer data with columns 'x', 'y', 'z'.

    model : str, optional
        The name of the trained TCN model to use for gait event detection. Default is 'Lendt_2025'.

    prob_thresh : float, optional
        Probability threshold for detecting gait events. Default is 0.4.
    
    peak_dist : int, optional
        Minimum distance between detected peaks in samples. Default is 50.
    
    echo : bool, optional
        Whether to print progress messages during prediction. Default is True.

    Returns
    -------
    gait_events : pd.DataFrame
        DataFrame containing detected gait events with columns 'IC' and 'TO'.
    '''

    # assert acc is a pandas dataframe
    assert isinstance(df, pd.DataFrame), "Input must be a pandas dataframe."

    # assert all necessary columns are in the dataframe
    assert all(col in df.columns for col in ['x', 'y', 'z', 'activity']), "Input DataFrame must contain 'x', 'y', 'z', and 'activity' columns."

    # load the trained TCN model
    if model == 'Lendt_2025':
        tcn_model = load_model('models/TCN.keras',
                               compile=False,
                               custom_objects={'TCN': TCN})
    else:
        raise ValueError("Model not recognized. Only 'Lendt_2025' is supported at the moment.")

    # create an empty dataframe to store gait events with the same length as df
    gait_events = pd.DataFrame(0, index=df.index, columns=['IC', 'FC'])

    # reshape the accelerometer data
    X = df[['x', 'y', 'z']].values.copy()
    X = X.reshape(1, X.shape[0], X.shape[1])

    # center around zero
    X = X - X.mean(axis=1)

    # predict probabilities for gait events
    y_pred = tcn_model.predict(X, verbose=echo)

    # add predicted probabilities to the dataframe
    df['IC_prob'] = y_pred[0,:,1]
    df['FC_prob'] = y_pred[0,:,2]

    # filter probabilities and detect peaks
    df['IC_prob_filt'] = filt_probs(df['IC_prob'], 'gaussian', 3)
    df['FC_prob_filt'] = filt_probs(df['FC_prob'], 'gaussian', 3)
    
    # find peaks in the filtered probabilities
    ix_pred_IC, _ = find_peaks(df['IC_prob_filt'], height=prob_thresh, distance=peak_dist)
    ix_pred_FC, _ = find_peaks(df['FC_prob_filt'], height=prob_thresh, distance=peak_dist)

    # mark detected events in the gait_events dataframe
    gait_events.loc[ix_pred_IC, 'IC'] = 1
    gait_events.loc[ix_pred_FC, 'FC'] = 1

    return gait_events
