from scipy.signal import butter, filtfilt, firwin, iirnotch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

class Filter(ABC):
    """
    Abstract base class for filters.
    Instances are callable: filtered_signal = filter_instance(signal)
    """

    def __init__(self):
        """
        """
        pass

    @abstractmethod
    def __call__(self, signal: np.ndarray, fs:float) -> np.ndarray:
        """
        Apply the filter to the input signal.

        Args:
            signal (np.ndarray): Input signal array.
            fs (float): Sampling frequency in Hz.

        Returns:
            np.ndarray: Filtered signal.
        """
        pass

class butter_HPF(Filter):
    """
    Butterworth High-Pass Filter class.

    This class implements a high-pass Butterworth filter using SciPy's 
    `butter` and `filtfilt` functions. It inherits from the abstract `Filter` class 
    and can be used as a callable to filter input signals.

    Attributes:
        cutoff (float): Cutoff frequency of the high-pass filter in Hz.
        order (int): Order of the Butterworth filter. Controls the steepness of the filter roll-off.

    Methods:
        coeff(fs: float) -> Tuple[np.ndarray, np.ndarray]:
            Computes the filter coefficients (b, a) for the Butterworth high-pass filter
            given a sampling frequency `fs`.
        
        __call__(data: np.ndarray, fs: float) -> np.ndarray:
            Applies zero-phase high-pass filtering to the input signal `data` using the
            filter coefficients and returns the filtered signal.
    
    Example:
        >>> hpf = butter_HPF(cutoff=50, order=4)
        >>> filtered_signal = hpf(signal, fs=500)
    """

    def __init__(self, cutoff: float, order: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.order = order

    def coeff(self, fs: float):
        nyq = 0.5 * fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype='high', analog=False)
        return b, a

    def __call__(self, data: np.ndarray, fs: float) -> np.ndarray:
        b, a = self.coeff(fs)
        y = filtfilt(b, a, data)
        return y

class butter_LPF(Filter):
    """
    Butterworth Low-Pass Filter class.

    This class implements a low-pass Butterworth filter using SciPy's 
    `butter` and `filtfilt` functions. It inherits from the abstract `Filter` class 
    and can be used as a callable to filter input signals.

    Attributes:
        cutoff (float): Cutoff frequency of the low-pass filter in Hz.
        order (int): Order of the Butterworth filter. Controls the steepness of the filter roll-off.

    Methods:
        coeff(fs: float) -> Tuple[np.ndarray, np.ndarray]:
            Computes the filter coefficients (b, a) for the Butterworth low-pass filter
            given a sampling frequency `fs`.
        
        __call__(data: np.ndarray, fs: float) -> np.ndarray:
            Applies zero-phase low-pass filtering to the input signal `data` using the
            filter coefficients and returns the filtered signal.
    
    Example:
        >>> lpf = butter_LPF(cutoff=40, order=4)
        >>> filtered_signal = lpf(signal, fs=500)
    """

    def __init__(self, cutoff: float, order: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.order = order

    def coeff(self, fs: float):
        nyq = 0.5 * fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        return b, a

    def __call__(self, data: np.ndarray, fs: float) -> np.ndarray:
        b, a = self.coeff(fs)
        y = filtfilt(b, a, data)
        return y

class butter_BSF(Filter):
    """
    Butterworth Band-Stop Filter class.

    This class implements a band-stop Butterworth filter using SciPy's 
    `butter` and `filtfilt` functions. It inherits from the abstract `Filter` class 
    and can be used as a callable to filter input signals.

    Attributes:
        lowcut (float): Lower cutoff frequency of the stop band in Hz.
        highcut (float): Upper cutoff frequency of the stop band in Hz.
        order (int): Order of the Butterworth filter. Controls the steepness of the stop band.

    Methods:
        coeff(fs: float) -> Tuple[np.ndarray, np.ndarray]:
            Computes the filter coefficients (b, a) for the Butterworth band-stop filter
            given a sampling frequency `fs`.
        
        __call__(data: np.ndarray, fs: float) -> np.ndarray:
            Applies zero-phase band-stop filtering to the input signal `data` using the
            filter coefficients and returns the filtered signal.

    Example:
        >>> bsf = butter_BSF(lowcut=48, highcut=52, order=4)
        >>> filtered_signal = bsf(signal, fs=500)
    """

    def __init__(self, lowcut: float, highcut: float, order: int = 5):
        super().__init__()
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def coeff(self, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        nyq = 0.5 * fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='bandstop', analog=False)
        return b, a

    def __call__(self, data: np.ndarray, fs: float) -> np.ndarray:
        b, a = self.coeff(fs)
        y = filtfilt(b, a, data)
        return y

class NLMSFilter(Filter):
    """
    Normalized Least Mean Squares (NLMS) adaptive notch filter.

    This filter adaptively removes sinusoidal interference at specified frequencies 
    from the input signal using the NLMS algorithm.

    Attributes:
        target_freqs (list of float): Frequencies to be removed (Hz).
        mu (float): Step size (learning rate) of the NLMS algorithm.
        epsilon (float): Small constant to avoid division by zero in normalization.

    Methods:
        __call__(signal: np.ndarray, fs: float) -> np.ndarray:
            Applies the NLMS adaptive filtering to the input signal.

    Example:
        >>> nlms = NLMSFilter(target_freqs=[50], mu=0.001)
        >>> filtered_signal = nlms(signal, fs=500)
    """

    def __init__(self, target_freqs=[50], mu=0.01, epsilon=1e-6):
        self.target_freqs = target_freqs
        self.mu = mu
        self.epsilon = epsilon

    def __call__(self, signal: np.ndarray, fs: float) -> np.ndarray:
        n_samples = len(signal)
        t = np.arange(n_samples) / fs

        # Generate reference sinusoids for target frequencies (sin and cos)
        reference = np.stack([
            func(2 * np.pi * f * t)
            for f in self.target_freqs
            for func in (np.sin, np.cos)
        ], axis=1)

        n_coeffs = reference.shape[1]
        w = np.zeros(n_coeffs)
        output = np.zeros(n_samples)

        for n in range(n_samples):
            x = reference[n]
            y = np.dot(w, x)
            e = signal[n] - y
            norm_factor = np.dot(x, x) + self.epsilon
            w += 2 * self.mu * e * x / norm_factor
            output[n] = e
        return output
    
class IIRNotchFilter:
    """
    Multi-frequency IIR notch filter with zero-phase filtering.

    Parameters:
        notch_freqs (float or list of float): Frequency or list of frequencies to notch out (Hz).
        Q (float or list of float): Quality factor(s). Either a single value applied to all frequencies
                                    or a list with one value per frequency.
    
    Methods:
        __call__(signal, fs): Apply zero-phase IIR notch filtering to the input signal.
    """

    def __init__(self, notch_freqs=60.0, Q=30.0):
        # Ensure notch_freqs is a list
        if np.isscalar(notch_freqs):
            notch_freqs = [notch_freqs]
        self.notch_freqs = notch_freqs

        # Handle Q: single value or list matching notch_freqs length
        if np.isscalar(Q):
            Q = [Q] * len(self.notch_freqs)
        elif len(Q) != len(self.notch_freqs):
            raise ValueError("Length of Q list must match length of notch_freqs list.")
        self.Q = Q

    def __call__(self, signal: np.ndarray, fs: float) -> np.ndarray:
        y = np.copy(signal)
        nyq = fs / 2.0

        # Apply each notch sequentially
        for freq, q in zip(self.notch_freqs, self.Q):
            b, a = iirnotch(freq / nyq, q)
            y = filtfilt(b, a, y)
        return y
class FIRNotchFilter(Filter):
    """
    FIR-based notch filter with automatic tap estimation and zero-phase filtering.

    Parameters:
        notch_freq (float): Frequency to notch out (e.g., 50 Hz).
        bandwidth (float): Bandwidth around the notch frequency (Hz).
        numtaps (int or None): Optional number of taps. If None, it's estimated automatically.
    
    Methods:
        __call__(signal, fs): Apply zero-phase FIR notch filtering on the input signal.
    """

    def __init__(self, notch_freq: float = 50.0, bandwidth: float = 5.0, numtaps: int = None):
        self.notch_freq = notch_freq
        self.bandwidth = bandwidth
        self.numtaps = numtaps  # If None, we'll estimate it later

    def estimate_taps(self, fs: float) -> int:
        transition_width = self.bandwidth / fs
        estimated = int(np.ceil(4 / transition_width))
        if estimated % 2 == 0:
            estimated += 1
        estimated = max(estimated, 101)
        max_taps = 1001  # max allowed taps (ajuste selon ton signal)
        return min(estimated, max_taps)

    def __call__(self, signal: np.ndarray, fs: float) -> np.ndarray:
        nyq = fs / 2.0
        f1 = (self.notch_freq - self.bandwidth / 2) / nyq
        f2 = (self.notch_freq + self.bandwidth / 2) / nyq

        taps = firwin(
            numtaps=self.numtaps if self.numtaps else self.estimate_taps(fs),
            cutoff=[f1, f2],
            pass_zero=True
        )

        return filtfilt(taps, 1.0, signal)

class LMSFilter(Filter):
    """
    Least Mean Squares (LMS) adaptive notch filter (non-normalized).

    This filter adaptively removes sinusoidal interference at specified frequencies 
    from the input signal using the LMS algorithm without normalization.

    Attributes:
        target_freqs (list of float): Frequencies to be removed (Hz).
        mu (float): Step size (learning rate) of the LMS algorithm.

    Methods:
        __call__(signal: np.ndarray, fs: float) -> np.ndarray:
            Applies the LMS adaptive filtering to the input signal.

    Example:
        >>> lms = LMSFilter(target_freqs=[50], mu=0.001)
        >>> filtered_signal = lms(signal, fs=500)
    """

    def __init__(self, target_freqs=[50], mu=0.01):
        self.target_freqs = target_freqs
        self.mu = mu

    def __call__(self, signal: np.ndarray, fs: float) -> np.ndarray:
        n_samples = len(signal)
        t = np.arange(n_samples) / fs

        # Generate reference sinusoids (sin and cos) for each target frequency
        reference = np.stack([
            func(2 * np.pi * f * t)
            for f in self.target_freqs
            for func in (np.sin, np.cos)
        ], axis=1)

        n_coeffs = reference.shape[1]
        w = np.zeros(n_coeffs)
        output = np.zeros(n_samples)

        for n in range(n_samples):
            x = reference[n]
            y = np.dot(w, x)
            e = signal[n] - y
            norm_x = np.dot(x, x)
            if norm_x > 1e-8:
                w += 2 * self.mu * e * x / norm_x
            output[n] = e

        return output

class RLSFilter(Filter):
    """
    Recursive Least Squares (RLS) adaptive notch filter.

    Attributes:
        target_freqs (list of float): Frequencies to be removed (Hz).
        lambda_ (float): Forgetting factor (close to but less than 1).
        delta (float): Initialization parameter for inverse correlation matrix.

    Methods:
        __call__(signal: np.ndarray, fs: float) -> np.ndarray:
            Applies the RLS adaptive filtering to the input signal.

    Example:
        >>> rls = RLSFilter(target_freqs=[50], lambda_=0.99, delta=1.0)
        >>> filtered_signal = rls(signal, fs=500)
    """

    def __init__(self, target_freqs=[50], lambda_=0.99, delta=1.0):
        self.target_freqs = target_freqs
        self.lambda_ = lambda_
        self.delta = delta

    def __call__(self, signal: np.ndarray, fs: float) -> np.ndarray:
        n_samples = len(signal)
        t = np.arange(n_samples) / fs
        # Reference matrix: sin and cos for each freq, shape (n_samples, 2*num_freqs)
        reference = np.stack([
            func(2 * np.pi * f * t)
            for f in self.target_freqs
            for func in (np.sin, np.cos)
        ], axis=1)

        n_coeffs = reference.shape[1]
        w = np.zeros(n_coeffs)
        P = np.eye(n_coeffs) / self.delta
        output = np.zeros(n_samples)

        for n in range(n_samples):
            x = reference[n]
            y = np.dot(w, x)
            e = signal[n] - y

            Px = P @ x
            k = Px / (self.lambda_ + x @ Px)
            w += k * e
            P = (P - np.outer(k, Px)) / self.lambda_

            output[n] = e

        return output
