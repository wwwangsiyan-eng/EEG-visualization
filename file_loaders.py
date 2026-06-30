"""
EEG file loaders supporting multiple formats.
Converts various formats to standardized internal format.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

from core_data_structures import EEGSignal, DataType

# Optional imports - handle gracefully if not installed
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

try:
    from neo.io import BlackrockIO
    try:
        from neo.io import NeurolynxIO, IntanIO
    except ImportError:
        # Some versions of Neo don't have these
        NeurolynxIO = None
        IntanIO = None
    NEO_AVAILABLE = True
except ImportError:
    NEO_AVAILABLE = False


class FileLoader(ABC):
    """Base class for file loaders."""
    
    @staticmethod
    @abstractmethod
    def can_load(file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        pass
    
    @abstractmethod
    def load(self, file_path: Path) -> EEGSignal:
        """Load file and return standardized EEGSignal."""
        pass
    
    @abstractmethod
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file."""
        pass


class MFFLoader(FileLoader):
    """Loader for EGI MFF format (human EEG)."""
    
    @staticmethod
    def can_load(file_path: Path) -> bool:
        """Check if file is MFF format."""
        if not MNE_AVAILABLE:
            return False
        return file_path.suffix.lower() == '.mff' or file_path.is_dir() and (file_path / 'signal.mff').exists()
    
    def load(self, file_path: Path) -> EEGSignal:
        """Load MFF file using MNE."""
        if not MNE_AVAILABLE:
            raise ImportError("MNE-Python not installed. Install with: pip install mne")
        
        try:
            raw = mne.io.read_raw_egi(str(file_path), preload=True)
        except Exception as e:
            raise ValueError(f"Failed to load MFF file: {e}")
        
        # Get data - MNE returns data in Volts (it applies the calibration from the file)
        # EGI MFF files have calibration factor ~1e-6 which MNE applies automatically
        data = raw.get_data().astype(np.float32)  # shape: (n_channels, n_samples)
        
        # Always convert from Volts to microvolts
        # MNE's read_raw_egi() returns data in Volts (SI units)
        # We want µV for EEG analysis (standard unit)
        data = data * 1e6  # Convert V to µV
        units_note = 'Converted from V to µV (MNE returns SI units)'
        
        sampling_rate = raw.info['sfreq']
        channel_names = raw.ch_names
        
        metadata = {
            'original_format': 'MFF',
            'n_channels': len(channel_names),
            'n_samples': data.shape[1],
            'duration_seconds': data.shape[1] / sampling_rate,
            'units': 'µV',
            'units_note': units_note
        }
        
        return EEGSignal(
            data=data,
            sampling_rate=sampling_rate,
            channel_names=channel_names,
            data_type=DataType.RAW_EEG,
            metadata=metadata
        )
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from MFF file."""
        if not MNE_AVAILABLE:
            return {}
        try:
            raw = mne.io.read_raw_egi(str(file_path), preload=False)
            return {
                'sampling_rate': raw.info['sfreq'],
                'n_channels': len(raw.ch_names),
                'duration_seconds': raw.times[-1] if len(raw.times) > 0 else 0
            }
        except:
            return {}


class NSXLoader(FileLoader):
    """Loader for Blackrock NSX format (animal EEG)."""
    
    @staticmethod
    def can_load(file_path: Path) -> bool:
        """Check if file is NSX format."""
        if not NEO_AVAILABLE:
            return False
        # Support both .nsx and .ns2 formats
        suffix = file_path.suffix.lower()
        return suffix in ['.nsx', '.ns2']
    
    def load(self, file_path: Path) -> EEGSignal:
        """Load NSX file using Neo."""
        if not NEO_AVAILABLE:
            raise ImportError("Neo not installed. Install with: pip install neo")
        
        try:
            reader = BlackrockIO(str(file_path.parent / file_path.stem))
            block = reader.read_block(0)
            segment = block.segments[0]
            
            # Extract analog signals
            if not segment.analogsignals:
                raise ValueError("No analog signals found in NSX file")
            
            signal = segment.analogsignals[0]
            data = signal.magnitude.T.astype(np.float32)  # shape: (n_channels, n_samples)
            sampling_rate = float(signal.sampling_rate.rescale('Hz').magnitude)
            channel_names = [f"Ch_{i}" for i in range(data.shape[0])]
            
            # Get original units and scale to µV if needed
            original_units = str(signal.units)
            units_note = f'Original units: {original_units}'
            
            # Blackrock NSX typically stores in µV or mV
            # Neo should handle the scaling, but verify
            if 'mV' in original_units or 'millivolt' in original_units.lower():
                data = data * 1000  # Convert mV to µV
                units_note = f'Scaled from mV to µV (original: {original_units})'
            elif 'V' in original_units and 'µ' not in original_units and 'micro' not in original_units.lower():
                data = data * 1e6  # Convert V to µV
                units_note = f'Scaled from V to µV (original: {original_units})'
            
            metadata = {
                'original_format': 'NSX_Blackrock',
                'n_channels': data.shape[0],
                'n_samples': data.shape[1],
                'duration_seconds': data.shape[1] / sampling_rate,
                'units': 'µV',
                'units_note': units_note
            }
            
            return EEGSignal(
                data=data,
                sampling_rate=sampling_rate,
                channel_names=channel_names,
                data_type=DataType.RAW_EEG,
                metadata=metadata
            )
        except Exception as e:
            raise ValueError(f"Failed to load NSX file: {e}")
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from NSX file."""
        if not NEO_AVAILABLE:
            return {}
        try:
            reader = BlackrockIO(str(file_path.parent / file_path.stem))
            block = reader.read_block(0)
            segment = block.segments[0]
            if segment.analogsignals:
                signal = segment.analogsignals[0]
                return {
                    'sampling_rate': float(signal.sampling_rate.rescale('Hz').magnitude),
                    'n_channels': signal.shape[1],
                    'duration_seconds': float(signal.t_stop.rescale('s').magnitude)
                }
        except:
            pass
        return {}


class NCSLoader(FileLoader):
    """Loader for Neuralynx NCS format (animal EEG)."""
    
    @staticmethod
    def can_load(file_path: Path) -> bool:
        """Check if file is NCS format."""
        # Check both that Neo is available AND NeuralynxIO is not None
        if not NEO_AVAILABLE or NeurolynxIO is None:
            return False
        return file_path.suffix.lower() == '.ncs'
    
    def load(self, file_path: Path) -> EEGSignal:
        """Load NCS file using Neo."""
        if not NEO_AVAILABLE:
            raise ImportError("Neo not installed. Install with: pip install neo")
        
        try:
            reader = NeurolynxIO(str(file_path))
            block = reader.read_block(0)
            segment = block.segments[0]
            
            if not segment.analogsignals:
                raise ValueError("No analog signals found in NCS file")
            
            signal = segment.analogsignals[0]
            data = signal.magnitude.T.astype(np.float32)  # shape: (1, n_samples) -> (n_channels, n_samples)
            sampling_rate = float(signal.sampling_rate.rescale('Hz').magnitude)
            channel_names = [file_path.stem]  # Use filename as channel name
            
            # Get original units and scale to µV if needed
            original_units = str(signal.units)
            units_note = f'Original units: {original_units}'
            
            # Neuralynx typically stores in µV but verify
            if 'mV' in original_units or 'millivolt' in original_units.lower():
                data = data * 1000  # Convert mV to µV
                units_note = f'Scaled from mV to µV (original: {original_units})'
            elif 'V' in original_units and 'µ' not in original_units and 'micro' not in original_units.lower():
                # Check if data looks like volts (very small values)
                channel_stds = np.std(data, axis=1)
                median_std = np.median(channel_stds)
                if median_std < 0.01:  # Likely in volts
                    data = data * 1e6  # Convert V to µV
                    units_note = f'Scaled from V to µV (median_std was {median_std:.6f} V)'
            
            metadata = {
                'original_format': 'NCS_Neurolynx',
                'n_channels': data.shape[0],
                'n_samples': data.shape[1],
                'duration_seconds': data.shape[1] / sampling_rate,
                'units': 'µV',
                'units_note': units_note,
                'original_units': original_units
            }
            
            return EEGSignal(
                data=data,
                sampling_rate=sampling_rate,
                channel_names=channel_names,
                data_type=DataType.RAW_EEG,
                metadata=metadata
            )
        except Exception as e:
            raise ValueError(f"Failed to load NCS file: {e}")
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from NCS file."""
        if not NEO_AVAILABLE:
            return {}
        try:
            reader = NeurolynxIO(str(file_path))
            block = reader.read_block(0)
            segment = block.segments[0]
            if segment.analogsignals:
                signal = segment.analogsignals[0]
                return {
                    'sampling_rate': float(signal.sampling_rate.rescale('Hz').magnitude),
                    'n_channels': signal.shape[1],
                    'duration_seconds': float(signal.t_stop.rescale('s').magnitude)
                }
        except:
            pass
        return {}


class RHDLoader(FileLoader):
    """Loader for Intan RHD format (animal EEG)."""
    
    @staticmethod
    def can_load(file_path: Path) -> bool:
        """Check if file is RHD format."""
        if not NEO_AVAILABLE:
            return False
        return file_path.suffix.lower() == '.rhd'
    
    def load(self, file_path: Path) -> EEGSignal:
        """Load RHD file using Neo."""
        if not NEO_AVAILABLE:
            raise ImportError("Neo not installed. Install with: pip install neo")
        
        try:
            reader = IntanIO(str(file_path))
            block = reader.read_block(0)
            segment = block.segments[0]
            
            if not segment.analogsignals:
                raise ValueError("No analog signals found in RHD file")
            
            signal = segment.analogsignals[0]
            data = signal.magnitude.T.astype(np.float32)  # shape: (n_channels, n_samples)
            sampling_rate = float(signal.sampling_rate.rescale('Hz').magnitude)
            channel_names = [f"Ch_{i}" for i in range(data.shape[0])]
            
            # Get original units and scale to µV if needed
            original_units = str(signal.units)
            units_note = f'Original units: {original_units}'
            
            # Intan RHD typically stores in µV but verify
            if 'mV' in original_units or 'millivolt' in original_units.lower():
                data = data * 1000  # Convert mV to µV
                units_note = f'Scaled from mV to µV (original: {original_units})'
            elif 'V' in original_units and 'µ' not in original_units and 'micro' not in original_units.lower():
                # Check if data looks like volts (very small values)
                channel_stds = np.std(data, axis=1)
                median_std = np.median(channel_stds)
                if median_std < 0.01:  # Likely in volts
                    data = data * 1e6  # Convert V to µV
                    units_note = f'Scaled from V to µV (median_std was {median_std:.6f} V)'
            
            metadata = {
                'original_format': 'RHD_Intan',
                'n_channels': data.shape[0],
                'n_samples': data.shape[1],
                'duration_seconds': data.shape[1] / sampling_rate,
                'units': 'µV',
                'units_note': units_note,
                'original_units': original_units
            }
            
            return EEGSignal(
                data=data,
                sampling_rate=sampling_rate,
                channel_names=channel_names,
                data_type=DataType.RAW_EEG,
                metadata=metadata
            )
        except Exception as e:
            raise ValueError(f"Failed to load RHD file: {e}")
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from RHD file."""
        if not NEO_AVAILABLE:
            return {}
        try:
            reader = IntanIO(str(file_path))
            block = reader.read_block(0)
            segment = block.segments[0]
            if segment.analogsignals:
                signal = segment.analogsignals[0]
                return {
                    'sampling_rate': float(signal.sampling_rate.rescale('Hz').magnitude),
                    'n_channels': signal.shape[1],
                    'duration_seconds': float(signal.t_stop.rescale('s').magnitude)
                }
        except:
            pass
        return {}


class DATLoader(FileLoader):
    """Loader for .dat files (BCI2000 or Curry format)."""
    
    @staticmethod
    def can_load(file_path: Path) -> bool:
        """Check if file is .dat format."""
        if not MNE_AVAILABLE:
            return False
        return file_path.suffix.lower() == '.dat'
    
    def load(self, file_path: Path) -> Tuple[EEGSignal, bool, str]:
        """Load .dat file using MNE."""
        if not MNE_AVAILABLE:
            return None, False, "MNE-Python not installed. Install with: pip install mne"
        
        try:
            try:
                raw = mne.io.read_raw_bci2000(str(file_path), preload=True, verbose=False)
            except Exception:
                try:
                    raw = mne.io.read_raw_curry(str(file_path), preload=True, verbose=False)
                except Exception as e:
                    return None, False, f"Could not parse .dat file as BCI2000 or Curry: {str(e)}"
            
            data = raw.get_data().astype(np.float32)
            data = data * 1e6  # Volts to µV
            sampling_rate = raw.info['sfreq']
            channel_names = raw.ch_names
            
            metadata = {
                'original_format': 'DAT',
                'n_channels': len(channel_names),
                'n_samples': data.shape[1],
                'duration_seconds': data.shape[1] / sampling_rate,
                'file_path': str(file_path),
                'units': 'µV'
            }
            
            signal = EEGSignal(
                data=data,
                sampling_rate=sampling_rate,
                channel_names=list(channel_names),
                data_type=DataType.RAW_EEG,
                metadata=metadata
            )
            
            # The interface from other loaders is `load(...) -> EEGSignal` 
            # (Exception: EDFLoader returns Tuple. Let's make this return EEGSignal but match EDFLoader pattern for ease if needed, actually FileLoader says `load(...) -> EEGSignal`. The `EDFLoader` violates this in `load`). Let's just return EEGSignal.
            return signal
            
        except Exception as e:
            raise ValueError(f"Failed to load DAT file: {str(e)}")
            
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from DAT file."""
        if not MNE_AVAILABLE:
            return {}
        try:
            try:
                raw = mne.io.read_raw_bci2000(str(file_path), preload=False, verbose=False)
            except Exception:
                raw = mne.io.read_raw_curry(str(file_path), preload=False, verbose=False)
            return {
                'sampling_rate': raw.info['sfreq'],
                'n_channels': len(raw.ch_names),
                'duration_seconds': raw.times[-1] if len(raw.times) > 0 else 0
            }
        except:
            return {}

class EDFLoader(FileLoader):
    """
    Loader for EDF (European Data Format) and EDF+ files.
    
    EDF is a standard format for exchange and storage of medical time series data,
    commonly used for polysomnography (PSG), EEG, ECG, and other biosignals.
    
    Supports:
    - .edf (European Data Format)
    - .bdf (BioSemi Data Format - 24-bit variant)
    
    Uses MNE-Python for robust EDF reading with proper handling of:
    - Channel labels and types
    - Physical dimensions (units)
    - Patient information
    - Recording date/time
    - Annotations (for EDF+)
    """
    
    @staticmethod
    def can_load(file_path: Path) -> bool:
        """Check if file is EDF/BDF format."""
        if not MNE_AVAILABLE:
            return False
        suffix = file_path.suffix.lower()
        return suffix in ['.edf', '.bdf']
    
    def load(self, file_path: Path) -> Tuple[EEGSignal, bool, str]:
        """
        Load EDF/BDF file using MNE.
        
        Returns:
            Tuple of (EEGSignal, success: bool, error_message: str)
        """
        if not MNE_AVAILABLE:
            return None, False, "MNE-Python not installed. Install with: pip install mne"
        
        try:
            # Read EDF file with MNE
            # preload=True loads all data into memory
            # stim_channel=False avoids auto-detection issues
            raw = mne.io.read_raw_edf(str(file_path), preload=True, stim_channel=False, verbose=False)
            
            # Extract data as float32 (n_channels, n_samples)
            # MNE returns data in Volts (SI units)
            data = raw.get_data().astype(np.float32)
            sampling_rate = raw.info['sfreq']
            channel_names = raw.ch_names
            
            # Always convert from Volts to microvolts
            # MNE's read_raw_edf() returns data in Volts (SI units)
            data = data * 1e6  # Convert V to µV
            units_note = 'Converted from V to µV (MNE returns SI units)'
            
            # Extract patient/recording info if available
            subject_info = raw.info.get('subject_info', {})
            meas_date = raw.info.get('meas_date', None)
            
            # Build comprehensive metadata
            metadata = {
                'original_format': 'EDF' if file_path.suffix.lower() == '.edf' else 'BDF',
                'n_channels': len(channel_names),
                'n_samples': data.shape[1],
                'duration_seconds': data.shape[1] / sampling_rate,
                'file_path': str(file_path),
                'units': 'µV',
                'units_note': units_note,
            }
            
            # Add patient info if available
            if subject_info:
                if 'his_id' in subject_info:
                    metadata['patient_id'] = subject_info['his_id']
                if 'birthday' in subject_info:
                    metadata['patient_birthday'] = str(subject_info['birthday'])
                if 'sex' in subject_info:
                    metadata['patient_sex'] = subject_info['sex']
            
            # Add recording date if available
            if meas_date is not None:
                metadata['recording_date'] = str(meas_date)
            
            # Extract channel types (EEG, EOG, EMG, ECG, etc.)
            ch_types = []
            for ch_name in channel_names:
                ch_name_upper = ch_name.upper()
                if any(x in ch_name_upper for x in ['EOG', 'EYE']):
                    ch_types.append('eog')
                elif any(x in ch_name_upper for x in ['EMG', 'CHIN', 'LEG']):
                    ch_types.append('emg')
                elif any(x in ch_name_upper for x in ['ECG', 'EKG', 'HEART']):
                    ch_types.append('ecg')
                elif any(x in ch_name_upper for x in ['RESP', 'AIRFLOW', 'THORAX', 'ABDOMEN']):
                    ch_types.append('resp')
                elif any(x in ch_name_upper for x in ['SPO2', 'SAO2', 'OXYGEN']):
                    ch_types.append('spo2')
                else:
                    ch_types.append('eeg')
            
            metadata['channel_types'] = ch_types
            
            # Get annotations if present (EDF+)
            annotations = raw.annotations
            if len(annotations) > 0:
                metadata['annotations'] = {
                    'onset': annotations.onset.tolist(),
                    'duration': annotations.duration.tolist(),
                    'description': annotations.description.tolist()
                }
                metadata['n_annotations'] = len(annotations)
            
            signal = EEGSignal(
                data=data,
                sampling_rate=sampling_rate,
                channel_names=list(channel_names),
                data_type=DataType.RAW_EEG,
                metadata=metadata
            )
            
            return signal, True, ""
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to load EDF file: {str(e)}"
            traceback.print_exc()
            return None, False, error_msg
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from EDF file without loading full data."""
        if not MNE_AVAILABLE:
            return {}
        try:
            # Read with preload=False to just get header info
            raw = mne.io.read_raw_edf(str(file_path), preload=False, verbose=False)
            
            metadata = {
                'sampling_rate': raw.info['sfreq'],
                'n_channels': len(raw.ch_names),
                'channel_names': list(raw.ch_names),
                'duration_seconds': raw.times[-1] if len(raw.times) > 0 else 0,
                'format': 'EDF' if file_path.suffix.lower() == '.edf' else 'BDF'
            }
            
            # Add measurement date if available
            if raw.info.get('meas_date'):
                metadata['recording_date'] = str(raw.info['meas_date'])
            
            return metadata
        except Exception as e:
            print(f"Error reading EDF metadata: {e}")
            return {}


class SimpleNCSLoader(FileLoader):
    """Fallback NCS loader using basic binary reading for Neuralynx files."""
    
    @staticmethod
    def can_load(file_path: Path) -> bool:
        return file_path.suffix.lower() == '.ncs'
    
    def get_metadata(self, file_path: Path) -> Dict:
        """Get metadata from file."""
        return {'format': 'Neuralynx NCS'}
    
    def load(self, file_path: Path) -> Tuple[Optional[EEGSignal], bool, str]:
        """Load Neuralynx .ncs file using basic binary parsing.
        
        NCS File Structure:
        - Header: 16384 bytes (ASCII text)
        - Data records: Each record is 1044 bytes
          - qwTimeStamp: 8 bytes (uint64) - microseconds
          - dwChannelNumber: 4 bytes (uint32)
          - dwSampleFreq: 4 bytes (uint32)
          - dwNumValidSamples: 4 bytes (uint32) - typically 512
          - snSamples: 512 * 2 bytes (int16[512])
        """
        try:
            import struct
            
            with open(file_path, 'rb') as f:
                # Read header (16384 bytes)
                header = f.read(16384).decode('utf-8', errors='ignore')
                
                # Extract sampling rate from header
                sampling_rate = 32000.0  # Default
                for line in header.split('\n'):
                    if 'SamplingFrequency' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                sampling_rate = float(parts[-1])
                            except:
                                pass
                        break
                
                # Extract ADBitVolts for voltage conversion
                ad_bit_volts = 1.0  # Default (no scaling)
                for line in header.split('\n'):
                    if 'ADBitVolts' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                ad_bit_volts = float(parts[-1])
                            except:
                                pass
                        break
                
                # Neuralynx NCS record structure (1044 bytes total):
                # - Timestamp: 8 bytes (uint64)
                # - Channel number: 4 bytes (uint32) 
                # - Sample frequency: 4 bytes (uint32)
                # - Number of valid samples: 4 bytes (uint32)
                # - Samples: 512 samples * 2 bytes = 1024 bytes (int16[512])
                RECORD_SIZE = 1044
                HEADER_SIZE_IN_RECORD = 20  # 8 + 4 + 4 + 4
                SAMPLES_PER_RECORD = 512
                
                # Read data records
                all_samples = []
                while True:
                    record = f.read(RECORD_SIZE)
                    if len(record) < RECORD_SIZE:
                        break
                    
                    # Parse record header
                    # timestamp, channel_num, sample_freq, num_valid = struct.unpack('<QIII', record[:20])
                    num_valid = struct.unpack('<I', record[16:20])[0]
                    
                    # Parse samples (int16)
                    sample_bytes = record[HEADER_SIZE_IN_RECORD:HEADER_SIZE_IN_RECORD + num_valid * 2]
                    if len(sample_bytes) == num_valid * 2:
                        samples = struct.unpack(f'<{num_valid}h', sample_bytes)
                        all_samples.extend(samples)
                
                if not all_samples:
                    return None, False, "No data found in file"
                
                # Convert to numpy array and apply voltage scaling
                # NCS stores data as int16, need to convert to µV
                data_array = np.array(all_samples, dtype=np.float32).reshape(1, -1)
                data_array = data_array * ad_bit_volts * 1e6  # Convert to µV
                
                # Extract channel name from filename (CSC1_0001 → CSC1)
                channel_name = file_path.stem.split('_')[0]
                
                signal = EEGSignal(
                    data=data_array,
                    sampling_rate=sampling_rate,
                    channel_names=[channel_name],
                    data_type=DataType.RAW_EEG,
                    metadata={
                        'filename': file_path.name,
                        'original_format': 'NCS_Neuralynx',
                        'ad_bit_volts': ad_bit_volts,
                        'units': 'µV',
                        'units_note': f'Converted from int16 * ADBitVolts * 1e6 (ADBitVolts={ad_bit_volts})'
                    }
                )
                
                return signal, True, ""
        
        except Exception as e:
            import traceback
            return None, False, f"SimpleNCSLoader error: {str(e)}"


class FileLoaderRegistry:
    """Registry and factory for file loaders."""
    
    def __init__(self):
        self.loaders = [
            MFFLoader(),
            NSXLoader(),
            NCSLoader(),
            SimpleNCSLoader(),  # Fallback
            RHDLoader(),
            EDFLoader(),  # EDF/BDF support
            DATLoader(),  # BCI2000 / Curry .dat support
        ]
    
    def get_loader(self, file_path: Path) -> Optional[FileLoader]:
        """Get appropriate loader for file."""
        file_path = Path(file_path)
        
        for loader in self.loaders:
            try:
                if loader.can_load(file_path):
                    return loader
            except Exception:
                continue
        
        return None
    
    def load_file(self, file_path: Path) -> Tuple[EEGSignal, bool, str]:
        """
        Load file with appropriate loader.
        
        Returns:
            (EEGSignal, success: bool, error_message: str)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None, False, f"File not found: {file_path}"
        
        loader = self.get_loader(file_path)
        if not loader:
            supported = self.get_supported_formats()
            return None, False, (f"Unsupported file format: {file_path.suffix}. "
                               f"Supported formats: {', '.join(supported)}")
        
        try:
            result = loader.load(file_path)
            # Loader.load() returns (signal, success, msg) tuple
            if isinstance(result, tuple) and len(result) == 3:
                return result
            else:
                # Shouldn't happen, but handle gracefully
                return result, True, ""
        except Exception as e:
            return None, False, str(e)
    
    def get_supported_formats(self) -> list:
        """Return list of supported file extensions."""
        return ['.mff', '.nsx', '.ns2', '.ncs', '.rhd', '.edf', '.bdf', '.dat']


# Global registry instance
_loader_registry = FileLoaderRegistry()


def load_eeg_file(file_path: Path) -> Tuple[EEGSignal, bool, str]:
    """
    Load EEG file from disk.
    
    Returns:
        (EEGSignal, success, error_message)
    """
    return _loader_registry.load_file(file_path)


def get_supported_formats() -> list:
    """Get list of supported file formats."""
    return _loader_registry.get_supported_formats()
