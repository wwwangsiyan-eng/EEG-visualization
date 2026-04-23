"""
Core data structures for EEG processing protocol system.
Defines standardized internal format and step definitions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class DataType(Enum):
    """Standardized data types flowing through the protocol pipeline."""
    RAW_EEG = "raw_eeg"  # shape: (n_channels, n_samples), float32
    FILTERED_EEG = "filtered_eeg"  # shape: (n_channels, n_samples), float32
    ICA_COMPONENTS = "ica_components"  # shape: (n_components, n_samples), float32
    ICA_WEIGHTS = "ica_weights"  # shape: (n_channels, n_components), float32
    POWER_SPECTRUM = "power_spectrum"  # shape: (n_channels, n_freqs), float32
    TIME_FREQUENCY = "time_frequency"  # shape: (n_channels, n_freqs, n_times), float32
    TIME_TAGS = "time_tags"  # shape: (n_events,) or list of timestamps
    TIME_PERIODS = "time_periods"  # shape: (n_periods, 2) - start/end times
    SCALAR = "scalar"  # shape: () or (1,) - single numeric value
    VECTOR = "vector"  # shape: (n,) - 1D array
    MATRIX = "matrix"  # shape: (m, n) - 2D array


@dataclass
class EEGSignal:
    """Standardized internal EEG data format."""
    data: np.ndarray  # shape depends on data_type
    sampling_rate: float  # Hz
    channel_names: List[str]
    data_type: DataType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=np.float32)
        if not isinstance(self.data_type, DataType):
            raise TypeError(f"data_type must be DataType enum, got {type(self.data_type)}")


@dataclass
class StepInput:
    """Input specification for an analysis step."""
    name: str
    required_type: DataType
    description: str


@dataclass
class StepOutput:
    """Output specification for an analysis step."""
    name: str
    data_type: DataType
    description: str


class AnalysisStep(ABC):
    """Base class for all analysis steps."""
    
    def __init__(self, name: str, step_id: str):
        self.name = name
        self.step_id = step_id
        self.parameters: Dict[str, Any] = {}
    
    @abstractmethod
    def get_inputs(self) -> List[StepInput]:
        """Return list of required inputs."""
        pass
    
    @abstractmethod
    def get_outputs(self) -> List[StepOutput]:
        """Return list of produced outputs."""
        pass
    
    @abstractmethod
    def process(self, inputs: Dict[str, EEGSignal]) -> Dict[str, EEGSignal]:
        """
        Execute the analysis step.
        
        Args:
            inputs: Dict mapping input names to EEGSignal objects
            
        Returns:
            Dict mapping output names to EEGSignal objects
        """
        pass
    
    def set_parameter(self, param_name: str, value: Any) -> None:
        """Set a processing parameter."""
        self.parameters[param_name] = value
    
    def validate_inputs(self, inputs: Dict[str, EEGSignal]) -> Tuple[bool, str]:
        """Validate that inputs match step requirements."""
        step_inputs = self.get_inputs()
        
        for step_input in step_inputs:
            if step_input.name not in inputs:
                return False, f"Missing required input: {step_input.name}"
            
            if inputs[step_input.name].data_type != step_input.required_type:
                return False, (f"Input '{step_input.name}' has type "
                             f"{inputs[step_input.name].data_type.value}, "
                             f"expected {step_input.required_type.value}")
        
        return True, ""


class MergeStep(AnalysisStep):
    """Base class for merge operations that combine multiple branches."""
    
    @abstractmethod
    def get_merge_logic_description(self) -> str:
        """Describe what this merge does."""
        pass
    
    @abstractmethod
    def accepts_inputs(self, input_types: List[DataType]) -> Tuple[bool, str]:
        """Check if this merge can handle the given input types."""
        pass


@dataclass
class ProtocolNode:
    """A node in the protocol execution graph."""
    step: AnalysisStep
    node_id: str
    upstream_nodes: List[str] = field(default_factory=list)  # node IDs
    is_merge: bool = False
    
    def __hash__(self):
        return hash(self.node_id)


@dataclass
class ProtocolGraph:
    """Directed acyclic graph representing the analysis protocol."""
    nodes: Dict[str, ProtocolNode] = field(default_factory=dict)
    name: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: ProtocolNode) -> None:
        """Add a node to the graph."""
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists")
        self.nodes[node.node_id] = node
    
    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        """Add a directed edge from one node to another."""
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node {from_node_id} not found")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node {to_node_id} not found")
        
        self.nodes[to_node_id].upstream_nodes.append(from_node_id)
    
    def get_execution_order(self) -> List[str]:
        """Return topological sort of nodes for execution."""
        # Kahn's algorithm
        in_degree = {node_id: len(node.upstream_nodes) 
                     for node_id, node in self.nodes.items()}
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            for node_id, node in self.nodes.items():
                if current in node.upstream_nodes:
                    in_degree[node_id] -= 1
                    if in_degree[node_id] == 0:
                        queue.append(node_id)
        
        if len(order) != len(self.nodes):
            raise ValueError("Protocol contains a cycle - cannot execute")
        
        return order
    
    def validate_dag(self) -> Tuple[bool, str]:
        """Validate that the graph is a valid DAG."""
        try:
            self.get_execution_order()
            return True, ""
        except ValueError as e:
            return False, str(e)
    
    def validate_step_compatibility(self) -> Tuple[bool, List[str]]:
        """
        Validate that all steps in the chain are compatible.
        Returns (is_valid, list_of_warnings_and_errors)
        """
        issues = []
        execution_order = self.get_execution_order()
        
        # Track available data at each point
        available_outputs: Dict[str, List[StepOutput]] = {
            "input": [StepOutput("raw_data", DataType.RAW_EEG, "Input EEG data")]
        }
        
        for node_id in execution_order:
            node = self.nodes[node_id]
            step = node.step
            
            # Check that all inputs are available
            for step_input in step.get_inputs():
                found = False
                for prev_node_id in node.upstream_nodes:
                    prev_node = self.nodes[prev_node_id]
                    for output in prev_node.step.get_outputs():
                        if output.data_type == step_input.required_type:
                            found = True
                            break
                    if found:
                        break
                
                if not found and step_input.required_type != DataType.RAW_EEG:
                    issues.append(
                        f"Step '{step.name}' ({node_id}) requires input type "
                        f"{step_input.required_type.value}, but no upstream step produces it. "
                        f"Available: {[o.data_type.value for outputs in available_outputs.values() for o in outputs]}"
                    )
            
            # Record outputs from this step
            available_outputs[node_id] = step.get_outputs()
        
        return len(issues) == 0, issues


@dataclass
class ExecutionResult:
    """Result of executing a single step during protocol execution."""
    step_id: str
    step_name: str
    outputs: Dict[str, EEGSignal]
    success: bool
    error_message: str = ""
    execution_time: float = 0.0


@dataclass
class ProtocolExecutionTrace:
    """Complete trace of protocol execution for a single file."""
    file_name: str
    protocol_name: str
    success: bool
    results: List[ExecutionResult] = field(default_factory=list)
    final_outputs: Dict[str, EEGSignal] = field(default_factory=dict)
    error_message: str = ""
    total_execution_time: float = 0.0


@dataclass
class EEGData:
    """Raw EEG data loaded from file."""
    data: np.ndarray  # shape: (n_channels, n_samples)
    sfreq: float  # sampling frequency in Hz
    channels: List[str]
    filename: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_signal(self) -> EEGSignal:
        """Convert to standardized EEGSignal format."""
        return EEGSignal(
            data=self.data,
            sampling_rate=self.sfreq,
            channel_names=self.channels,
            data_type=DataType.RAW_EEG,
            metadata={**self.metadata, "source_file": self.filename}
        )


@dataclass
class ProtocolStep:
    """Step in an analysis protocol."""
    name: str
    step_type: str  # e.g., "notch_filter", "bandpass_filter", etc.
    parameters: Dict[str, Any]  # Parameters for this step
    input_requirements: Dict[str, str]  # {input_name: expected_type}
    output_name: str
    channel_specific: bool = False  # If True, step operates on each channel
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "step_type": self.step_type,
            "parameters": self.parameters,
            "input_requirements": self.input_requirements,
            "output_name": self.output_name,
            "channel_specific": self.channel_specific,
            "description": self.description
        }


@dataclass
class ProcessingProtocol:
    """A complete analysis protocol defined by user."""
    name: str
    description: str = ""
    version: str = "1.0"
    steps: List[ProtocolStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: ProtocolStep) -> None:
        """Add a step to the protocol."""
        self.steps.append(step)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata
        }


@dataclass
class AnalysisResult:
    """Result of running a protocol on data."""
    protocol_name: str
    file_name: str
    success: bool
    outputs: Dict[str, Any]  # output_name -> output_value
    error_message: str = ""
    execution_time: float = 0.0
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
