__version__ = '0.1.0'
__author__ = 'Mario Sieg'
__email__ = 'mario.sieg.64@gmail.com'
__author_email__ = 'mario.sieg.64@gmail.com'

from ._quant import Context, RoundMode, QuantConfig, quant_torch, quant_numpy, QuantDtype, compute_config_properties_from_data, compute_config_properties_from_data_torch, compute_config_properties_from_data_numpy

__all__ = ['Context', 'RoundMode', 'QuantConfig', 'quant_torch', 'quant_numpy', 'QuantDtype', 'compute_config_properties_from_data', 'compute_config_properties_from_data_torch', 'compute_config_properties_from_data_numpy']