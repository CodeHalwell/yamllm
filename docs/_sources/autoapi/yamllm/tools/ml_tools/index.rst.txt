yamllm.tools.ml_tools
=====================

.. py:module:: yamllm.tools.ml_tools


Classes
-------

.. autoapisummary::

   yamllm.tools.ml_tools.DataLoader
   yamllm.tools.ml_tools.DataPreprocessor
   yamllm.tools.ml_tools.EDAAnalyzer
   yamllm.tools.ml_tools.ModelEvaluator
   yamllm.tools.ml_tools.ModelTrainer


Module Contents
---------------

.. py:class:: DataLoader

   Bases: :py:obj:`yamllm.tools.base.Tool`


   .. py:method:: execute(filepath: str, **kwargs) -> pandas.DataFrame


   .. py:attribute:: supported_formats
      :value: ['.csv', '.xlsx', '.json', '.parquet']



.. py:class:: DataPreprocessor

   Bases: :py:obj:`yamllm.tools.base.Tool`


   .. py:method:: execute(data: pandas.DataFrame, numeric_strategy: str = 'mean', categorical_strategy: str = 'mode', scale: bool = True) -> pandas.DataFrame


   .. py:attribute:: encoders


   .. py:attribute:: scalers


.. py:class:: EDAAnalyzer

   Bases: :py:obj:`yamllm.tools.base.Tool`


   .. py:method:: execute(data: pandas.DataFrame) -> Dict[str, Any]


.. py:class:: ModelEvaluator

   Bases: :py:obj:`yamllm.tools.base.Tool`


   .. py:method:: execute(y_true: numpy.ndarray, y_pred: numpy.ndarray, task_type: str = 'classification') -> Dict[str, float]


.. py:class:: ModelTrainer

   Bases: :py:obj:`yamllm.tools.base.Tool`


   .. py:method:: execute(data: pandas.DataFrame, target: str, model, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]


