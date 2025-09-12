# Controllers

Controllers contain the logic for action selection and value estimation. The following controllers have been implemented: 
- **PPOController:** an actor-critic controller with simple feed-forward networks
- **LSTMController:** an actor-critic controller with feature extraction and LSTM layers, followed by action and value heads


## Structure
All controllers are structured according to ``BaseController``, which guarantees compatibility with synchronous and asynchronous training pipelines. 