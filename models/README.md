## Model definitions and helper functions

The model code is organized into
* base model (e.g. ResNet101 or I3D) stored under [bases/](bases/)
* model wrapper (e.g TSN or AsyncTF) that wraps the base model to add some functionality, stored under [wrappers/](wrappers/)
* criterion (e.g. SoftMax or AsyncTF) which implements the loss function that operates on the output of the wrapper, stored under criteria [criteria/](criteria/)
* layers (e.g. VerboseGradients or BalanceLabels), which includes additional layers. Stored under [layers/](layers/)

Please see the subdirectories for abstract classes and instruction how to extend any of those to add new bases/wrappers/criteria.

The codebase can then be instructed to use the new bases/wrappeers/criteria by just adding a new file to the right folder and using the name of the file (without extension) as along with the following command line arguments:

* --base
* --wrapper
* --criterion

Good luck!
