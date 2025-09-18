# Vendored Code

## Fire

Taken from https://github.com/google/jax-md/blob/master/jax_md under an Apache 2.0 license

## Pymbar

Taken https://github.com/choderalab/pymbar/commit/683644086ca57c5077fb99ac85f623472764343f under a MIT license


### Modifications

* Removes `pymbar/tests` directory
* Removes `pymbar/__init__.py` import of timeseries module to avoid warnings
* Changes `force_no_jax=True` to avoid memory issues mentioned in https://github.com/choderalab/pymbar/issues/564
* Update imports to reflect vendoring
* Remove `numexpr` dependency, since only used to compute logsumexp