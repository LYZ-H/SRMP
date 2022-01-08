---
- nodes:
<<<nodes>>>
#- topo:
# The topology coordinator to generate communication 
# topology dynamically; period is the coordination time xx s
- coordinator:
  - {name: coordinator, host: localhost, port: 45000, period: 100}
# time to update topology, which will send to the workers by coordinator
- request_topo_period: 60
# The probability of initiating a fetch parameters request
- fetch_probability: 1

# The timeout value is used for flow-control
- timeout_ms: 2500

# Choose interpolation method: clock, loss or constant
- interpolation: constant

# Diverge models when loss is reaching the value specified here (use 0 to disable)
- divergence_threshold: 0.5

# Individual interpolation methods configuration:

- constant: { value: 0.5 }

- clock: 0

- loss: 0
