"""Factor plugins: one module per factor, each registering itself with
`src.features.registry.register_factor`. Every module here is imported by
`registry.discover()`, so adding a factor is adding a file (modules whose
names start with `_` are skipped, for shared helpers)."""
