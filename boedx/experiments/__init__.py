"""
BOEDX experiment definitions.

Each sub-module defines one concrete environment (subclass of
``GenericBankBOEDEnv``), its configuration dataclass, the list of
policy-state variants to compare, and a ``main()`` CLI entry point.

Available experiments
---------------------
- ``source_location``      — 2-D two-source localisation, optimised with SAC (RL)
- ``source_location_nes``  — same environment, optimised with OpenAI-ES (NES)
- ``prey_population``      — prey-population predator model, discrete 1-D action (RL)
"""
