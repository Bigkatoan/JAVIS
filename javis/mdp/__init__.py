"""JAVIS-specific MDP terms for the payload/balance task.

Split by manager, mirroring mjlab's own `tasks/*/mdp` layout:

- `events`     -- payload and component-mass domain randomization.
- `actions`    -- the ODrive PI velocity loop.
- `rewards`    -- balance terms that tolerate a leaning equilibrium.
- `observations` -- privileged mass state for the critic.
- `curriculums`  -- the difficulty ramp that widens the DR ranges.

Anything already robot-agnostic in mjlab (`mjlab.envs.mdp`,
`mjlab.tasks.velocity.mdp`) is imported directly by the task rather than
duplicated here.
"""
