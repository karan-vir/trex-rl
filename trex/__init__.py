from gymnasium.envs.registration import register

# Lets you write gym.make("TRex-v0") after `import trex`.
register(id="TRex-v0", entry_point="trex.env:TRexEnv")
