import asyncio

import carb
from isaacsim.core.api import World


async def run():
    world = World.instance()
    if world is None:
        carb.log_info("Creating new world instance")
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
        await world.reset_async()
    else:
        # clear all existing tasks and callback
        carb.log_info("World exists. Reseting world")
        world.clear_all_callbacks()
        world.scene.clear()
        world.reset()
        carb.log_info("World reset done")

    await world.play_async()


asyncio.ensure_future(run())
