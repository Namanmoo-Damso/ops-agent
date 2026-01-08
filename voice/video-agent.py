"""Video Monitoring Agent - Runs with port parameter."""
import asyncio
import os
import sys
from pathlib import Path

from livekit.agents import AgentServer, JobContext, cli


# Use port parameter (dev_default=0 means random port in dev mode)
server = AgentServer(port=8082)


@server.rtc_session(agent_name="video-agent")
async def entrypoint(ctx: JobContext):
    """Video agent entrypoint."""
    await ctx.connect()
    print(f"Video agent connected to room: {ctx.room.name}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    cli.run_app(server)
