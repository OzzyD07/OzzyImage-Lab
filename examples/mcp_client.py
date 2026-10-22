"""Example: OZZY_URL=https://your-tunnel.ngrok-free.app OZZY_TOKEN=... python examples/mcp_client.py"""
import asyncio
import os

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main():
    url = os.environ['OZZY_URL'].rstrip('/') + '/mcp/'
    async with httpx.AsyncClient(headers={'Authorization': 'Bearer ' + os.environ['OZZY_TOKEN']}, timeout=120) as http:
        async with streamable_http_client(url, http_client=http) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print(await session.call_tool('capabilities', {}))
                result = await session.call_tool('generate_image', {'settings': {'prompt': 'A blue ceramic mug on a wooden table', 'seed': 42}})
                print(result)


if __name__ == '__main__':
    asyncio.run(main())
