import asyncio
from lambda_function import async_handler

async def test_lambda():
    event = {}
    context = {}
    result = await async_handler(event, context)
    print(result)

if __name__ == "__main__":
    asyncio.run(test_lambda())