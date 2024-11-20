import asyncio
from lambda_function import async_handler

async def test_lambda():
    event = {}  # You can add test event data here if needed
    context = {}  # You can create a mock context object if needed
    result = await async_handler(event, context)
    print(result)

if __name__ == "__main__":
    asyncio.run(test_lambda())